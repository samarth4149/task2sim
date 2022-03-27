import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('.'))
from collections import OrderedDict

import torch
import torch.distributed as dist

from timm import create_model
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict

from utils.data import create_loader
from data.prep_datasets import prep_datasets
from utils import ioutils
from utils.distributed import init_distributed_mode
from utils.fix_infiniband import fix_infiniband
from utils.misc import set_seed
from utils.model import freeze_backbone, add_downstream_modules, forward_downstream, set_head_training, \
    get_classifier_name
from classifier.config import ClassifierConfig

from classifier.eval import validate, extract_features, knn_classifier
from classifier.train_epoch import train_epoch
from utils.model import get_features
from tqdm import tqdm


def run_classifier(C_, args, backbone_state_dict=None, skip_distributed=False):
    torch.backends.cudnn.benchmark = True
    num_hosts = len(os.environ.get('LSB_MCPU_HOSTS', 'localhost cpus').split()) // 2
    if num_hosts > 1:
        fix_infiniband()

    if not skip_distributed:
        # For distributed training
        init_distributed_mode(args)

    # This part should only be done by rank 0 process
    if args.rank == 0:
        os.makedirs(C_.SAVE_DIR, exist_ok=True)
        wandb_dir = C_.SAVE_DIR
        if C_.LOG_LEVEL <= 1:
            silent = True
        else:
            silent = False
        wandb = ioutils.WandbWrapper(args.debug, silent)
        wandb.init(name=C_.EXPT_NAME or None, dir=wandb_dir,
                   config={**C_, **vars(args)}, reinit=True, project=C_.PROJECT)
    else:
        wandb = None  # Would not get used for a non-master process

    if C_.SEED is not None:
        set_seed(C_, args.rank)

    # val_dataset is None if SPLIT_FRAC=1
    train_dataset, val_dataset, test_dataset = prep_datasets(C_, args)
    num_classes = train_dataset.num_classes

    train_loader = create_loader(C_, args, train_dataset,
                                 is_training=True, pin_memory=True)

    # NOTE : in case of TRAIN_ONLY, test_dataset is train_dataset with no transforms
    if C_.SPLIT_FRAC and not C_.VAL_AS_TEST:
        val_loader = create_loader(C_, args, val_dataset,
                                     is_training=False, pin_memory=True)
    test_loader = create_loader(C_, args, test_dataset,
                               is_training=False, pin_memory=True)


    mixup_fn = None
    mixup_active = C_.MIXUP > 0 or C_.CUTMIX > 0. or C_.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=C_.MIXUP, cutmix_alpha=C_.CUTMIX, cutmix_minmax=C_.CUTMIX_MINMAX,
            prob=C_.MIXUP_PROB, switch_prob=C_.MIXUP_SWITCH_PROB, mode=C_.MIXUP_MODE,
            label_smoothing=C_.SMOOTHING, num_classes=num_classes)

    model = create_model(
        C_.MODEL, num_classes=num_classes,
        drop_rate=C_.DROP_RATE, drop_path_rate=C_.DROP_PATH_RATE)

    if C_.DOWNSTREAM_EVAL == 'linear' or C_.DOWNSTREAM_EVAL == 'finetune':
        add_downstream_modules(C_, model)
        if C_.DOWNSTREAM_EVAL == 'linear':
            freeze_backbone(model)
        model.forward = forward_downstream.__get__(model, torch.nn.Module)
    elif C_.DOWNSTREAM_EVAL == 'knn':
        model.forward = get_features.__get__(model, torch.nn.Module)

    if C_.DOWNSTREAM_EVAL == 'knn':
        pass
    elif C_.DOWNSTREAM_EVAL == 'linear':
        # So only batchnorm and dropout in the linear head are affected
        set_head_training(model, True)
    else:
        model.train()
    model.cuda()

    optimizer = create_optimizer_v2(
        model, C_.OPTIM, learning_rate=C_.LR, weight_decay=C_.WD, momentum=0.9)

    sched_args = ioutils.AttrDict(
        sched=C_.SCHED, epochs=C_.MAX_EPOCHS, decay_rate=0.1,
        min_lr=C_.MIN_LR, warmup_lr=C_.WARMUP_LR,
        warmup_epochs=C_.WARMUP_EPOCHS, cooldown_epochs=0,
        # used with MultiStepLR Scheduler
        decay_epochs=C_.DECAY_EPOCHS)
    lr_scheduler, _ = create_scheduler(
        args=sched_args, optimizer=optimizer)
    loss_scaler = NativeScaler()  # for mixed precision training

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.local_rank == 0 and C_.LOG_LEVEL >= 2:
        print('number of params:', n_parameters)

    if C_.RESUME:
        if C_.RESUME.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                C_.RESUME, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(C_.RESUME, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        if args.local_rank == 0:
            print('Model loaded from \'{}\''.format(C_.RESUME))
            if 'epoch' in checkpoint:
                print('Epochs trained : {}'.format(checkpoint['epoch']))

        if (not C_.EVAL_ONLY and 'optimizer' in checkpoint
                and 'epoch' in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            elif 'scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            if 'loss_scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['loss_scaler'])
            elif 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

            if args.local_rank == 0:
                print('Resuming from epoch : {}'.format(start_epoch))
    elif C_.BACKBONE_PATH or (backbone_state_dict is not None):
        if backbone_state_dict is not None:
            checkpoint = backbone_state_dict
        elif C_.BACKBONE_PATH.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                C_.BACKBONE_PATH, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(C_.BACKBONE_PATH, map_location='cpu')

        model_classifier_name = get_classifier_name(model)
        model_state_dict = model.state_dict()
        if 'model' in checkpoint:
            saved_wts = checkpoint['model']
        else:
            saved_wts = checkpoint # In case the statedict is directly stored
        for key in model_state_dict:
            # Do not load classifier or the state_dicts of additional modules like final batchnorm
            if key.startswith(model_classifier_name) or key not in saved_wts:
                pass
            else:
                model_state_dict[key] = saved_wts[key]
        model.load_state_dict(model_state_dict)

        if args.local_rank == 0:
            print('Backbone loaded from \'{}\''.format(C_.BACKBONE_PATH))
        start_epoch = 1
    else:
        start_epoch = 1

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank])

    if C_.DOWNSTREAM_EVAL == 'knn':
        # extract features and run knn_eval
        if args.local_rank == 0:
            print('Extracting Train features')
        train_features, train_labels = extract_features(args, model, train_loader)
        if args.local_rank == 0:
            print('Extracting Test features')
        test_features, test_labels = extract_features(args, model, test_loader)

        if args.rank == 0:
            train_features = torch.nn.functional.normalize(train_features, dim=1, p=2)
            test_features = torch.nn.functional.normalize(test_features, dim=1, p=2)
            train_labels = train_labels.long()
            test_labels = test_labels.long()

            log_dict = OrderedDict()
            for k in C_.NUM_NEIGHBORS:
                print('Computing Accuracy')
                top1_acc = knn_classifier(
                    train_features, train_labels, test_features, test_labels,
                    k=k, T=0.07, num_classes=num_classes)

                log_dict.update({
                    'K={} Top 1 Acc'.format(k): ioutils.FormattedLogItem(top1_acc, '{:.2f}')
                })
            print(ioutils.get_log_str(log_dict, title='KNN Eval Results'))
            wandb.log(ioutils.rm_format(log_dict))

        if args.distributed:
            dist.barrier()
        return {'Test Acc' : top1_acc}

    if C_.EVAL_ONLY:
        test_acc = validate(C_, args, model, test_loader)
        log_dict = {
            'Test Acc': ioutils.FormattedLogItem(100. * test_acc, '{:.2f}'),
        }
        if C_.SPLIT_FRAC and not C_.VAL_AS_TEST:
            val_acc = validate(C_, args, model, val_loader)
            log_dict.update(
                {'Val Acc': ioutils.FormattedLogItem(100. * val_acc, '{:.2f}')})
        if C_.LOG_LEVEL >= 1:
            print(ioutils.get_log_str(log_dict))
        if args.rank == 0:
            wandb.log(ioutils.rm_format(log_dict))
            wandb.join()

        return

    if C_.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif C_.SMOOTHING:
        criterion = LabelSmoothingCrossEntropy(smoothing=C_.SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if C_.SPLIT_FRAC and not C_.VAL_AS_TEST:
        best_acc = 0 # best validation accuracy

    best_test_acc = 0
    best_epoch = 0

    if C_.LOG_LEVEL == 1:
        pbar = tqdm(total=C_.MAX_EPOCHS+1-start_epoch)

    for epoch in range(start_epoch, C_.MAX_EPOCHS+1):
        log_dict = OrderedDict({'Epoch' : epoch})
        train_metrics = train_epoch(
            C_, args, epoch, model, train_loader, optimizer,
            lr_scheduler, criterion, loss_scaler, mixup_fn, wandb)
        log_dict.update(train_metrics)
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)


        if args.rank == 0 and C_.SAVE_MODEL:
            save_dict = {
                'epoch': epoch,
                'model': get_state_dict(model),
                'optimizer': optimizer.state_dict(),
                'criterion': criterion.state_dict(),
                'loss_scaler': loss_scaler.state_dict()
            }
            if lr_scheduler is not None:
                save_dict.update({'lr_scheduler': lr_scheduler.state_dict()})
            save_dict.update(train_metrics)
            torch.save(save_dict, os.path.join(C_.SAVE_DIR, 'checkpoint.pth.tar'))
            if epoch % C_.CKPT_SAVE_INTERVAL == 0:
                shutil.copyfile(
                    os.path.join(C_.SAVE_DIR, 'checkpoint.pth.tar'.format(epoch)),
                    os.path.join(C_.SAVE_DIR, 'checkpoint_epoch_{}.pth.tar'.format(epoch)))

        if epoch % C_.VAL_INTERVAL == 0:
            if not C_.TRAIN_ONLY:
                test_acc = validate(C_, args, model, test_loader)
                log_dict.update(OrderedDict({
                    'Epoch': epoch,
                    'Test Acc': ioutils.FormattedLogItem(100. * test_acc, '{:.2f}'),
                }))

                if C_.SPLIT_FRAC and not C_.VAL_AS_TEST:
                    val_acc = validate(C_, args, model, val_loader)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_test_acc = test_acc
                        best_epoch = epoch
                        if C_.SAVE_MODEL and args.rank == 0:
                            shutil.copyfile(
                                os.path.join(C_.SAVE_DIR, 'checkpoint.pth.tar'),
                                os.path.join(C_.SAVE_DIR, 'model-best.pth.tar')
                            )
                    log_dict.update({
                        'Val Acc': ioutils.FormattedLogItem(100. * val_acc, '{:.2f}'),
                        'Best Val Acc': ioutils.FormattedLogItem(100. * best_acc, '{:.2f}'),
                    })
                else:
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        best_epoch = epoch
                log_dict.update({
                    'Best Test Acc': ioutils.FormattedLogItem(100. * best_test_acc, '{:.2f}'),
                    'Best epoch': best_epoch
                })

            if args.rank == 0:
                if C_.LOG_LEVEL >= 1:
                    print(ioutils.get_log_str(log_dict))
                wandb.log(ioutils.rm_format(log_dict))

        if C_.LOG_LEVEL == 1:
            pbar.set_description(
                'Loss:{}, Train Acc:{}'.format(train_metrics['Epoch Loss'], train_metrics['Epoch Train Acc']))
            pbar.update(1)

    if C_.TRAIN_ONLY:
        # In this case, test_dataset is the same as train
        train_acc = validate(C_, args, model, test_loader)
        if args.rank == 0:
            print('Final Train Acc : {:.2f}'.format(100. * train_acc))
        wandb.log({'Final Train Acc' : 100. * train_acc})
        ret_dict = {'model_state_dict': model.state_dict()}
    else:
        ret_dict = {'Test Acc': test_acc}

    if args.rank == 0:
        wandb.join()

    return ret_dict

if __name__ == '__main__':
    args, unknown = ioutils.parse_known_args()
    if len(unknown) > 0:
        # This is needed to be able to pass arguments in the normal argparse way
        # for wandb sweeps
        override_list = args.cfg_override + ioutils.override_from_unknown(unknown)
    else:
        override_list = args.cfg_override
    C_ = ClassifierConfig(args.cfg_yml, override_list)

    run_classifier(C_, args)