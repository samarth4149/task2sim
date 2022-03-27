from collections import OrderedDict

import torch
import torch.nn.parallel
# from apex.optimizers.fused_lamb import FusedLAMB
from timm.utils.distributed import reduce_tensor
from torch.cuda.amp import autocast
from tqdm import tqdm

from utils import ioutils
from utils.misc import AverageMeter
from utils.misc import MetricLogger


def train_epoch(C_, args, epoch, model, loader, optimizer,
                lr_scheduler, criterion, loss_scaler, mixup_fn, wandb):
    if args.local_rank == 0 and C_.LOG_LEVEL >= 2:
        pbar = tqdm(total=len(loader))
        pbar.set_description('Ep:{:03d}'.format(epoch))

    num_updates = (epoch-1) * len(loader)

    loss_meter_s = AverageMeter()
    train_acc_meter_s = AverageMeter()
    loss_meter_l = AverageMeter()
    train_acc_meter_l = AverageMeter()
    metric_logger = MetricLogger(delimiter='  ')

    if C_.LOG_LEVEL >= 3:
        header = 'Epoch: [{}/{}]'.format(epoch, C_.MAX_EPOCHS)
        iterator = enumerate(metric_logger.log_every(loader, C_.LOG_INTERVAL, header))
    else:
        iterator = enumerate(loader)

    for batch_idx, batch in iterator:
        img, tgt = batch
        img = img.cuda(non_blocking=True)
        tgt = tgt.cuda(non_blocking=True)
        if mixup_fn is not None:
            img, tgt_tr = mixup_fn(img, tgt)
        else:
            tgt_tr = tgt

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(img)
            loss = criterion(logits, tgt_tr)
            curr_train_acc = (logits.argmax(dim=1) == tgt).sum()/float(len(tgt))

        loss_scaler(
            loss, optimizer,
            clip_grad=C_.GRAD_CLIP, clip_mode='norm',
            parameters=model.parameters(),
            create_graph=False)

        num_updates += 1

        lr_scheduler.step_update(num_updates)

        # NOTE: uncomment below to use warmup in steps but decay in epochs
        # if num_updates == C_.WARMUP_STEPS:
        #     lr_scheduler.t_initial = lr_scheduler.t_initial//len(loader)
        #     lr_scheduler.warmup_t = lr_scheduler.warmup_t//len(loader)
        #     lr_scheduler.t_in_epochs = True

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            reduced_train_acc = reduce_tensor(curr_train_acc.data, args.world_size)
        else:
            reduced_loss = loss
            reduced_train_acc = curr_train_acc

        loss_meter_s.update(reduced_loss.item(), len(tgt))
        train_acc_meter_s.update(reduced_train_acc.item(), len(tgt))
        loss_meter_l.update(reduced_loss.item(), len(tgt))
        train_acc_meter_l.update(reduced_train_acc.item(), len(tgt))

        torch.cuda.synchronize()

        if args.local_rank == 0:
            if (batch_idx + 1) % C_.LOG_INTERVAL == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                log_dict = OrderedDict({
                    'Loss' : loss_meter_s.avg,
                    'Train Acc' : 100. * train_acc_meter_s.avg,
                    'LR' : lr
                })
                if C_.LOG_LEVEL >= 2:
                    pbar.set_description(
                        'Ep:{:03d}, Loss:{:.4f}, Tr Acc:{:.2f}'.format(
                            epoch, log_dict['Loss'], log_dict['Train Acc']),
                        refresh=True)
                if args.rank == 0:
                    wandb.log(log_dict)
                    if C_.LOG_LEVEL == 4:
                        log_str = ioutils.get_log_str(log_dict, title='Intra Epoch Log')
                        print(log_str)
            loss_meter_s.reset()
            train_acc_meter_s.reset()

            if C_.LOG_LEVEL >= 2:
                pbar.update(1)

    lrl = [param_group['lr'] for param_group in optimizer.param_groups]
    lr = sum(lrl) / len(lrl)
    return OrderedDict({
        'Epoch Lr' : ioutils.FormattedLogItem(lr, '{:.6f}'),
        'Epoch Loss' : ioutils.FormattedLogItem(loss_meter_l.avg, '{:.6f}'),
        'Epoch Train Acc' : ioutils.FormattedLogItem(100. * train_acc_meter_l.avg, '{:.2f}')
    })


