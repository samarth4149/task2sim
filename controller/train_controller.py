import os
import shutil
import sys
sys.path.insert(0, os.path.abspath('.')) # paths from base are prioritized
import torch
from utils import ioutils
from controller.config import ControllerConfig
from controller.model import ControllerModel
from controller.task_db import load_task_db
from controller.task_db import get_task_dim
from generator.generator_params import GenParams
import numpy as np
from collections import OrderedDict
from controller.utils import get_argmax_gen_params_from_outputs
from controller.utils import sample_gen_params_from_outputs_softmax
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from argparse import Namespace
from utils.misc import AverageMeter, set_seed
from timm.scheduler import create_scheduler
import json
import wandb as wb
from pathlib import Path
import copy

def eval(cont_model, task_list, cont_out_params, task_db, targets, reward_db):
    cont_model.eval()
    task_vecs = torch.stack([torch.tensor(task_db[k].get_hessian()) for k in task_list]).cuda()
    task_vecs = F.normalize(task_vecs)

    with torch.no_grad():
        cont_outs = cont_model(task_vecs)

    idx_targets = copy.deepcopy(targets)
    for i, tn in enumerate(task_list):
        idx_targets[tn]['materials'] = idx_targets[tn]['materials']//2

    conf_mat = np.array(
        [[cont_outs[k][i, idx_targets[tn][k]].cpu().numpy()
          for k in cont_out_params]
         for i, tn in enumerate(task_list)])

    soft_pred_mat = np.array(
        [[cont_outs[k][i, 1].cpu().numpy()
          for k in cont_out_params]
         for i, tn in enumerate(task_list)])

    gt_mat = np.array([[targets[tn][k] for k in cont_out_params] for tn in task_list])

    max_reward = np.mean([max(reward_db[tn].values()) for tn in task_list])

    pred_gps = get_argmax_gen_params_from_outputs(len(task_list), cont_outs)
    acc_mat = np.array([[pred_gps[i][k] == targets[tn][k] for k in cont_out_params] for i, tn in enumerate(task_list)], dtype=float)
    pred_mat = np.array([[pred_gps[i][k] for k in cont_out_params] for i in range(len(task_list))])

    if 'materials' in cont_out_params:
        mat_loc = cont_out_params.index('materials')
        gt_mat[:, mat_loc] = np.floor_divide(gt_mat[:, mat_loc], 2)
        pred_mat[:, mat_loc] = np.floor_divide(pred_mat[:, mat_loc], 2)

    reward = np.mean([reward_db[tn][gp.get_tuple_str()] for (gp, tn) in zip(pred_gps, task_list)])

    avg_hamming_dist = 0
    for i, gp1 in enumerate(pred_gps):
        for j, gp2 in enumerate(pred_gps[i + 1:]):
            avg_hamming_dist += (np.array(gp1.get_tuple()) != np.array(gp2.get_tuple())).sum()
    avg_hamming_dist = (2 * avg_hamming_dist) / float(len(pred_gps) * (len(pred_gps) - 1))

    cont_model.train()

    return (conf_mat, gt_mat, acc_mat, pred_mat, soft_pred_mat), max_reward, reward, avg_hamming_dist, pred_gps

def get_dist_wtd_preds(cont_model, train_task_list, test_task_list, train_task_db, test_task_db, cont_out_params):
    train_task_vecs = torch.stack([torch.tensor(train_task_db[k].get_hessian()) for k in train_task_list]).cuda()
    train_task_vecs = F.normalize(train_task_vecs, dim=1)
    test_task_vecs = torch.stack([torch.tensor(test_task_db[k].get_hessian()) for k in test_task_list]).cuda()
    test_task_vecs = F.normalize(test_task_vecs, dim=1)

    sim_mat = torch.mm(test_task_vecs, train_task_vecs.t())
    sim_mat = sim_mat.div(sim_mat.sum(dim=1, keepdim=True))

    with torch.no_grad():
        train_cont_outs = cont_model(train_task_vecs)

    test_outs = {k:torch.mm(sim_mat, train_cont_outs[k]) for k in train_cont_outs}
    pred_mat = np.array([[test_outs[k][i, 1].cpu().numpy() for k in cont_out_params] for i in range(len(test_task_list))])

    return pred_mat


def train_controller(C_:ControllerConfig, args:Namespace):
    if C_.SEED is not None:
        set_seed(C_, 0)

    os.makedirs(C_.SAVE_DIR, exist_ok=True)

    wandb = ioutils.WandbWrapper(debug=args.debug, write_to_disk=False)
    wandb_dir = C_.SAVE_DIR
    wandb.init(name=C_.EXPT_NAME or None, dir=wandb_dir,
               config=C_, reinit=True, project=C_.PROJECT)

    # Prepare dataset of tasks vectors (task2vec)
    task_db = load_task_db(C_.TASK_DB_PATHS)
    train_task_db = {k: v for k, v in task_db.items() if k in C_.TRAIN_TASKS}
    test_task_db = {k: v for k, v in task_db.items() if k in C_.TEST_TASKS}

    # Load database of pre-computed rewards
    with open(C_.REWARD_DB_PATH, 'r') as f:
        reward_db = json.load(f)

    train_task_db_iter = list(sorted(train_task_db.items()))
    task_dim = get_task_dim(train_task_db)

    # get dimensions for output heads
    param_ranges = GenParams.get_ranges()  # dict {head_param:range}
    param_ranges.materials = 2 # Set separately to limit to 2 options
    head_out_dims = OrderedDict({k: v for k, v in sorted(param_ranges.items()) if k in C_.CONT_OUT_PARAMS})

    # initialize appropriately (get dimensions from task2vec computed)
    cont_model = ControllerModel(task_dim, C_.NUM_HIDDEN_NODES, head_out_dims)

    if C_.MODEL_LOAD_PATH:
        ckpt = torch.load(C_.MODEL_LOAD_PATH)
        cont_model.load_state_dict(ckpt['cont_state_dict'])

    cont_model.train()
    cont_model.cuda()

    if C_.EVAL_ONLY:
        train_targets = {k: GenParams.from_tuple_str(max(reward_db[k], key=reward_db[k].get)) for k in
                         C_.TRAIN_TASKS}
        test_targets = {k: GenParams.from_tuple_str(max(reward_db[k], key=reward_db[k].get)) for k in
                        C_.TEST_TASKS}
        train_mats, train_max_reward, train_avg_reward, train_avg_hamming_dist, _ = \
            eval(cont_model, C_.TRAIN_TASKS, C_.CONT_OUT_PARAMS, train_task_db, train_targets, reward_db)
        test_mats, test_max_reward, test_avg_reward, test_avg_hamming_dist, _ = \
            eval(cont_model, C_.TEST_TASKS, C_.CONT_OUT_PARAMS, test_task_db, test_targets, reward_db)

        dist_wtd_pred_mat = get_dist_wtd_preds(cont_model, C_.TRAIN_TASKS, C_.TEST_TASKS, train_task_db, test_task_db, C_.CONT_OUT_PARAMS)
        log_info = {
            'Train Max Avg Reward' : train_max_reward,
            'Test Max Avg Reward' : test_max_reward,
            'Train Avg Reward' : train_avg_reward,
            'Test Avg Reward' : test_avg_reward,
            'Train Acc': 100. * train_mats[2].mean(),
            'Test Acc': 100. * test_mats[2].mean(),
        }
        log_str = ioutils.get_log_str(log_info, title='Evaluation Results')
        print(log_str)
        log_info.update({
            'Train Conf Mat' : wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TRAIN_TASKS, train_mats[0]),
            'Test Conf Mat' : wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TEST_TASKS, test_mats[0]),
            'Train GT Mat': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TRAIN_TASKS, train_mats[1]),
            'Test GT Mat': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TEST_TASKS, test_mats[1]),
            'Train Acc Mat': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TRAIN_TASKS, train_mats[2]),
            'Test Acc Mat': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TEST_TASKS, test_mats[2]),
            'Train Preds': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TRAIN_TASKS, train_mats[3]),
            'Test Preds': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TEST_TASKS, test_mats[3]),
            'Train Soft Pred Mat': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TRAIN_TASKS, train_mats[4]),
            'Test Soft Pred Mat' : wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TEST_TASKS, test_mats[4]),
            'Test Dist wtd. Pred Mat' : wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TEST_TASKS, dist_wtd_pred_mat),
        })
        wandb.log(log_info)
        return

    # optimizer
    optimizer = optim.SGD(cont_model.parameters(), momentum=0.9, lr=C_.LR, weight_decay=C_.WD)

    sched_args = ioutils.AttrDict(
        sched='step', epochs=C_.NUM_EPOCHS, decay_rate=C_.LR_DECAY_GAMMA,
        min_lr=1e-8, warmup_lr=1e-6, warmup_epochs=C_.WARMUP_EPOCHS,
        cooldown_epochs=0, decay_epochs=C_.LR_DECAY_EPOCH)
    scheduler, _ = create_scheduler(args=sched_args, optimizer=optimizer)

    if C_.USE_PBAR:
        pbar = tqdm(total=C_.NUM_EPOCHS)

    loss_meter = AverageMeter()
    average_reward = AverageMeter()
    replay_mem = {tn:{} for tn in C_.TRAIN_TASKS}

    # Training loop
    for epoch in range(C_.NUM_EPOCHS):
        train_rewards = {}
        task_order = np.random.permutation(len(train_task_db))
        average_reward.reset()

        for step in range(int(np.ceil(len(train_task_db) / C_.BATCH_SIZE))):
            batch = [train_task_db_iter[idx] for idx in task_order[step * C_.BATCH_SIZE:(step + 1) * C_.BATCH_SIZE]]
            task_names = [b[0] for b in batch]

            # NOTE : the get_hessian function by default gets scaled hessian
            task_vecs = torch.stack([torch.tensor(b[1].get_hessian()) for b in batch]).cuda()
            task_vecs = F.normalize(task_vecs, dim=1)

            if epoch >= C_.IMITATION_START_EPOCH:
                sub_step_nums = C_.NUM_REPLAY_STEPS + 1
            else:
                sub_step_nums = 1

            for sub_step in range(sub_step_nums):
                # get generator param outputs from controller
                cont_outs = cont_model(task_vecs)  # This is a dict of outputs of all heads
                pred_gps = get_argmax_gen_params_from_outputs(len(task_vecs), cont_outs)

                if sub_step == 0:
                    if C_.EXPL_NOISE > 0.:
                        eps = C_.EXPL_NOISE / np.ceil((epoch + 1) / C_.NOISE_RED_EPOCH)
                        # Add noise to controller outputs
                        for k in cont_outs:
                            with torch.no_grad():
                                eps_vec = eps / (cont_outs[k].shape[1] - 1) * torch.ones(cont_outs[k].shape[0], 1,
                                                                                         device=cont_outs[k].device)
                                eps_vec = eps_vec.expand_as(cont_outs[k]).clone()
                                max_idxs = cont_outs[k].argmax(1)
                                eps_vec[torch.arange(len(max_idxs)), max_idxs] -= (eps + eps / (cont_outs[k].shape[1] - 1))
                            cont_outs[k] = cont_outs[k] + eps_vec

                if sub_step == 0:
                    # sample from this distribution
                    gps = sample_gen_params_from_outputs_softmax(
                        len(batch), cont_outs)
                else:
                    # Get best parameters seen for self imitation
                    gps = []
                    for i, tn in enumerate(task_names):
                        gps.append(GenParams.from_tuple_str(max(replay_mem[tn], key=replay_mem[tn].get)))

                gps_collated = {
                    k: torch.tensor([gp[k] for gp in gps], device=cont_outs[k].device) for k in cont_outs}

                rewards = []
                for tn, gp in zip(task_names, gps):
                    rewards.append(reward_db[tn][gp.get_tuple_str()])
                    replay_mem[tn][gp.get_tuple_str()] = reward_db[tn][gp.get_tuple_str()]

                baseline_rewards = []
                for tn, gp in zip(task_names, pred_gps):
                    r = reward_db[tn][gp.get_tuple_str()]
                    baseline_rewards.append(r)
                    train_rewards[tn] = r
                    replay_mem[tn][gp.get_tuple_str()] = reward_db[tn][gp.get_tuple_str()]

                average_reward.update(np.mean(rewards), len(rewards))

                neg_log_probs = 0
                for k in cont_outs:
                    if k == 'materials':
                        neg_log_probs = neg_log_probs - torch.log(cont_outs[k][:, gps_collated[k].div(2, rounding_mode='floor')]) # mapping 1,3 to 0,1
                    else:
                        neg_log_probs = neg_log_probs - torch.log(cont_outs[k][:, gps_collated[k]])
                advantages = (torch.FloatTensor(rewards) - torch.FloatTensor(baseline_rewards)).to(neg_log_probs.device)

                reinforce_loss = (advantages * neg_log_probs).mean()

                loss = reinforce_loss
                with torch.no_grad():
                    loss_meter.update(loss.item(), len(task_vecs))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if C_.USE_PBAR:
                pbar.set_description('Loss:{}'.format(loss.item()), refresh=True)

        scheduler.step(epoch)
        replay_max = {tn + ' replay max':max(list(replay_mem[tn].values())) for tn in C_.TRAIN_TASKS}

        save_dict = {
            'cont_state_dict': cont_model.state_dict(),
        }

        log_dict = {
            'Epoch': epoch + 1,
            'Loss': loss_meter.avg,
            'Average Reward': average_reward.avg,
            'Avg Train Reward (intermediate)' : np.mean(list(train_rewards.values())),
            'Avg Replay Max' : np.mean(list(replay_max.values())),
        }

        wandb.log(log_dict)
        torch.save(
            save_dict, os.path.join(C_.SAVE_DIR, 'checkpoint.pth.tar'))

        if C_.CKPT_SAVE_INTERVAL and (epoch+1)%C_.CKPT_SAVE_INTERVAL == 0:
            shutil.copyfile(
                os.path.join(C_.SAVE_DIR, 'checkpoint.pth.tar'),
                os.path.join(C_.SAVE_DIR, 'checkpoint_epoch_{:04d}.pth.tar'.format(epoch)))

        if C_.USE_PBAR:
            pbar.update(1)

    # Final eval
    train_targets = {k: GenParams.from_tuple_str(max(reward_db[k], key=reward_db[k].get)) for k in
                     C_.TRAIN_TASKS}
    test_targets = {k: GenParams.from_tuple_str(max(reward_db[k], key=reward_db[k].get)) for k in
                    C_.TEST_TASKS}
    train_mats, train_max_reward, train_avg_reward, train_avg_hamming_dist, final_train_preds = \
        eval(cont_model, C_.TRAIN_TASKS, C_.CONT_OUT_PARAMS, train_task_db, train_targets, reward_db)
    test_mats, test_max_reward, test_avg_reward, test_avg_hamming_dist, final_test_preds = \
        eval(cont_model, C_.TEST_TASKS, C_.CONT_OUT_PARAMS, test_task_db, test_targets, reward_db)
    log_info = {
        'Train Max Avg Reward': train_max_reward,
        'Test Max Avg Reward': test_max_reward,
        'Train Avg Reward': train_avg_reward,
        'Test Avg Reward': test_avg_reward,
        'Train Avg Hamming Dist': train_avg_hamming_dist,
        'Test Avg Hamming Dist': test_avg_hamming_dist,
        'Train Acc': 100. * train_mats[2].mean(),
        'Test Acc': 100. * test_mats[2].mean(),
    }
    log_str = ioutils.get_log_str(log_info, title='Evaluation Results')
    print(log_str)
    log_info.update({
        'Train Conf Mat': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TRAIN_TASKS, train_mats[0]),
        'Test Conf Mat': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TEST_TASKS, test_mats[0]),
        'Train GT Mat': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TRAIN_TASKS, train_mats[1]),
        'Test GT Mat': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TEST_TASKS, test_mats[1]),
        'Train Acc Mat': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TRAIN_TASKS, train_mats[2]),
        'Test Acc Mat': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TEST_TASKS, test_mats[2]),
        'Train Preds': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TRAIN_TASKS, train_mats[3]),
        'Test Preds': wb.plots.HeatMap(C_.CONT_OUT_PARAMS, C_.TEST_TASKS, test_mats[3]),
    })

    wandb.log(log_info)

    final_train_preds = {k:final_train_preds[i].get_tuple() for i, k in enumerate(C_.TRAIN_TASKS)}
    final_test_preds = {k:final_test_preds[i].get_tuple() for i, k in enumerate(C_.TEST_TASKS)}
    with open(Path(C_.SAVE_DIR) / 'final_train_preds.json', 'w') as f:
        json.dump(final_train_preds, f, indent=4)
    with open(Path(C_.SAVE_DIR) / 'final_test_preds.json', 'w') as f:
        json.dump(final_test_preds, f, indent=4)
    wandb.join()


if __name__ == '__main__':
    args, unknown = ioutils.parse_known_args()
    if len(unknown) > 0:
        # This is needed to be able to pass arguments in the normal argparse way
        # for wandb sweeps
        override_list = args.cfg_override + ioutils.override_from_unknown(unknown)
    else:
        override_list = args.cfg_override
    C_ = ControllerConfig(args.cfg_yml, override_list)

    train_controller(C_, args)