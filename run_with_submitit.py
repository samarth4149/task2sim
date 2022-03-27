# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

from classifier import run_classifier as m
from classifier.config import ClassifierConfig
from utils import ioutils
import submitit


def parse_args(get_defaults=False):
    classification_parser = ioutils.get_parser()
    parser = argparse.ArgumentParser("Submitit for ViT-TDW", parents=[classification_parser])
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=360, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument('--long_job', default=False, action='store_true', help='Requests 48hr job')
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')

    if get_defaults:
        args = parser.parse_args([])
        C_ = ClassifierConfig()
    else:
        args, unknown = parser.parse_known_args()
        if len(unknown) > 0:
            # This is needed to be able to pass arguments in the normal argparse way
            # for wandb sweeps
            override_list = args.cfg_override + ioutils.override_from_unknown(unknown)
        else:
            override_list = args.cfg_override
        C_ = ClassifierConfig(args.cfg_yml, override_list)

    return C_, args


def get_init_file(root):
    # Init file must not exist, but it's parent dir must exist.
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    init_file = Path(root) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, cfg, args):
        self.args = args
        self.cfg = cfg

    def __call__(self):
        self._setup_gpu_args()
        m.run_classifier(self.cfg, self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(root='expts/dist_init_files').as_uri()
        checkpoint_file = os.path.join(self.cfg.SAVE_DIR, "checkpoint.pth.tar")
        if os.path.exists(checkpoint_file):
            self.cfg.defrost()
            self.cfg.RESUME = checkpoint_file
            self.cfg.freeze()
        else:
            raise Exception('No checkpoint to resume from')

        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.cfg, self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        # self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.local_rank = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main1(C_, args):
    if args.job_dir == "":
        args.job_dir = os.path.join(C_.SAVE_DIR, 'slurm')

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    kwargs = {}
    if args.comment:
        kwargs['slurm_comment'] = args.comment


    addnl_params = {
        'gres': 'gpu:{:d}'.format(num_gpus_per_node)
    }
    if args.long_job:
        addnl_params['qos'] = 'dcs-48hr'

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=6,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_signal_delay_s=120,
        slurm_additional_parameters=addnl_params,
        **kwargs
    )

    executor.update_parameters(name=C_.EXPT_NAME)

    args.dist_url = get_init_file(root='expts/dist_init_files').as_uri()

    trainer = Trainer(C_, args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)

def main():
    C_, args = parse_args()
    main1(C_, args)

if __name__ == "__main__":
    main()
