import argparse
import math
import os
import random
import string
from datetime import datetime

import git
import wandb
import yaml
from pathlib import Path
import json


def get_parser():
    parser = argparse.ArgumentParser(description='ViT-TDW', add_help=False)
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='No wandb logging if true')
    parser.add_argument(
        '--quick', action='store_true', default=False,
        help='Reduces dataset size for a quick run')
    parser.add_argument(
        '--local_rank', default=0, type=int,
        help='Used for distributed training')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # NOTE : world_size is mostly set by distributed launching utilities/scripts
    # It needs to be set if command is being launched in slrum without run_with_submitit
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument(
        '--cfg-yml', default='',
        help='Path to a config file for a training job'
    )
    parser.add_argument(
        '--cfg-override',
        default=[],
        nargs='*',
        help='A sequence of key-value pairs specifying certain config arguments. '
             'The actual config will be updated and recorded in the serialization directory.',
    )
    return parser

def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args

def parse_known_args():
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    return args, unknown

def override_from_unknown(unknown):
    ret = []
    for s in unknown:
        if s.startswith('--'):
            s = s[2:]
            if '=' in s:
                ret += s.split('=')
            else:
                ret.append(s)
        else:
            ret.append(s)
    return ret

def get_default_args():
    parser = get_parser()
    args = parser.parse_args([])
    return args

def boolfromstr(s):
    if s.lower().startswith('true'):
        return True
    elif s.lower().startswith('false'):
        return False
    else:
        raise Exception('Incorrect option passed for a boolean')

def get_sha():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def save_expt_info(args, sha):
    if isinstance(args, dict):
        # in case a dictionary or attrdict is used for args
        save_args = dict(args)
    else:
        save_args = vars(args)

    info = {
        'args': save_args,
        'githash': sha
    }
    with open(os.path.join(args.save_dir, 'info.yaml'), 'w') as outfile:
        yaml.dump(info, outfile)

class FormattedLogItem:
    def __init__(self, item, fmt):
        self.item = item
        self.fmt = fmt
    def __str__(self):
        return self.fmt.format(self.item)

def rm_format(dict_obj):
    ret = dict_obj
    for key in ret:
        if isinstance(ret[key], FormattedLogItem):
            ret[key] = ret[key].item
    return ret

def get_log_str(log_info, title='Expt Log', sep_ch='-'):
    now = str(datetime.now().strftime('%H:%M %d-%m-%Y'))
    log_str = (sep_ch * math.ceil((80 - len(title))/2.) + title
               + (sep_ch * ((80 - len(title))//2)) + '\n')
    log_str += '{:<25} : {}\n'.format('Time', now)
    for key in log_info:
        log_str += '{:<25} : {}\n'.format(key, log_info[key])
    log_str += sep_ch * 80
    return log_str

def write_to_log(args, log_str, mode='a+'):
    with open(os.path.join(args.save_dir, 'log.txt'), mode) as outfile:
        print(log_str, file=outfile)

def gen_unique_name(length=4):
    '''
    Returns a string of 'length' lowercase letters
    '''
    return ''.join([random.choice(
        string.ascii_lowercase) for i in range(length)])


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class WandbWrapper():
    def __init__(self, debug=False, silent=False, write_to_disk=True):
        self.debug = debug
        self.num_tries = 10
        self.write_to_disk = write_to_disk
        if debug and not silent:
            print('Wandb Wrapper : debug mode. No logging with wandb')

    def init(self, *args, **kwargs):
        self.log_file = Path(kwargs['dir']) / 'log.json'
        if not isinstance(kwargs['config'], dict):
            save_config = vars(kwargs['config'])
        else:
            save_config = kwargs['config']
        self.log_data = {'config' : save_config, 'history': []}
        self.num_logs = 0
        if not self.debug:
            init_tries = 0
            while True:
                try:
                    wandb.init(*args, **kwargs)
                    break
                except Exception as e:
                    print('[Trial:{}] wandb could not init : {}'.format(init_tries, e))
                    if init_tries > self.num_tries:
                        wandb.alert(
                            'Expt name \'{}\' : Could not init in {} attempts'.format(
                                kwargs['name'], self.num_tries))
                    init_tries += 1
            self.run = wandb.run
        else:
            self.run = AttrDict({'dir' : kwargs['dir']})

    def commit_to_disk(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=4)

    def log(self, *args, **kwargs):
        if 'commit' in kwargs and kwargs['commit'] == False:
            self.log_data['history'][-1].update(args[0])
        else:
            self.log_data['history'].append(args[0])
        if not self.debug:
            wandb.log(*args, **kwargs)
        self.num_logs += 1

        if self.write_to_disk and self.num_logs%10 == 0:
            self.commit_to_disk()

    def join(self, *args, **kwargs):
        if self.write_to_disk:
            self.commit_to_disk()
        if not self.debug:
            wandb.join(*args, **kwargs)
