'''
A module for package-wide configuration
management. Inspired by Ross Girchick's yacs template
Also,
kd's source -- https://github.com/kdexd/probnmn-clevr/blob/master/probnmn/config.py
'''
import os
from typing import List, Any
from yacs.config import CfgNode as CN


class ClassifierConfig(CN):
    '''
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be over-written by (first) through a YAML file and (second) through
    a list of attributes and values.

    - This class definition contains details relevant to all the training phases
        but is listed as the final adaptation training phase

    Parameters
    ===========
    config_yaml
        Path to a YAML file containing configuration parameters to override
    config_override: List[Any], optional (default=[])
        A list of sequential attributes and values of parameters to override. This happens
        after overriding from YAML file.

    '''
    def __init__(self, config_yaml='',
                 config_override: List[Any] = []):
        # HACKY ALERT : making this work for functions in yacs calling the
        # config node init with a dict using cls(). E.g. _load_cfg_from_yaml_str
        if isinstance(config_yaml, dict):
            super(ClassifierConfig, self).__init__(config_yaml)
            return
        else:
            super(ClassifierConfig, self).__init__()

        self.SEED = None
        self.EXPT_NAME = ''
        self.NOTES = '' # Notes for wandb run
        self.PROJECT = 'vit-tdw'
        self.SAVE_DIR = 'expts/tmp_last'
        self.SAVE_MODEL = True # If false, does not save model
        self.DATA_DIR = ''
        self.SUBSET_PATH = '' # path to an npy file with list of indices for making a subset dataset
        self.TEST_SUBSET_PATH = '' # path to an npy file with list of indices for making a subset dataset
        self.RESUME = ''  # path to model checkpoint to resume from
        self.BACKBONE_PATH = ''

        # log levels
        # 0 : No printed output
        # 1 : tqdm over epochs
        # 2 : tqdm for each epoch
        # 3 : metric logger outputs for each epoch
        # 4 : Print expt log within each train epoch
        self.LOG_LEVEL = 3

        self.DATASET = '' # set to name of dataset for downstream tasks

        self.TRAIN_ONLY = False
        self.SPLIT_FRAC = None # If set, only use a fraction for training and set aside rest for validation
        self.VAL_AS_TEST = False # If true, validation split would be used in the test_dataset
        self.EVAL_ONLY = False  # Will only do the adapt and eval part if this is True

        # Augmentation options
        self.SMOOTHING = 0.1 # Label smoothing
        self.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
        self.COLOR_JITTER = 0.4 # color jitter factor
        self.MIXUP = 0.8 # mixup alpha
        self.CUTMIX = 1.
        self.INTERPOLATION = 'bicubic'
        # two ratios in [0,1] which overrides lambda in cropping.
        # Random crop made between min ratio and max ratio
        self.CUTMIX_MINMAX = None
        # Probability of performing mixup or cutmix when either/both is enabled
        self.MIXUP_PROB = 1.
        # Probability of switching to cutmix when both mixup and cutmix enabled
        self.MIXUP_SWITCH_PROB = 0.5
        # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
        self.MIXUP_MODE = 'batch'

        # Model options
        self.MODEL = 'resnet50'
        self.DROP_RATE = 0.  # dropout rate
        self.DROP_PATH_RATE = 0.1

        # Optimizer options
        self.OPTIM = 'adamw'
        self.LR = 1e-3 # learning rate
        self.WD = 0.05 # weight decay

        # Scheduler options
        self.SCHED = 'cosine'
        self.MIN_LR = 1e-5 # minimum lr which the scheduler ends up at
        self.WARMUP_EPOCHS = 5 # number of steps for linear learning rate warmup
        self.WARMUP_LR = 1e-6 # lr from which warmup starts
        self.DECAY_EPOCHS = [25, 37] # used for multi step lr schedule

        # Other training options
        # if it is a numeric value that is not None, gradient clipping done
        # at global l2 norm equaling it
        self.GRAD_CLIP = None

        self.BATCH_SIZE = 64  # In the distributed training, this is the batch size per process
        self.MAX_EPOCHS = 10
        self.LOG_INTERVAL = 25 # This is in number of steps
        self.VAL_INTERVAL = 1 # This is in epochs
        self.CKPT_SAVE_INTERVAL = 100 # Save a different checkpoint every K epochs

        self.NUM_WORKERS = 3

        # For downstream evaluation
        self.DOWNSTREAM_EVAL = ''  # options ''/'linear'/'finetune'/'knn'/'task2vec'
        self.DOWNSTREAM_DROP = 0.2 # dropout for downstream evaluation
        self.NUM_NEIGHBORS = [] # used for 'knn' eval

        # Override parameter values from YAML file first, then from override list
        if config_yaml:
            self.merge_from_file(config_yaml)
        self.merge_from_list(config_override)

        self.freeze()


