from typing import List, Any

from yacs.config import CfgNode as CN


class ControllerConfig(CN):
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
            super(ControllerConfig, self).__init__(config_yaml)
            return
        else:
            super(ControllerConfig, self).__init__()

        self.USE_PBAR = True
        self.SEED = 2

        self.PROJECT = 'tdw_controller-tests'
        self.EXPT_NAME = ''

        self.TASK_DB_PATHS = [ # sTODO : add these paths to release branch
            'task_db/resnet18_imagenet_12_seen_tasks.pt',
            'task_db/resnet18_imagenet_8_unseen_tasks.pt',
        ]
        self.REWARD_DB_PATH = 'controller_db/reward_db.json'

        self.SAVE_DIR = 'expts/tmp_last_controller_test'
        self.MODEL_LOAD_PATH = ''
        self.EVAL_ONLY = False

        self.TRAIN_TASKS = [
            'CropDisease',
            'EuroSAT',
            'SVHN',
            'ChestX',
            'Sketch',
            'DTD',
            'Flowers102',
            'DeepWeeds',
            'Resisc45',
            'Omniglot',
            'ISIC',
            'Kaokore'
        ]
        self.TEST_TASKS = [
            'CUB',
            'PacsC',
            'PacsS',
            'AID',
            'USPS',
            'FMD',
            'CactusAerial',
            'ChestXPneumonia'
        ]

        self.NUM_EPOCHS = 1000
        self.BATCH_SIZE = 4

        self.LR = 0.001
        self.LR_DECAY_GAMMA = 0.1
        self.LR_DECAY_EPOCH = 500
        self.WD = 1.e-5
        self.EXPL_NOISE = 0.4 # This adds noise to the controller distribution
        self.NOISE_RED_EPOCH = 5
        self.WARMUP_EPOCHS = 50

        self.NUM_REPLAY_STEPS = 5
        self.IMITATION_START_EPOCH = 50

        self.CONT_OUT_PARAMS = [
            'pose_rot',
            'pose_scale',
            'lighting_intensity',
            'lighting_color',
            'lighting_dir',
            'blur',
            'backgr',
            'materials'
        ]
        self.NUM_HIDDEN_NODES = 400
        self.CKPT_SAVE_INTERVAL = None

        # Override parameter values from YAML file first, then from override list
        if config_yaml:
            self.merge_from_file(config_yaml)
        self.merge_from_list(config_override)

        self.freeze()