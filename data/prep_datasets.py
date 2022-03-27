from classifier.config import ClassifierConfig
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import numpy as np
from torch.utils.data import Subset
from utils.data import ConcatDataset, SubsetCls, SubsetWIdx
from timm.data import create_transform
from data.transfer_broad_datasets import SimpleDataset, SimpleDatasetWIdx
from utils.ioutils import AttrDict
import yaml

def get_random_split(dataset, trn_percent: float, seed=0, ret_idx=False):
    """ split dataset into train/val index for the "Transfer Broad" datasets
        Args:
            trn_percent: percentage for training
    """
    rng = np.random.RandomState(seed)
    all_idx = list(range(len(dataset)))
    rng.shuffle(all_idx)

    trn_len = int(len(dataset) * trn_percent)

    trn_indices = all_idx[:trn_len]
    val_indices = all_idx[trn_len:]

    if ret_idx:
        return SubsetWIdx(dataset, trn_indices), SubsetWIdx(dataset, val_indices)
    else:
        return Subset(dataset, trn_indices), Subset(dataset, val_indices)

def get_dataset_from_name(data_name, data_class, split_frac,
                          train_transform, val_transform, opt):
    # Load yaml config
    with open('configs/datasets.yaml', 'r') as f:
        datasets = yaml.load(f, Loader=yaml.FullLoader)

    train_dataset = data_class(
        datasets[data_name]['data_path'], transform=train_transform,
        dataset_name='{}_train'.format(data_name), opt=opt, raise_error=True)
    if split_frac:
        train_dataset, val_dataset = get_random_split(
            train_dataset, split_frac, ret_idx=opt.get_subset_idx)
    else:
        val_dataset = None

    train_dataset.num_classes = datasets[data_name]['num_class']
    test_dataset = data_class(
        datasets[data_name]['data_path'], transform=val_transform,
        dataset_name='{}_test'.format(data_name), opt=opt, raise_error=True)
    test_dataset.num_classes = train_dataset.num_classes

    return train_dataset, val_dataset, test_dataset

def prep_datasets(C_ : ClassifierConfig, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if C_.DOWNSTREAM_EVAL == 'knn' or C_.DOWNSTREAM_EVAL == 'task2vec':
        train_transform = val_transform
    else:
        train_transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=C_.COLOR_JITTER,
            auto_augment=C_.AUTO_AUGMENT,
            interpolation=C_.INTERPOLATION
        )

    if C_.DATA_DIR:
        if C_.TRAIN_ONLY:
            train_dataset = ImageFolder(C_.DATA_DIR, transform=train_transform)
            train_dataset.num_classes = len(train_dataset.classes)
            val_dataset = None
            test_dataset = ImageFolder(C_.DATA_DIR, transform=val_transform)
            if C_.SUBSET_PATH:
                try:
                    idxs = np.load(C_.SUBSET_PATH)
                except Exception as e:
                    raise Exception(
                        'Exception while loading subset idxs file:{}'.format(e))
                train_dataset = SubsetCls(train_dataset, idxs)
                test_dataset = SubsetCls(test_dataset, idxs)
        else:
            all_train_data = ImageFolder(
                root=os.path.join(C_.DATA_DIR, 'train'), transform=train_transform)

            if C_.SUBSET_PATH:
                try:
                    idxs = np.load(C_.SUBSET_PATH)
                except Exception as e:
                    raise Exception(
                        'Exception while loading subset idxs file:{}'.format(e))
                all_train_data = SubsetCls(all_train_data, idxs)

            if C_.SPLIT_FRAC:
                # split into train and val
                rng = np.random.get_state()
                np.random.seed(44)  # so the split is the same each time
                train_idxs = []
                val_idxs = []
                train_targets = np.array(all_train_data.targets)
                all_idxs = np.arange(len(train_targets))
                for cl in range(len(all_train_data.classes)):
                    curr_idxs = np.random.permutation(all_idxs[train_targets == cl])
                    curr_train_idxs = curr_idxs[:int(C_.SPLIT_FRAC * len(curr_idxs))]
                    curr_val_idxs = curr_idxs[int(C_.SPLIT_FRAC * len(curr_idxs)):]
                    train_idxs.append(curr_train_idxs)
                    val_idxs.append(curr_val_idxs)

                train_idxs = np.concatenate(train_idxs)
                val_idxs = np.concatenate(val_idxs)
                np.random.set_state(rng)

                val_dataset = Subset(all_train_data, val_idxs)
                train_dataset = Subset(all_train_data, train_idxs)
                train_dataset.num_classes = len(train_dataset.dataset.classes)
            else:
                train_dataset = all_train_data
                train_dataset.num_classes = len(train_dataset.classes)
                val_dataset = None

            test_dataset = ImageFolder(
                root=os.path.join(C_.DATA_DIR, 'val'), transform=val_transform)
            if C_.TEST_SUBSET_PATH:
                try:
                    idxs = np.load(C_.TEST_SUBSET_PATH)
                except Exception as e:
                    raise Exception(
                        'Exception while loading subset idxs file:{}'.format(e))
                test_dataset = SubsetCls(test_dataset, idxs)
    else:
        if not C_.DATASET:
            raise Exception('DATASET needs to be set if DATA_DIR is None')

        # Additional options for SimpleDataset
        opt = AttrDict(
            seed=C_.SEED, split_fraction=0.7,
            train_n_way=None, get_subset_idx=False)
        if C_.DOWNSTREAM_EVAL == 'knn':
            if C_.SPLIT_FRAC is None:
                data_class = SimpleDatasetWIdx
            else:
                data_class = SimpleDataset
                opt.get_subset_idx = True
        else:
            data_class = SimpleDataset
        if '-' in C_.DATASET:
            # There are multiple datasets that need to be combined
            dataset_names = C_.DATASET.split('-')
            num_classes = []
            train_datasets = []
            val_datasets = []
            test_datasets = []
            for data_name in dataset_names:
                trd, vd, tsd = get_dataset_from_name(
                    data_name, data_class, C_.SPLIT_FRAC, train_transform, val_transform, opt)
                train_datasets.append(trd)
                val_datasets.append(vd)
                test_datasets.append(tsd)
                num_classes.append(trd.num_classes)

            train_dataset = ConcatDataset(train_datasets)
            if val_datasets[0] is not None:
                val_dataset = ConcatDataset(val_datasets)
            else:
                val_dataset = None
            test_dataset = ConcatDataset(test_datasets)
        else:
            train_dataset, val_dataset, test_dataset = get_dataset_from_name(
                C_.DATASET, data_class, C_.SPLIT_FRAC, train_transform, val_transform, opt)

    if C_.SPLIT_FRAC and C_.VAL_AS_TEST:
        test_dataset = val_dataset
        val_dataset = None

    if args.quick:
        # restrict to 32 batches
        if val_dataset:
            val_dataset.indices = val_dataset.indices[:32 * C_.BATCH_SIZE]
        train_dataset.indices = train_dataset.indices[:32 * C_.BATCH_SIZE]
        test_dataset.samples = test_dataset.samples[:32 * C_.BATCH_SIZE]

    return train_dataset, val_dataset, test_dataset