import torch
import torchvision.transforms as transforms
from abc import abstractmethod
from torchvision.datasets import ImageFolder
from PIL import ImageFile
from tqdm import tqdm
from data import transfer_broad_datasets as tbd
import os
import numpy as np
from numpy.random import RandomState

ImageFile.LOAD_TRUNCATED_IMAGES = True

identity = lambda x: x

# NOTE: temporary store cache in TMP_PATH
# TODO : Set up proper TMP_PATH
TMP_PATH = '.tmp'
# os.makedirs(TMP_PATH, exist_ok = True)

def get_split(dname):
    splits = dname.split("_")
    if len(splits) > 1:
        base, mode = splits[0], splits[-1]
    else:
        base, mode = splits[0], None
    # These datasets have no train/test split, manually create them
    # SUN397, ISIC, ChestX, EuroSAT, Omniglot, sketch, DeepWeeds, Resisc45
    dataset_no_split = [
        "SUN397", "ISIC", "ChestX", "EuroSAT", "Omniglot", "Sketch",
        "DeepWeeds", "Resisc45", "AID", "FMD"
    ]
    data_indices_suffix = ""
    if base in dataset_no_split:
        if mode is not None:
            data_indices_suffix = "_partial"
    return base, data_indices_suffix, mode

def get_image_folder(dataset_name, data_path=None):

    base_dataset_name, data_indices_suffix, mode = get_split(dataset_name)

    if base_dataset_name in tbd.__dict__.keys():
        dset = tbd.__dict__[base_dataset_name](data_path,
                                               mode=mode)
    else:
        dset = ImageFolder(data_path)

    return dset, base_dataset_name, data_indices_suffix, mode


def map_ind_to_label(dataset_name, data):
    tmpfile = os.path.join(TMP_PATH, dataset_name + f"_indices.npy")
    if not os.path.exists(tmpfile):
        sub_meta_indices = _get_ind_to_label(data, dataset_name)
        if not os.path.exists(TMP_PATH):
            os.makedirs(TMP_PATH, exist_ok=True)

        np.save(os.path.join(TMP_PATH, dataset_name + f"_indices.npy"),
                sub_meta_indices)

def prepare_data_indices(dataset_name, data_path, opt=None):
    base_dataset_name, data_indices_suffix, mode = get_split(dataset_name)
    indfile = os.path.join(TMP_PATH, base_dataset_name + f"_indices.npy")

    if not os.path.exists(indfile):
        data, *_ = get_image_folder(dataset_name, data_path)
        map_ind_to_label(base_dataset_name, data)
    if data_indices_suffix:
        tmpfile = os.path.join(
            TMP_PATH, base_dataset_name +
            f"_indices{data_indices_suffix}_{mode}_{opt.split_fraction}.npy")
        if not os.path.exists(tmpfile):
            data_dict = np.load(indfile, allow_pickle=True).item()
            if "disjoint" in data_indices_suffix:
                create_disjoint_indices(data_dict,
                                        base_dataset_name,
                                        num_split=4,
                                        min_way=opt.train_n_way,
                                        fraction=opt.split_fraction)

            elif "_sup" in data_indices_suffix or "_unsup" in data_indices_suffix or "partial" in data_indices_suffix:
                create_partial_data(data_dict,
                                    base_dataset_name,
                                    fraction=opt.split_fraction)

def _get_ind_to_label(data, dataset_name=None):
    sub_meta_indices = {}

    # Dummy dataset to be passed to DataLoader
    class LoaderInd:
        def __init__(self, data) -> None:
            self.data = data

        def __len__(self):
            return len(data)

        def __getitem__(self, index):
            try:
                _, label = self.data[index]
            except FileNotFoundError:
                return None, None
            return label, index

    # NOTE : Not my code.
    # This has been done instead of a simple loop, not really bothering to change it
    _loader = torch.utils.data.DataLoader(LoaderInd(data),
                                          batch_size=None,
                                          batch_sampler=None,
                                          collate_fn=identity,
                                          num_workers=3,
                                          shuffle=False)
    for label, i in tqdm(_loader,
                         total=len(data),
                         desc=f"storing indices {dataset_name}: "):
        if label is None:
            continue
        if label not in sub_meta_indices:
            sub_meta_indices[label] = []
        sub_meta_indices[label].append(i)

    return sub_meta_indices


def create_partial_data(data_dict, dataset, fraction=0.5, seed=0):
    #NOTE: fraction is train split
    random_state = RandomState(seed)
    dict_train = {}
    dict_test = {}
    for k in data_dict:
        indices = data_dict[k]
        unsup_length = int(fraction * len(indices))

        random_state.shuffle(indices)
        dict_train[k] = indices[:unsup_length]
        dict_test[k] = indices[unsup_length:]

    np.save(
        os.path.join(TMP_PATH,
                     dataset + f"_indices_partial_train_{fraction}.npy"),
        dict_train)
    np.save(
        os.path.join(TMP_PATH,
                     dataset + f"_indices_partial_test_{fraction}.npy"),
        dict_test)


def create_disjoint_indices(data_dict,
                            dataset,
                            num_split=4,
                            min_way=5,
                            fraction=0.5,
                            seed=0):
    #NOTE: fraction is train split
    num_classes = len(data_dict)
    for i_split in range(num_split):
        random_state = RandomState(seed + i_split)
        dict_unsupervised = {}
        dict_supervised = {}

        if num_classes >= 2 * min_way:
            unsupervised_classes = random_state.choice(num_classes,
                                                       int(num_classes *
                                                           fraction),
                                                       replace=False)
            supervised_classes = [
                c for c in range(num_classes) if c not in unsupervised_classes
            ]

        else:
            cls_list = np.arange(num_classes)
            random_state.shuffle(cls_list)
            num_unsup = max(int(num_classes * fraction), min_way)
            unsupervised_classes = cls_list[:num_unsup]
            supervised_classes = cls_list[-num_unsup:]

        for k in data_dict:
            if k in unsupervised_classes:
                dict_unsupervised[k] = data_dict[k]
            if k in supervised_classes:
                dict_supervised[k] = data_dict[k]

            np.save(
                os.path.join(
                    TMP_PATH, dataset +
                    f"_indices_unsup_disjoint_{i_split}_{fraction}.npy"),
                dict_unsupervised)
            np.save(
                os.path.join(
                    TMP_PATH, dataset +
                    f"_indices_sup_disjoint_{i_split}_{fraction}.npy"),
                dict_supervised)

        if i_split == 0:
            np.save(
                os.path.join(
                    TMP_PATH,
                    dataset + f"_indices_unsup_disjoint_{fraction}.npy"),
                dict_unsupervised)
            np.save(
                os.path.join(TMP_PATH, dataset +
                             f"_indices_sup_disjoint_{fraction}.npy"),
                dict_supervised)


def create_overlap_data(data_dict, dataset, fraction=0.7, seed=0):
    num_classes = len(data_dict)
    random_state = RandomState(seed)

    dict_unsupervised = {}
    dict_supervised = {}

    min_way = int(num_classes * fraction)

    cls_list = np.arange(num_classes)
    random_state.shuffle(cls_list)

    num_unsup = min_way
    unsupervised_classes = cls_list[:num_unsup]
    supervised_classes = cls_list[-num_unsup:]

    for k in data_dict:
        if k in unsupervised_classes:
            dict_unsupervised[k] = data_dict[k]
        if k in supervised_classes:
            dict_supervised[k] = data_dict[k]

    print("Overlapped indices ",
          set(unsupervised_classes).intersection(set(supervised_classes)))

    np.save(
        os.path.join(TMP_PATH,
                     dataset + f"_indices_unsup_overlap_{fraction}.npy"),
        dict_unsupervised)
    np.save(
        os.path.join(TMP_PATH,
                     dataset + f"_indices_sup_overlap_{fraction}.npy"),
        dict_supervised)
