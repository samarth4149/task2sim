import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
import os
from scipy.io import loadmat
from torchvision.datasets.folder import default_loader
from itertools import chain
from data.transfer_broad_utils import prepare_data_indices, get_image_folder
from pathlib import Path
import h5py

identity = lambda x: x

# TODO : change this
TMP_PATH = '.tmp'

class TorchDataset(Dataset):
    """ parent class for torchvision datasets  """
    def __init__(self, ):
        super().__init__()

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, index: int):
        image, label = self.dset[index]
        image = image.convert('RGB')
        return image, label

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 transform,
                 target_transform=identity,
                 dataset_name=None,
                 consecutive_label=False,
                 opt=None,
                 raise_error=True):
        self.transform = transform
        self.target_transform = target_transform
        self.raise_error = raise_error
        self.rng = np.random.RandomState(seed=opt.seed)

        self.data, base_dataset_name, data_indices_suffix, mode = get_image_folder(
            dataset_name, data_path)

        self.cls_to_idx = None
        if 'partial' in data_indices_suffix or 'disjoint' in data_indices_suffix or 'overlap' in data_indices_suffix:
            tmpfile = os.path.join(
                TMP_PATH, base_dataset_name +
                f"_indices{data_indices_suffix}_{mode}_{opt.split_fraction}.npy"
            )
            if not os.path.exists(tmpfile):
                prepare_data_indices(dataset_name, data_path, opt=opt)

            class_indices = np.load(tmpfile, allow_pickle=True).item()
            self.list_classes = list(class_indices.keys())
            self.indices = list(chain.from_iterable(class_indices.values()))
            self.indices = np.array(self.indices)
            if consecutive_label:
                self.cls_to_idx = {
                    c: i
                    for i, c in enumerate(sorted(self.list_classes))
                }
            print(f"loading indices from {tmpfile}")
        else:
            self.indices = None

        if self.indices is not None:
            targets = np.array(self.data.targets)
            self.targets = targets[self.indices]
        else:
            self.targets = self.data.targets


    def __getitem__(self, i):
        idx = i
        if self.indices is not None:
            idx = self.indices[i]

        get_idx = idx
        while True:
            try:
                img, target = self.data[get_idx]
            except FileNotFoundError as fe:
                if self.raise_error:
                    raise FileNotFoundError(str(fe))
                if self.indices is not None:
                    get_idx = int(self.rng.choice(len(self.indices)))
                    get_idx = self.indices[get_idx]
                else:
                    get_idx = int(self.rng.choice(len(self.data)))
                img = None

            if img is not None:
                break

        if self.cls_to_idx is not None:
            target = self.cls_to_idx[target]

        img = self.transform(img)
        target = self.target_transform(target)
        return img, target

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self.data)

class SimpleDatasetWIdx(SimpleDataset):
    def __init__(self, *args, **kwargs):
        super(SimpleDatasetWIdx, self).__init__(*args, **kwargs)

    def __getitem__(self, i):
        idx = i
        if self.indices is not None:
            idx = self.indices[i]
        try:
            img, target = self.data[idx]
        except FileNotFoundError as fe:
            if self.raise_error:
                raise FileNotFoundError(str(fe))

            rand_idx = int(self.rng.choice(len(self.data)))
            img, target = self.data[rand_idx]

        if self.cls_to_idx is not None:
            target = self.cls_to_idx[target]

        img = self.transform(img)
        target = self.target_transform(target)
        return img, target, i, idx


class ConcatProportionDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, len_mode='max', return_data_idx=False):
        super().__init__()
        self.datasets = datasets
        self.len_mode = len_mode
        self.return_data_idx = return_data_idx
        self.ptr = 0

    def __len__(self):
        if self.len_mode == 'max':
            return max([len(dat) for dat in self.datasets])
        else:
            return min([len(dat) for dat in self.datasets])

    def __getitem__(self, idx):
        # data_idx = int(np.random.choice(len(self.datasets)))
        data_idx = self.ptr
        self.ptr = (self.ptr + 1) % len(self.datasets)

        dataset = self.datasets[data_idx]
        data, label = self.get_data(dataset, idx)
        if self.return_data_idx:
            label = data_idx
        return data, label

    def get_data(self, dataset, idx):
        if self.len_mode == 'max' and idx >= len(dataset):
            idx = np.random.choice(len(dataset))
        elif self.len_mode != 'max':
            idx = np.random.choice(len(dataset))
        data = dataset[idx]
        return data


class Omniglot(TorchDataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        self.dset = torchvision.datasets.Omniglot(data_root,
                                                  background=False,
                                                  download=True)
        self.targets = [d[1] for d in self.dset._flat_character_images]


class CIFAR100(TorchDataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        if mode is None or mode == '':
            mode = 'train'
        self.dset = torchvision.datasets.CIFAR100(data_root,
                                                  train=mode == 'train',
                                                  download=True)
        self.targets = self.dset.targets


class CIFAR10(TorchDataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        if mode is None or mode == '':
            mode = 'train'
        self.dset = torchvision.datasets.CIFAR10(data_root,
                                                 train=mode == 'train')
        self.targets = self.dset.targets


class Kaokore(Dataset):
    def __init__(self, data_root, mode='train') -> None:
        self.data_root = data_root
        self.mode = mode
        self.samples = self._make_dataset(mode)
        self.samples = self.samples.reset_index()
        self.targets = [4*s['gender'] + s['status'] for _, s in self.samples.iterrows()]

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index: int):
        sample = self.samples.iloc[index]
        gender, status = sample['gender'], sample['status']
        label = gender * 4 + status
        imfile = sample['image']
        imfile = os.path.join(self.data_root, 'images_256', imfile)
        if not os.path.exists(imfile):  # some images were not found
            raise FileNotFoundError('File \'{}\' not found'.format(imfile))
        image = default_loader(imfile)
        label = np.long(label)
        return image, label

    def _make_dataset(self, mode) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.data_root, 'labels.csv'))
        if mode == 'train' or mode == 'test':
            df = df[df['set'] == mode]
        return df


class DeepWeeds(Dataset):
    def __init__(self, data_root, mode='train') -> None:
        self.data_root = data_root
        self.samples = self._make_dataset()
        self.targets = [s['Label'] for _, s in self.samples.iterrows()]

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index: int):
        sample = self.samples.iloc[index]
        imfile, label = sample['Filename'], sample['Label']
        imfile = os.path.join(self.data_root, 'images', imfile)
        if not os.path.exists(imfile):  # some images were not found
            raise FileNotFoundError('File \'{}\' not found'.format(imfile))
        image = default_loader(imfile)
        label = np.long(label)
        return image, label

    def _make_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.data_root, 'labels.csv'))
        return df


class DeepFashion(Dataset):
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        if mode == 'train' or mode == 'test':
            self.samples = self._make_dataset(mode)
        else:
            self.samples = self._make_dataset('train')
            self.samples.extend(self._make_dataset('test'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        imfile, label = self.samples[index]
        image = default_loader(imfile)
        label = np.long(label - 1)
        return image, label

    def _make_dataset(self, mode):
        img_list = []
        with open(os.path.join(self.data_root, f'{mode}.txt'), 'r') as fp:
            for line in fp:
                img_list.append(os.path.join(self.data_root, line.strip()))
        labels = []
        with open(os.path.join(self.data_root, f'{mode}_cate.txt'), 'r') as fp:
            for line in fp:
                labels.append(int(line.strip()))
        instances = list(zip(img_list, labels))
        return instances


class SVHN(TorchDataset):
    def __init__(self, data_root, mode='test'):
        super().__init__()
        if mode is None or mode == '':
            mode = 'train'
        self.dset = torchvision.datasets.SVHN(data_root,
                                              split=mode,
                                              download=True)
        self.targets = self.dset.labels


class Flowers102(Dataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        self.data_root = data_root
        image_path = os.path.join(data_root, "jpg")
        self.file_images = sorted([
            os.path.join(image_path, imf) for imf in os.listdir(image_path)
            if imf.endswith('.jpg')
        ])
        self.info_label = loadmat(os.path.join(data_root,
                                               'imagelabels.mat'))['labels'][0]
        splits = loadmat(os.path.join(data_root, "splits.mat"))

        # NOTE: For CVPR: In the original dataset, train has only 1k images, anv test has 6k images
        # we are using the original test as train
        # For ICCV: we fixed it to the original split
        # train split now contains both train val
        if mode == 'train':
            # self.data_split = splits['tstid'][0]
            self.data_split = (splits['trnid'][0].tolist() +
                               splits['valid'][0].tolist())
        # elif mode == 'val':
        #     self.data_split = splits['valid'][0].tolist()
        elif mode == 'test':
            self.data_split = splits['tstid'][0].tolist()
        else:
            self.data_split = np.arange(1, len(self.file_images) + 1)

        self.targets = np.array(self.info_label, dtype=np.long)[np.array(self.data_split) - 1] - 1

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx: int):
        index = self.data_split[idx] - 1
        imfile = self.file_images[index]
        image = default_loader(imfile)
        label = np.long(self.info_label[index]) - 1
        return image, label


class Pets(Dataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        self.data_root = data_root
        image_path = os.path.join(data_root, "images")
        self.file_images = sorted(
            [imf for imf in os.listdir(image_path) if imf.endswith('jpg')])

        if mode == 'train':
            name = 'trainval.txt'
        elif mode == 'test':
            name = 'test.txt'
        else:
            name = 'list.txt'
        self._load_labels(os.path.join(data_root, 'annotations', name))

    def __len__(self):
        return len(self.name_to_label)

    def __getitem__(self, index: int):
        imfile = os.path.join(self.data_root, "images",
                              self.name_to_label[index][0] + ".jpg")
        image = default_loader(imfile)
        label = np.long(self.name_to_label[index][1])
        return image, label

    def _load_labels(self, filename):
        self.name_to_label = []
        with open(filename, 'r') as fp:
            for line in fp:
                if not line.startswith('#'):
                    name, label = line.split(' ')[:2]
                    self.name_to_label.append((name, int(label) - 1))


class SUN(Dataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        class_file = os.path.join(data_root, 'ClassName.txt')
        self.data_root = data_root
        self.classes = []
        with open(class_file, 'r') as fp:
            for line in fp:
                self.classes.append(line.strip())
        self.classes = sorted(self.classes)
        self.cls_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        imfile, label = self.samples[index]
        image = default_loader(imfile)
        label = np.long(label)
        return image, label

    def _make_dataset(self):
        instances = []
        for target_class in sorted(self.cls_to_idx.keys()):
            class_index = self.cls_to_idx[target_class]
            target_dir = self.data_root + target_class
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir,
                                                  followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if os.path.isfile(path):
                        item = path, class_index
                        instances.append(item)
        return instances


class miniImageNet(Dataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        if mode is None or mode == '':
            mode = 'train'
        data = np.load(os.path.join(data_root,
                                    f'mini-imagenet-cache-{mode}.pkl'),
                       allow_pickle=True)
        self.image_data = data['image_data']
        self.class_dict = data['class_dict']
        self.classes = sorted(list(self.class_dict.keys()))
        self.cls_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        imindex, label = self.samples[index]
        image = Image.fromarray(
            self.image_data[imindex].astype('uint8')).convert('RGB')
        label = np.long(label)
        return image, label

    def _make_dataset(self):
        instances = []
        for target_class in sorted(self.cls_to_idx.keys()):
            class_index = self.cls_to_idx[target_class]
            target_index = self.class_dict[target_class]
            instances.extend([(_ind, class_index) for _ind in target_index])
        return instances


class tieredImageNet(Dataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        partition = mode
        self.data_root = data_root
        self.partition = partition

        self.image_file_pattern = '%s_images.npz'
        self.label_file_pattern = '%s_labels.pkl'

        # modified code to load tieredImageNet
        image_file = os.path.join(self.data_root,
                                  self.image_file_pattern % partition)
        self.imgs = np.load(image_file)['images']
        label_file = os.path.join(self.data_root,
                                  self.label_file_pattern % partition)
        self.labels = self._load_labels(label_file)['labels']

    def __getitem__(self, item):
        img = Image.fromarray(np.asarray(self.imgs[item]).astype('uint8'))
        target = self.labels[item] - min(self.labels)
        return img, target

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _load_labels(file):
        try:
            with open(file, 'rb') as fo:
                data = pickle.load(fo)
            return data
        except:
            with open(file, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            return data



class ChestX(Dataset):
    def __init__(self, data_root, mode='train'):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = os.path.join(data_root, 'images_resized')
        self.csv_path = os.path.join(data_root, 'Data_Entry_2017.csv')
        self.used_labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax"
        ]

        self.labels_maps = {
            "Atelectasis": 0,
            "Cardiomegaly": 1,
            "Effusion": 2,
            "Infiltration": 3,
            "Mass": 4,
            "Nodule": 5,
            "Pneumothorax": 6
        }

        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name = []
        self.labels = []

        for name, label in zip(self.image_name_all, self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[
                    0] != "Pneumonia" and label[0] in self.used_labels:
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)

        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)
        self.targets = self.labels

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]

        # Open image
        # img_as_img = Image.open(self.img_path + single_image_name).resize(
        #     (256, 256)).convert('RGB')

        img_as_img = default_loader(os.path.join(self.img_path, single_image_name))

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]

        return (img_as_img, single_image_label)

    def __len__(self):
        return self.data_len


class ISIC(Dataset):
    def __init__(self,data_root, mode='train'):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = os.path.join(data_root, 'ISIC2018_Input_Resized')
        self.csv_path = os.path.join(data_root, 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')

        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name = np.asarray(self.data_info.iloc[:, 0])

        self.labels = np.asarray(self.data_info.iloc[:, 1:])
        self.labels = (self.labels != 0).argmax(axis=1)
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.targets = self.labels

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]
        img_as_img = default_loader(
            os.path.join(self.img_path, single_image_name + ".jpg"))

        single_image_label = self.labels[index]
        return (img_as_img, single_image_label)

    def __len__(self):
        return self.data_len


class ImageNet1K(datasets.ImageFolder):
    def __init__(self, data_root: str, mode='train'):
        if mode == 'test':
            mode = 'val'
        super().__init__(os.path.join(data_root, mode))


class SIN(datasets.ImageFolder):
    def __init__(self, data_root: str, mode='train'):
        if mode == 'test':
            mode = 'val'
        super().__init__(os.path.join(data_root, mode))


class SININ(datasets.ImageFolder):
    "SIN + ImageNet dataset"

    def __init__(self, data_root: str, mode='train'):
        if mode == 'test':
            mode = 'val'
        imnet_data = ImageNet1K(os.path.join(data_root, 'imagenet1k'), mode)
        sin_data = SIN(os.path.join(data_root, 'SIN'), mode)
        self.data = torch.utils.data.ConcatDataset((imnet_data, sin_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]


class ImageNet100(datasets.ImageFolder):
    def __init__(self, data_root: str, mode='train'):
        if mode == 'test':
            mode = 'val'
        self.class_list = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "imagenet-100_categories.txt")
        super().__init__(os.path.join(data_root, mode))

    def _find_classes(self, dir):
        with open(self.class_list, 'r') as fp:
            classes = fp.read().strip().split(" ")
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class ImageNet1K_subset(datasets.ImageFolder):
    def __init__(self, data_root: str, mode='train'):
        if mode == 'test':
            mode = 'val'
        self.class_list = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "imagenet-100_categories.txt")
        super().__init__(os.path.join(data_root, mode))

    def _find_classes(self, dir):
        with open(self.class_list, 'r') as fp:
            classes = fp.read().strip().split(" ")
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class CropDisease(datasets.ImageFolder):
    def __init__(self, data_root: str, mode='train'):
        if mode == 'train':
            path = os.path.join(data_root, 'train')
        elif mode == "test":
            path = os.path.join(data_root, 'test')
        else:
            path = os.path.join(data_root, 'all')
        super().__init__(path)
        # Already has targets from ImageFolder

class ExDark(Dataset):
    def __init__(self, data_root: str, mode='train'):
        super().__init__()
        self.samples = self._make_dataset(data_root, mode)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        imfile, label = self.samples[index]
        image = default_loader(imfile)
        label = np.long(label)
        return image, label

    def _make_dataset(self, data_root, mode):
        gt_path = os.path.join(data_root, 'labels.csv')
        img_root = os.path.join(data_root, 'images')
        self.classes = sorted(os.listdir(img_root))
        idx_to_cls = {i: c for i, c in enumerate(self.classes)}

        df_labels = pd.read_csv(
            gt_path,
            header=None,
            skiprows=1,
            sep=' ',
            names=['Name', 'Class', 'Light', 'In/Out', 'split'])
        if mode == 'train':
            df_labels = df_labels[(df_labels['split'] == 1)
                                  | (df_labels['split'] == 2)]
        elif mode == 'test':
            df_labels = df_labels[(df_labels['split'] == 3)]

        instances = []
        for _, row in df_labels.iterrows():
            cls_idx = row['Class'] - 1
            image = os.path.join(img_root, idx_to_cls[cls_idx], row['Name'])
            assert os.path.exists(image)
            instances.append((image, cls_idx))
        return instances


class CUB(Dataset):
    def __init__(self, data_root: str, mode='train'):
        super().__init__()
        self.samples = self._make_dataset(data_root, mode)
        self.targets = np.array([s[1] for s in self.samples], dtype=np.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        imfile, label = self.samples[index]
        image = default_loader(imfile)
        label = np.long(label)
        return image, label

    def _make_dataset(self, data_root, mode):
        img_root = os.path.join(data_root, 'images')

        def fn_read(path):
            _list = []
            with open(path, 'r') as fp:
                for line in fp:
                    line = line.strip()
                    if len(line) > 0:
                        imname, lab = line.split(' ')
                        lab = int(lab)
                        imname = os.path.join(img_root, imname)
                        assert os.path.exists(imname)
                        _list.append((imname, lab))
            return _list

        instances = []
        if mode == 'train':
            instances = fn_read(os.path.join(data_root, 'train_filelist.txt'))
        elif mode == 'test':
            instances = fn_read(os.path.join(data_root, 'test_filelist.txt'))
        else:
            instances = fn_read(os.path.join(data_root, 'train_filelist.txt'))
            instances.extend(fn_read(os.path.join(data_root, 'test_filelist.txt')))
        return instances


class DTD(Dataset):
    def __init__(self, data_root: str, mode='train'):
        super().__init__()
        self.samples = self._make_dataset(data_root, mode)
        self.targets = [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        imfile, label = self.samples[index]
        image = default_loader(imfile)
        label = np.long(label)
        return image, label

    def _make_dataset(self, data_root, mode):
        img_root = os.path.join(data_root, 'images')
        classes = sorted(os.listdir(img_root))
        cls_to_idx = {c: i for i, c in enumerate(classes)}

        def fn_read(path):
            _list = []
            with open(path, 'r') as fp:
                for line in fp:
                    line = line.strip()
                    if len(line) > 0:
                        lab = cls_to_idx[line.split('/')[0]]
                        imname = os.path.join(img_root, line)
                        assert os.path.exists(imname)
                        _list.append((imname, lab))
            return _list

        instances = []
        if mode == 'train':
            instances = fn_read(os.path.join(data_root, 'labels',
                                             'train1.txt'))
            instances.extend(
                fn_read(os.path.join(data_root, 'labels', 'val1.txt')))
        elif mode == 'test':
            instances = fn_read(os.path.join(data_root, 'labels', 'test1.txt'))
        else:
            instances = fn_read(os.path.join(data_root, 'labels',
                                             'train1.txt'))
            instances.extend(
                fn_read(os.path.join(data_root, 'labels', 'val1.txt')))
            instances.extend(
                fn_read(os.path.join(data_root, 'labels', 'test1.txt')))

        return instances


class PACSBase(Dataset):
    def __init__(self, data_root, domain, mode):
        super(PACSBase, self).__init__()
        if mode == 'test':
            mode = 'val'
        with h5py.File(Path(data_root) / f'{domain}_{mode}.hdf5', 'r') as f:
            temp_imgs = np.array(f['images'])
            temp_labels = np.array(f['labels'], dtype=np.long)
            temp_imgs = temp_imgs[:, :, :, ::-1]
            temp_labels = temp_labels - 1
            self.images = temp_imgs
            self.targets = temp_labels
            self.n_classes = 7

    def __getitem__(self, idx):
        img = Image.fromarray(np.uint8(self.images[idx]))
        tgt = self.targets[idx]
        return img, tgt

    def __len__(self):
        return len(self.images)

class PacsC(PACSBase):
    def __init__(self, data_root, mode='train'):
        super(PacsC, self).__init__(data_root, 'cartoon', mode)

class PacsS(PACSBase):
    def __init__(self, data_root, mode='train'):
        super(PacsS, self).__init__(data_root, 'sketch', mode)

class USPS(TorchDataset):
    def __init__(self, data_root, mode='test'):
        super().__init__()
        if mode is None or mode == '':
            mode = 'train'
        if mode == 'train':
            train = True
        else:
            train = False
        self.dset = torchvision.datasets.USPS(data_root,
                                              train=train,
                                              download=False)
        self.targets = self.dset.targets

class ChestXPneumonia(datasets.ImageFolder):
    def __init__(self, data_root, mode='train'):
        super(ChestXPneumonia, self).__init__(str(Path(data_root) / mode))

class CactusAerial(datasets.ImageFolder):
    def __init__(self, data_root, mode='train'):
        if mode == 'train':
            data_path = Path(data_root) / 'training_set' / 'training_set'
        else:
            data_path = Path(data_root) / 'validation_set' / 'validation_set'
        super(CactusAerial, self).__init__(data_path)