import math
from typing import TypeVar, Iterable, List, Sequence

import torch.utils
from torch import distributed as dist
from torch.utils.data import Dataset, IterableDataset, Subset, Sampler
import bisect

from classifier.config import ClassifierConfig

T_co = TypeVar('T_co', covariant=True)
import numpy as np


# TODO : add custom subset datasets that can use classes or idxs

class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets. Changes the class label of

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.all_num_classes = [d.num_classes for d in self.datasets]
        self.num_classes = sum(self.all_num_classes)
        self.targets = np.concatenate(
            [sum(self.all_num_classes[:i])+np.array(d.targets) for i,d in enumerate(datasets)])

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        sample = list(self.datasets[dataset_idx][sample_idx])

        # change target label according to dataset
        sample[1] += sum(self.all_num_classes[:dataset_idx])
        return sample

class SubsetCls(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        if not hasattr(self.dataset, 'targets'):
            raise Exception('dataset needs to have attribute \'targets\'')
        self.targets = np.array(self.dataset.targets)
        self.classes, self.targets = np.unique(self.targets, return_inverse=True)
        self.num_classes = len(self.classes)

    def __getitem__(self, idx):
        sample, tgt = self.dataset[self.indices[idx]]
        return sample, self.targets[idx]


    def __len__(self):
        return len(self.indices)

class SubsetWIdx(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]] + (idx,)

    def __len__(self):
        return len(self.indices)

def get_uniform_subset(dataset: Dataset[T_co], num=5, numtype='shot'):
    assert numtype in ['shot', 'pct']
    assert hasattr(dataset, 'targets')

    all_idxs = np.arange(len(dataset.targets))
    clss_idxs = {k: all_idxs[dataset.targets == k] for k in np.unique(dataset.targets)}

    all_final_idxs = []
    for cl, idxs in clss_idxs.items():
        if numtype == 'shot':
            all_final_idxs.append(idxs[:num])
        else:
            if int(num * len(idxs)) == 0:
                raise Exception('Too few images. Only {} imgs in class'.format(len(idxs)))
            all_final_idxs.append(idxs[:int(num * len(idxs))])

    all_final_idxs = np.concatenate(all_final_idxs)
    return Subset(dataset, all_final_idxs)


def create_loader(C_ : ClassifierConfig, args, dataset, is_training, pin_memory):
    if args.distributed:
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = OrderedDistributedSampler(dataset)
    else:
        sampler = None

    return torch.utils.data.DataLoader(
        dataset, batch_size=C_.BATCH_SIZE, num_workers=C_.NUM_WORKERS,
        pin_memory=pin_memory, shuffle=((sampler is None) and is_training),
        sampler=sampler, drop_last=is_training)


class OrderedDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples