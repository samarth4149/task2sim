# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import itertools
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging
from t2v import variational
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.optimizer import Optimizer
from t2v.utils import AverageMeter, get_error, get_device


class Embedding:
    def __init__(self, hessian, scale, meta=None):
        self.hessian = np.array(hessian)
        self.scale = np.array(scale)
        self.meta = meta

    def get_hessian(self, normalized=True):
        hess = np.array(self.hessian)
        if normalized:
            scale = np.array(self.scale)
            hess = hess / scale
        return hess


class ProbeNetwork(ABC, nn.Module):
    """Abstract class that all probe networks should inherit from.

    This is a standard torch.nn.Module but needs to expose a classifier property that returns the final classicifation
    module (e.g., the last fully connected layer).
    """

    @property
    @abstractmethod
    def classifier(self):
        raise NotImplementedError("Override the classifier property to return the submodules of the network that"
                                  " should be interpreted as the classifier")

    @classifier.setter
    @abstractmethod
    def classifier(self, val):
        raise NotImplementedError("Override the classifier setter to set the submodules of the network that"
                                  " should be interpreted as the classifier")


class Task2Vec:

    def __init__(self, model: ProbeNetwork, skip_layers=0, max_samples=None, classifier_opts=None,
                 method='montecarlo', method_opts=None, loader_opts=None, bernoulli=False):
        if classifier_opts is None:
            classifier_opts = {}
        if method_opts is None:
            method_opts = {}
        if loader_opts is None:
            loader_opts = {}
        assert method in ('variational', 'montecarlo')
        assert skip_layers >= 0

        self.model = model
        # Fix batch norm running statistics (i.e., put batch_norm layers in eval mode)
        self.model.train()
        self.device = get_device(self.model)
        self.skip_layers = skip_layers
        self.max_samples = max_samples
        self.classifier_opts = classifier_opts
        self.method = method
        self.method_opts = method_opts
        self.loader_opts = loader_opts
        self.bernoulli = bernoulli
        self.loss_fn = nn.CrossEntropyLoss() if not self.bernoulli else nn.BCEWithLogitsLoss()
        self.loss_fn = self.loss_fn.to(self.device)

    def embed(self, dataset: Dataset):
        # Cache the last layer features (needed to train the classifier) and (if needed) the intermediate layer features
        # so that we can skip the initial layers when computing the embedding
        if self.skip_layers > 0:
            self._cache_features(dataset, indexes=(self.skip_layers, -1), loader_opts=self.loader_opts,
                                 max_samples=self.max_samples)
        else:
            self._cache_features(dataset, max_samples=self.max_samples)
        # Fits the last layer classifier using cached features
        self._fit_classifier(**self.classifier_opts)

        if self.skip_layers > 0:
            dataset = torch.utils.data.TensorDataset(self.model.layers[self.skip_layers].input_features,
                                                     self.model.layers[-1].targets)
        self.compute_fisher(dataset)
        embedding = self.extract_embedding(self.model)
        return embedding

    def montecarlo_fisher(self, dataset: Dataset, epochs: int = 1):
        logging.info("Using montecarlo Fisher")
        if self.skip_layers > 0:
            dataset = torch.utils.data.TensorDataset(self.model.layers[self.skip_layers].input_features,
                                                     self.model.layers[-1].targets)
        data_loader = _get_loader(dataset, **self.loader_opts)
        device = get_device(self.model)
        logging.info("Computing Fisher...")

        for p in self.model.parameters():
            p.grad2_acc = torch.zeros_like(p.data)
            p.grad_counter = 0
        for k in range(epochs):
            logging.info(f"\tepoch {k + 1}/{epochs}")
            for i, (data, target) in enumerate(tqdm(data_loader, leave=False, desc="Computing Fisher")):
                data = data.to(device)
                output = self.model(data, start_from=self.skip_layers)
                # The gradients used to compute the FIM needs to be for y sampled from
                # the model distribution y ~ p_w(y|x), not for y from the dataset
                if self.bernoulli:
                    target = torch.bernoulli(F.sigmoid(output)).detach()
                else:
                    target = torch.multinomial(F.softmax(output, dim=-1), 1).detach().view(-1)
                loss = self.loss_fn(output, target)
                self.model.zero_grad()
                loss.backward()
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad2_acc += p.grad.data ** 2
                        p.grad_counter += 1
        for p in self.model.parameters():
            if p.grad_counter == 0:
                del p.grad2_acc
            else:
                p.grad2_acc /= p.grad_counter
        logging.info("done")

    def _run_epoch(self, data_loader: DataLoader, model: ProbeNetwork, loss_fn,
                   optimizer: Optimizer, epoch: int, train: bool = True,
                   add_compression_loss: bool = False, skip_layers=0, beta=1.0e-7):
        metrics = AverageMeter()
        device = get_device(model)

        for i, (input, target) in enumerate(tqdm(data_loader, leave=False, desc="Computing Fisher")):
            input = input.to(device)
            target = target.to(device)
            output = model(input, start_from=skip_layers)

            loss = loss_fn(output, target)
            lz = beta * variational.get_compression_loss(model) if add_compression_loss else torch.zeros_like(loss)
            loss += lz

            error = get_error(output, target)

            metrics.update(n=input.size(0), loss=loss.item(), lz=lz.item(), error=error)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # logging.info(
        print(
            "{}: [{epoch}] ".format('Epoch' if train else '', epoch=epoch) +
            "Data/Batch: {:.3f}/{:.3f} ".format(metrics.avg["data_time"], metrics.avg["batch_time"]) +
            "Loss {:.3f} Lz: {:.3f} ".format(metrics.avg["loss"], metrics.avg["lz"]) +
            "Error: {:.2f}".format(metrics.avg["error"])
        )
        return metrics.avg

    def variational_fisher(self, dataset: Dataset, epochs=1, beta=1e-7):
        logging.info("Training variational fisher...")
        parameters = []
        for layer in self.model.layers[self.skip_layers:-1]:
            if isinstance(layer, nn.Module):  # Skip lambda functions
                variational.make_variational(layer)
                parameters += variational.get_variational_vars(layer)
        bn_params = []
        # Allows batchnorm parameters to change
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_params += list(m.parameters())
        # Avoids computing the gradients wrt to the weights to save time and memory
        for p in self.model.parameters():
            if p not in set(parameters) and p not in set(self.model.classifier.parameters()):
                p.old_requires_grad = p.requires_grad
                p.requires_grad = False

        optimizer = torch.optim.Adam([
            {'params': parameters},
            {'params': bn_params, 'lr': 5e-4},
            {'params': self.model.classifier.parameters(), 'lr': 5e-4}],
            lr=1e-2, betas=(.9, 0.999))
        if self.skip_layers > 0:
            dataset = torch.utils.data.TensorDataset(self.model.layers[self.skip_layers].input_features,
                                                     self.model.layers[-1].targets)
        train_loader = _get_loader(dataset, **self.loader_opts)

        for epoch in range(epochs):
            self._run_epoch(train_loader, self.model, self.loss_fn, optimizer, epoch, beta=beta,
                            add_compression_loss=True, train=True)

        # Resets original value of requires_grad
        for p in self.model.parameters():
            if hasattr(p, 'old_requires_grad'):
                p.requires_grad = p.old_requires_grad
                del p.old_requires_grad

    def compute_fisher(self, dataset: Dataset):
        """
        Computes the Fisher Information of the weights of the model wrt the model output on the dataset and stores it.

        The Fisher Information Matrix is defined as:
            F = E_{x ~ dataset} E_{y ~ p_w(y|x)} [\nabla_w log p_w(y|x) \nabla_w log p_w(y|x)^t]
        where p_w(y|x) is the output probability vector of the network and w are the weights of the network.
        Notice that the label y is sampled from the model output distribution and not from the dataset.

        This code only approximate the diagonal of F. The result is stored in the model layers and can be extracted
        using the `get_fisher` method. Different approximation methods of the Fisher information matrix are available,
        and can be selected in the __init__.

        :param dataset: dataset with the task to compute the Fisher on
        """
        if self.method == 'variational':
            fisher_fn = self.variational_fisher
        elif self.method == 'montecarlo':
            fisher_fn = self.montecarlo_fisher
        else:
            raise ValueError(f"Invalid Fisher method {self.method}")
        fisher_fn(dataset, **self.method_opts)

    def _cache_features(self, dataset: Dataset, indexes=(-1,), max_samples=None, loader_opts: dict = None):
        logging.info("Caching features...")
        if loader_opts is None:
            loader_opts = {}
        data_loader = DataLoader(dataset, shuffle=False, batch_size=loader_opts.get('batch_size', 64),
                                 num_workers=loader_opts.get('num_workers', 6), drop_last=False)

        device = next(self.model.parameters()).device

        def _hook(layer, inputs):
            if not hasattr(layer, 'input_features'):
                layer.input_features = []
            layer.input_features.append(inputs[0].data.cpu().clone())

        hooks = [self.model.layers[index].register_forward_pre_hook(_hook)
                 for index in indexes]
        if max_samples is not None:
            n_batches = min(
                math.floor(max_samples / data_loader.batch_size) - 1, len(data_loader))
        else:
            n_batches = len(data_loader)
        targets = []

        for i, (input, target) in tqdm(enumerate(itertools.islice(data_loader, 0, n_batches)), total=n_batches,
                                       leave=False,
                                       desc="Caching features"):
            targets.append(target.clone())
            self.model(input.to(device))
        for hook in hooks:
            hook.remove()
        for index in indexes:
            self.model.layers[index].input_features = torch.cat(self.model.layers[index].input_features)
        self.model.layers[-1].targets = torch.cat(targets)

    def _fit_classifier(self, optimizer='adam', learning_rate=0.0004, weight_decay=0.0001,
                        epochs=10):
        """Fits the last layer of the network using the cached features."""
        logging.info("Fitting final classifier...")
        if not hasattr(self.model.classifier, 'input_features'):
            raise ValueError("You need to run `cache_features` on model before running `fit_classifier`")
        targets = self.model.classifier.targets.to(self.device)
        features = self.model.classifier.input_features.to(self.device)

        dataset = torch.utils.data.TensorDataset(features, targets)
        data_loader = _get_loader(dataset, **self.loader_opts)

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer {optimizer}')

        loss_fn = nn.CrossEntropyLoss()
        for epoch in tqdm(range(epochs), desc="Fitting classifier", leave=False):
            metrics = AverageMeter()
            for data, target in data_loader:
                optimizer.zero_grad()
                output = self.model.classifier(data)
                loss = loss_fn(self.model.classifier(data), target)
                error = get_error(output, target)
                loss.backward()
                optimizer.step()
                metrics.update(n=data.size(0), loss=loss.item(), error=error)
            logging.info(f"[epoch {epoch}]: " + "\t".join(f"{k}: {v}" for k, v in metrics.avg.items()))

    def extract_embedding(self, model: ProbeNetwork):
        """
        Reads the values stored by `compute_fisher` and returns them in a common format that describes the diagonal of the
        Fisher Information Matrix for each layer.

        :param model:
        :return:
        """
        hess, scale = [], []
        for name, module in model.named_modules():
            if module is model.classifier:
                continue
            # The variational Fisher approximation estimates the variance of noise that can be added to the weights
            # without increasing the error more than a threshold. The inverse of this is proportional to an
            # approximation of the hessian in the local minimum.
            if hasattr(module, 'logvar0') and hasattr(module, 'loglambda2'):
                logvar = module.logvar0.view(-1).detach().cpu().numpy()
                hess.append(np.exp(-logvar))
                loglambda2 = module.loglambda2.detach().cpu().numpy()
                scale.append(np.exp(-loglambda2).repeat(logvar.size))
            # The other Fisher approximation methods directly approximate the hessian at the minimum
            elif hasattr(module, 'weight') and hasattr(module.weight, 'grad2_acc'):
                grad2 = module.weight.grad2_acc.cpu().detach().numpy()
                filterwise_hess = grad2.reshape(grad2.shape[0], -1).mean(axis=1)
                hess.append(filterwise_hess)
                scale.append(np.ones_like(filterwise_hess))
        return Embedding(hessian=np.concatenate(hess), scale=np.concatenate(scale), meta=None)


def _get_loader(trainset, testset=None, batch_size=64, num_workers=6, num_samples=10000, drop_last=True):
    if getattr(trainset, 'is_multi_label', False):
        raise ValueError("Multi-label datasets not supported")
    # TODO: Find a way to standardize this
    if hasattr(trainset, 'labels'):
        labels = trainset.labels
    elif hasattr(trainset, 'targets'):
        labels = trainset.targets
    elif isinstance(trainset, Subset):
        # For subset, first need to use indices from subset and then check if indices exists in original data
        target_array = np.array(trainset.dataset.targets)
        labels = target_array[np.array(trainset.indices)]
    elif hasattr(trainset, 'data'): # For the transferbroad datasets
        # NOTE: this would work only when there isn't a train/val split
        if trainset.indices:
            labels = list(trainset.data[idx][1] for idx in trainset.indices)
        else:
            labels = list(datum[1] for datum in trainset.data)
    else:
        labels = list(trainset.tensors[1].cpu().numpy())
    num_classes = int(getattr(trainset, 'num_classes', max(labels) + 1))
    class_count = np.eye(num_classes)[labels].sum(axis=0)
    weights = 1. / class_count[labels] / num_classes
    weights /= weights.sum()

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=num_samples)
    # No need for mutli-threaded loading if everything is already in memory,
    # and would raise an error if TensorDataset is on CUDA
    num_workers = num_workers if not isinstance(trainset, torch.utils.data.TensorDataset) else 0
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=batch_size,
                                              num_workers=num_workers, drop_last=drop_last)

    if testset is None:
        return trainloader
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                                 num_workers=num_workers)
        return trainloader, testloader
