import torch
from timm.models.vision_transformer import VisionTransformer
from timm.models.resnet import ResNet
import torch.nn.functional as F

def get_classifier_name(model):
    if isinstance(model, VisionTransformer):
        return 'head'
    elif isinstance(model, ResNet):
        return 'fc'
    else:
        raise Exception('Model type \'{}\' currently not supported'.format(type(model)))

def freeze_backbone(model : torch.nn.Module):
    classifier_name = get_classifier_name(model)
    for name, param in model.named_parameters():
        if classifier_name in name:
            pass
        else:
            param.requires_grad = False

    # so parameters like batchnorm and dropout don't change while linear eval
    for module in model.modules():
        module.eval()

def add_downstream_modules(C_, model):
    if C_.DOWNSTREAM_DROP > 0:
        model.add_module('downstream_drop', torch.nn.Dropout(C_.DOWNSTREAM_DROP))
    else:
        model.downstream_drop = None
    model.add_module('downstream_bn', torch.nn.BatchNorm1d(model.num_features, affine=False))

def set_head_training(model, train=True):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    if train:
        model.downstream_bn.train()
        if model.downstream_drop:
            model.downstream_drop.train()
    else:
        model.downstream_bn.eval()
        if model.downstream_drop:
            model.downstream_drop.eval()

def forward_downstream(self, x):
    x = self.forward_features(x)
    if isinstance(self, ResNet):
        x = self.global_pool(x)
    if self.downstream_drop:
        x = self.downstream_drop(x)
    x = self.downstream_bn(x)
    if isinstance(self, VisionTransformer):
        out = self.head(x)
    elif isinstance(self, ResNet):
        out = self.fc(x)
    return out


def get_features(self, x):
    if isinstance(self, VisionTransformer):
        return self.forward_features(x)
    elif isinstance(self, ResNet):
        x = self.forward_features(x)
        return self.global_pool(x)
