import torch.nn as nn
from collections import OrderedDict

class ControllerModel(nn.Module):
    def __init__(self, in_dim, out_dim, head_out_dims:OrderedDict):
        super(ControllerModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict(
            {key:nn.Sequential(nn.Linear(out_dim, head_out), nn.Softmax(dim=1))
             for key, head_out in head_out_dims.items()})

    def forward(self, x):
        return OrderedDict({key:head(self.backbone(x)) for key, head in self.heads.items()})