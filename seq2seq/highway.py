# https://arxiv.org/pdf/1505.00387.pdf
# https://github.com/kefirski/pytorch_Highway
# https://github.com/bamtercelboo/pytorch_Highway_Networks/blob/master/models/model_HighWay.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, dim, layer_num):
        super(Highway, self).__init__()
        self.num_layers = layer_num
        self.gate = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_num)])

    def forward(self, x, non_x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            allow_transformation = torch.mul(non_x, gate)
            allow_carry = torch.mul(x, (1-gate))
            x = torch.add(allow_transformation, allow_carry)
        return x
