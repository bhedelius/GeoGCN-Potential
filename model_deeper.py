import torch
import numpy as np

from external import *
from se3_transformer import *


class Model(nn.Module):
    def __init__(self, heads = 8, c_in=4, c_int=4, c_hid=4, c_out=1, depth=5):
        super(Model, self).__init__()
        layers = []
        
        layers.append(MultiHead(d_in=0, d_out=2, c_in=c_in, c_hid=c_hid, c_out = c_int, heads=heads))
        layers.append(GNonlinear(d_in=2, c_in=c_int, c_out=c_int))

        for _ in range(depth-2):
            layers.append(MultiHead(d_in=2, d_out=2, c_in=c_int, c_hid=c_hid, c_out=c_int, heads=heads))
            layers.append(GNonlinear(d_in=2, c_in=c_int, c_out=c_int))

        layers.append(MultiHead(d_in=2, d_out=1, c_in=c_int, c_hid=c_hid, c_out=c_out, heads=heads))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, pc, f):
        out = f
        for i, layer in enumerate(self.layers):
            if i%2==0:
                out = layer(pc, out)
            else:
                out = layer(out)
        return out
