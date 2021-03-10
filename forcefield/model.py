import torch
import numpy as np

from external import *
from se3_transformer import *


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        layers = []
        heads = 8
        c = 3
        c_hid = 3
        depth = 3
        
        layers.append(MultiHead(d_in=0, d_out=2, c_in=4, c_hid=c_hid, c_out = c, heads=heads))
        layers.append(GNonlinear(d_in=2, c_in=c, c_out=c))

        #for _ in range(depth-2):
        layers.append(MultiHead(d_in=2, d_out=2, c_in=c, c_hid=c_hid, c_out=c, heads=heads))
        layers.append(GNonlinear(d_in=2, c_in=c, c_out=c))

        layers.append(MultiHead(d_in=2, d_out=1, c_in=c, c_hid=c_hid, c_out=1, heads=heads))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, pc, f):
        out = f
        for i, layer in enumerate(self.layers):
            if i%2==0:
                out = layer(pc, out)
            else:
                out = layer(out)
        return out
