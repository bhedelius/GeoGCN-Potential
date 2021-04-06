import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

import dgl

from se3_transformer import PointCloud
from model import Model

from anidataset_ddp import *

import pickle as pkl
                 

try:
    print(type(dataset))
except:
    print('Loading dataset')
    dataset = AniDataset()


torch.cuda.empty_cache()


dev = 'cuda:1'
EPOCHS = 100
energy_weight = 1e-5

#setup the model
model = Model().to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
objective = nn.MSELoss()

def create_graph(seq_length_tensor):
    u = []
    v = []
    i = 0
    idx = 0

    while seq_length_tensor[i] != 0:
        l = int(seq_length_tensor[i].item())
        u_temp = []
        v_temp = []
        for j in range(l):
            for k in range(j):
                u_temp.append(j + idx)
                v_temp.append(k + idx)

        u += u_temp + v_temp
        v += v_temp + u_temp

        idx += l
        i += 1
   
    graph = dgl.graph((u,v))
    return graph



def train(position_tensor, species_tensor, forces_tensor, energy_tensor, seq_length_tensor):
    graph = create_graph(seq_length_tensor)
    num_nodes = graph.num_nodes()

    species_tensor = species_tensor[:num_nodes].to(dev)
    forces_tensor = forces_tensor[:num_nodes].to(dev)
    energy_tensor = energy_tensor[:num_nodes].to(dev)

    graph.ndata['pos'] = position_tensor[:num_nodes]
    pc = PointCloud(graph=graph).to(dev)
   
    f = {0: species_tensor.unsqueeze(-1)}
    optimizer.zero_grad()
    y_hat = model(pc, f)

    energy_loss = 0
    force_loss = 0
    idx = 0
    j = 0

    while seq_length_tensor[j] != 0:
        l = int(seq_length_tensor[j])

        energy_out = torch.sum(y_hat[0][idx:idx+l, 0, 0])
        energy_target = energy_tensor[j]
        energy_loss += objective(energy_out, energy_target)

        force_out = y_hat[1][idx:idx+l,0,:,]
        force_target = forces_tensor[idx:idx+l]
        force_loss += objective(force_out, force_target)

        idx += l
        i += 1

    energy_loss /= j
    force_loss  /= j
    loss = energy_weight * energy_loss + force_loss
    loss.backward()

    total_norm = torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    if total_norm == total_norm:
        optimizer.step()
    else:
        print('gradient update skipped bc of Nans')
    losses.append(loss.item())
    return energy_loss.item(), force_loss.item()

torch.autograd.set_detect_anomaly(True)
losses = []

for EPOCH in range(EPOCHS):
    for n in range(100000):
        position_tensor, species_tensor, forces_tensor, energy_tensor, seq_length_tensor = dataset.sample_batches()
        in_data = (position_tensor[0], species_tensor[0], forces_tensor[0], energy_tensor[0], seq_length_tensor[0])
        print(EPOCH, n, train(*in_data))
        if n % 1000 == 0:
            torch.save({'model': model,
                        'optimizer_state_dict': optimizer.state_dict()},
                       f'./models_ddp/{EPOCH}_{n}.pkl)')
            np.save(f'./models_ddp/{EPOCH}_{n}.npy', np.array(losses))
            losses = []


##### HOW TO LOAD A SAVED MODEL
'''
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
'''

#import plotext as plt
#plt.plot(range(len(losses)), losses)
#plt.show()




