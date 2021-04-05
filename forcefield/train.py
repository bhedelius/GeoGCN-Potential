import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

import dgl

from se3_transformer import PointCloud
from model import Model

from anidataset import *

import pickle as pkl
                 

try:
    print(type(dataset))
except:
    print('Loading dataset')
    dataset = AniDataset() 


torch.cuda.empty_cache()


dev = 'cuda:0'
BATCH_SIZE = 50
EPOCHS = 100
energy_weight = 1e-5
#setup the model
model = Model().to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
objective = nn.MSELoss()

def train(i):
    graphs = []
    for idx in order[i:min(i+BATCH_SIZE,len(dataset))]:
        pos, species, forces, energy = dataset[idx] 
        pc = PointCloud(pos=pos, cutoff = 1000)
        graph = pc.graph
        graph.ndata[0] = species.unsqueeze(-1).float()
        graph.ndata['forces'] = forces
        graph.ndata['energy'] = torch.ones(graph.num_nodes())*energy
        graphs.append(graph)
    batch = dgl.batch(graphs)
    pc = PointCloud(graph=batch).to(dev)
    f = {0: pc.graph.ndata[0]}

    optimizer.zero_grad()
    y_hat = model(pc, f)

    energy_loss = 0
    force_loss = 0
    begin_node = 0
    end_node = 0
    for graph in graphs:
        end_node += graph.num_nodes()
        energy_out = torch.sum(y_hat[0][begin_node:end_node,0,0])
        force_out = y_hat[1][begin_node:end_node,0,:]
        energy_loss += objective(energy_out, pc.graph.ndata['energy'][begin_node])
        force_loss  += objective(force_out,  pc.graph.ndata['forces'][begin_node:end_node])
        begin_node = end_node
    energy_loss /= len(graphs)
    force_loss  /= len(graphs)
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
    order = torch.randperm(len(dataset))
    for i in range(len(dataset)//BATCH_SIZE):
        print(EPOCH, i, train(i))
        if i % 1000 == 0:
            torch.save({'model': model,
                        'optimizer_state_dict': optimizer.state_dict()},
                       f'./models/{EPOCH}_{i}.pkl)')
            np.save(f'./models/{EPOCH}_{i}.npy',np.array(losses))
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
