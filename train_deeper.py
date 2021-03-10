import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

import dgl

from se3_transformer import PointCloud
from model_deeper import Model

from anidataset import *

import pickle as pkl

def load(fname): 
        with open(fname, 'rb') as handle:
                b = pkl.load(handle)    
        return b                       
                    

try:
	print(type(dataset))
except:
	print('Loading dataset')
	dataset = AniDataset() 


torch.cuda.empty_cache()

#Let's see if this trains!
losses = []

dev = 'cuda:1'

EPOCHS = 100

#setup the model
model = Model().to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, eps=1e-4)
objective = nn.MSELoss()

BATCH_SIZE = 50

def train(i):
    graphs = []
    for idx in order[i:min(i+BATCH_SIZE,len(dataset))]:
        pos, species, forces = dataset[idx] 
        pc = PointCloud(pos=pos, cutoff = 1000)
        graph = pc.graph
        graph.ndata[0] = species.unsqueeze(-1).float()
        graph.ndata['forces'] = forces
        graphs.append(graph)
    graph = dgl.batch(graphs)
    pc = PointCloud(graph=graph)
    pc.to(dev)
    #build the point graph
    f = {0: pc.graph.ndata[0]}
    y_hat = model(pc, f)
    loss = objective(y_hat[1][:,0,:], pc.graph.ndata['forces'])
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if total_norm == total_norm:
        optimizer.step()
    else:
        print('gradient update skipped bc of Nans')
    losses.append(loss.item())
    return loss.item()


for EPOCH in range(EPOCHS):
    order = torch.randperm(len(dataset))
    for i in range(len(dataset)//BATCH_SIZE):
        print(EPOCH, i, train(i))
        if i % 1000 == 0:
            torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses}, './models_deeper/'+str(EPOCH)+'_'+str(i))

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




