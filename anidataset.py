# -*- coding: utf-8 -*-
"""ANIDataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12tKs6_CXI9vk3VrkvDqzw-tx25UejuaE
"""

import h5py
import numpy as np

import os

import torch
from torch.utils.data import Dataset

#!pip install dgl-cu101
import dgl

# Install Ani-1x Dataset:
#
#!wget https://s3-eu-west-1.amazonaws.com/pstorage-npg-968563215/18112775/ani1xrelease.h5

def iter_data_buckets(h5filename, keys=['wb97x_dz.energy']):
    """ Iterate over buckets of data in ANI HDF5 file. 
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard('atomic_numbers')
    keys.discard('coordinates')
    with h5py.File(h5filename, 'r') as f:
        for grp in f.values():
            Nc = grp['coordinates'].shape[0]
            mask = np.ones(Nc, dtype=np.bool)
            data = dict((k, grp[k][()]) for k in keys)
            for k in keys:
                v = data[k].reshape(Nc, -1)
                mask = mask & ~np.isnan(v).any(axis=1)
            if not np.sum(mask):
                continue
            d = dict((k, data[k][mask]) for k in keys)
            d['atomic_numbers'] = grp['atomic_numbers'][()]
            d['coordinates'] = grp['coordinates'][()][mask]
            yield d

eye = np.eye(4)
species_dict = {1: eye[0], 6: eye[1], 7: eye[2], 8: eye[3]}

class AniDataset(Dataset):
  def __init__(self, dir='./', cutoff=100):
    super(AniDataset, self).__init__()
    self.parse(dir)
    self.cutoff = cutoff

  def parse(self, dir):
    self.species = []
    self.pos = []
    self.forces = []

    h5file = os.path.join(dir,'ani1xrelease.h5')
    iter = iter_data_buckets(h5file, keys=['mp2_tz.corr_energy','wb97x_tz.forces'])
    for molecule in iter:
      species = molecule['atomic_numbers']
      for pos, forces  in zip(molecule['coordinates'], molecule['wb97x_tz.forces']):
        self.species.append(species)
        self.pos.append(pos)
        self.forces.append(forces)

  @staticmethod
  def get_edges(pos, cutoff):
    dist_mat = np.linalg.norm(pos[None,:,:]-pos[:,None,:],axis=2)

    u = []
    v = []
    N = len(pos)
    for i in range(N):
      for j in range(N):
        if i!=j and dist_mat[i,j]<cutoff:
          u.append(i)
          v.append(j)
    edges = torch.tensor(u, dtype=torch.long), torch.tensor(v, dtype=torch.long)
    return edges

  def __getitem__(self, i):
    pos = self.pos[i]
    species = self.species[i]
    forces = self.forces[i]

    pos = torch.tensor(pos)
    species = torch.tensor([species_dict[atom] for atom in species], dtype=torch.long)
    forces = torch.tensor(forces)

    edges = self.get_edges(pos, self.cutoff)

    graph = dgl.graph(edges)
    graph.ndata['pos'] = pos
    graph.ndata['species'] = species
    graph.ndata['forces'] = forces

    return graph

  def __len__(self):
    return len(self.pos)

#ani_dataset = AniDataset()

#ani_dataset[4000]

#species = ani_dataset.species[0]

#[species_dict[atom] for atom in species]

