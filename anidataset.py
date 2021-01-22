import torch
from torch.utils.data import DataLoader, Dataset

import dgl

import os
import pyanitools as pya

species_dict = {'H': 0, 'C': 1, 'N': 2, 'O': 3}

class AniDataset(Dataset):
  def __init__(self, dir='./ANI-1_release'):
    super(AniDataset, self).__init__()
    self.parse(dir)

  def parse(self, dir):
    self.species = []
    self.pos = []
    self.energies = []
    for i in range(1,2):# 9):
      hdf5file = os.path.join(dir,'ani_gdb_s0{}.h5'.format(i))
      adl = pya.anidataloader(hdf5file)
      for molecule in adl:
        species = molecule['species']
        for pos, energy  in zip(molecule['coordinates'], molecule['energies']):
          self.species.append(species)
          self.pos.append(pos)
          self.energies.append(energy)

  @staticmethod
  def get_edges(pos):
    dist_mat = np.linalg.norm(pos[None,:,:]-pos[:,None,:],axis=2)

    u = []
    v = []
    N = len(pos)
    for i in range(N):
      for j in range(N):
        if i!=j and dist_mat[i,j]<5:
          u.append(i)
          v.append(j)
    edges = torch.tensor(u, dtype=torch.long), torch.tensor(v, dtype=torch.long)
    return edges

  def __getitem__(self, i):
    pos = self.pos[i]
    species = self.species[i]
    energy = self.energies[i]

    pos = torch.tensor(pos)
    u, v = self.get_edges(pos)
    species = torch.tensor([species_dict[atom] for atom in species], dtype=torch.long)
    energy = torch.tensor(energy)

    graph = dgl.graph(edges)
    graph.ndata['pos'] = pos
    graph.ndata['species'] = species
    graph.energy = energy
    return graph

  def __len__(self):
    return len(self.energies)
