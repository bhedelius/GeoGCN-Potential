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
    def __init__(self, dir='./', mode='train'):
        super(AniDataset, self).__init__()
        self.mode = mode
        self.parse(dir)

    def parse(self, dir):
        self.species = []
        self.pos = []
        self.forces = []
        self.energies = []
        h5file = os.path.join(dir,'ani1xrelease.h5')
        iter = iter_data_buckets(h5file, keys=['wb97x_dz.forces', 'wb97x_dz.energy'])

        for i, molecule in enumerate(iter):
            if (self.mode=='train' and i%5 != 0) or (self.mode=='val' and i%10 == 0) or (self.mode=='test' and i%10 == 5): 
                species = molecule['atomic_numbers']
                for pos, forces, energy in zip(molecule['coordinates'], molecule['wb97x_dz.forces'], molecule['wb97x_dz.energy']): 
                    self.species.append(species)
                    self.pos.append(pos)
                    self.forces.append(forces)
                    self.energies.append(energy)

    def __getitem__(self, i):
        pos = self.pos[i]
        species = self.species[i]
        forces = self.forces[i]
        energy = self.energies[i]

        pos = torch.tensor(pos)
        species = torch.tensor([species_dict[atom] for atom in species], dtype=torch.long)
        forces = torch.tensor(forces)

        return pos, species, forces, energy

    def __len__(self):
        return len(self.pos)

    def sample_batches(self, n_batches=4, max_seq_length=10000, max_n_edges=30000):
        L = len(self)
        position_tensor = torch.zeros(n_batches, max_seq_length, 3)
        species_tensor = torch.zeros(n_batches, max_seq_length, 4)
        forces_tensor = torch.zeros(n_batches, max_seq_length, 3)
        energy_tensor = torch.zeros(n_batches, max_seq_length)
        seq_length_tensor = torch.zeros(n_batches, max_seq_length)

        for i in range(n_batches):
            idx = 0

            # Get data for first item
            m = np.random.randint(L)
            pos, species, forces, energy = self[m]
            l = len(pos)

            running_n_edges = l**2
            j = 0

            while running_n_edges < max_n_edges:
                # Load data into vectors
                position_tensor[i, idx:idx+l, :] = pos
                species_tensor[i, idx:idx+l, :] = species
                forces_tensor[i, idx:idx+l, :] = forces
                energy_tensor[i, j] = energy
                seq_length_tensor[i, j] = l

                idx += l

                # Get data for next item
                m = np.random.randint(L)
                pos, species, forces, energy = self[m]
                l = len(pos)

                running_n_edges += l**2
                j += 1

        return position_tensor, species_tensor, forces_tensor, energy_tensor, seq_length_tensor
