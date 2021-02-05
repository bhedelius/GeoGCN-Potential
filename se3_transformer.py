# -*- coding: utf-8 -*-
"""se3-transformer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/bhedelius/ccee4e14859b0fa4f7cb6567a1a69777/se3-transformer.ipynb
"""

# Install libraries
!pip install --pre dgl-cu101

# Import libraries
import torch
from torch import nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl import DGLGraph

import numpy as np
from scipy.special import sph_harm as sph_harm_func

from numpy import sqrt
from scipy.special import factorial


def clebsch_gordan(j1,j2,m1,m2,J,M=None):
  ''' Using equation from Wikipedia:
  https://en.wikipedia.org/wiki/Table_of_Clebsch–Gordan_coefficients.

  The equation isn't numerically stable and not fast (speed isn't too important)
  TODO: Possibly find more stable implementation'''
  if M is None:
    M=m1+m2
  if M<0:
    return (-1)**(J-j1-j2)*clebsch_gordan(j1,j2,-m1,-m2,J,-M)
  if j1<j2:
    return (-1)**(J-j1-j2)*clebsch_gordan(j2,j1,m2,m1,J,M)
  if not M==m1+m2:
    return 0
  A = sqrt((2*J+1)*factorial(J+j1-j2)*factorial(J-j1+j2)*factorial(j1+j2-J)/factorial(j1+j2+J+1))
  B = sqrt(factorial(J+M)*factorial(J-M)*factorial(j1-m1)*factorial(j1+m1)*factorial(j2-m2)*factorial(j2+m2))
  k_max = min([j1+j2-J,j1-m1,j2+m2])
  k_min = max([0,-(J-j2+m1),-(J-j1-m2)])
  C = 0
  for k in range(int(k_min), int(k_max)+1):
    C += (-1)**k/(factorial(k)*factorial(j1+j2-J-k)*factorial(j1-m1-k)*factorial(j2+m2-k)*factorial(J-j2+m1+k)*factorial(J-j1-m2+k))
  return A*B*C

def clebsch_gordan_mat(j1,j2,J,m):
  mat = torch.zeros((int(2*j1+1),int(2*j2+1)))
  for i, m1 in enumerate(torch.arange(-j1, j1+1)):
    for j, m2 in enumerate(torch.arange(-j2, j2+1)):
      mat[i,j] = clebsch_gordan(j1,j2,m1,m2,J,m)
  return mat

def clebsch_gordan_mats(j1,j2,J):
  mats = torch.zeros(2*J+1,2*j1+1,2*j2+1)
  for x, m in enumerate(torch.arange(-J, J+1)):
    for y, m1 in enumerate(torch.arange(-j1, j1+1)):
      for z, m2 in enumerate(torch.arange(-j2, j2+1)):
        mats[x,y,z] = clebsch_gordan(j1,j2,m1,m2,J)
  return mats

clebsch_gordan(1/2,1/2,-1/2,1/2,1,0)

class PointCloud:
  '''Represents a point cloud in R^3. This class calculates and stores relevant
  information such as the vectors, distances, directions, and spherical
  harmonics of the vectors.'''
  # new_g, new_ntypes, new_etypes, new_nframes, new_eframes
  def __init__(self, pos, cutoff=8.0):
    edges = self._find_edges_(pos, cutoff)
    self.graph = dgl.graph(edges)
    self.graph.ndata['pos'] = pos
    self._calc_edge_info_()
    self.sph_harm = dict()
    self.wj = dict()

  def _find_edges_(self, pos, cutoff):
    # Use positions to create graph. Need to improve! Currently O(n^2)
    self.vec_mat = pos[:,None,:]-pos[None,:,:]
    self.dist_mat = torch.sqrt(torch.sum((self.vec_mat)**2,axis=-1))
    u = []
    v = []
    for j in range(len(pos)):
      for i in range(j):
        if self.dist_mat[i,j] < cutoff:
          u.append(i)
          v.append(j)
    u, v = torch.tensor(u+v), torch.tensor(v+u)
    return (u,v)

  def _calc_edge_info_(self):
    # Calculate and store position and angle information
    u,v = self.graph.edges()[0], self.graph.edges()[1]
    vec = self.vec_mat[u,v]
    self.graph.edata['vec'] = vec
    dist = self.dist_mat[u,v]
    self.graph.edata['dist'] = dist
    dir = vec/dist[:,None]
    self.graph.edata['dir'] = dir
    self.graph.edata['theta'] = torch.atan2(dir[:,1], dir[:,0])
    self.graph.edata['phi'] = torch.arccos(dir[:,2])

  def get_sph_harm(self, J):
    # Returns spherical harmonic of order J.
    if not J in self.sph_harm.keys():
      m = torch.arange(-J,J+1)
      theta = self.graph.edata['theta']
      phi = self.graph.edata['phi']
      self.sph_harm[J] = torch.real(sph_harm_func(m[None,:], J, theta[:,None], phi[:,None])).double()
    return self.sph_harm[J]
    
  def get_wj(self, l, k):
    # This needs to be improved
    if not (l,k) in self.wj.keys():
      wj = torch.zeros(k+l-abs(k-l)+1, self.graph.number_of_edges(), 2*l+1, 2*k+1)
      for i, J in enumerate(range(abs(k-l), k+l+1)):
        sh = self.get_sph_harm(J)
        cg = clebsch_gordan_mats(l,k,J).double()
        wj[i] = torch.einsum("em,mlk->elk",sh, cg)
      self.wj[(l,k)] = wj.transpose(0,1).clone()
    return self.wj[(l,k)]

class WLayer(nn.Module):
  def __init__(self, k, l, channels=1):
    super(WLayer, self).__init__()
    self.k = k
    self.l = l
    self.channels = channels

    r_size = (channels)*(k+l-abs(k-l)+1)*(2*l+1)*(2*k+1)

    self.radial = nn.Sequential(nn.Linear(1,5), nn.ReLU(),
                                nn.Linear(5,r_size))

  def forward(self, pc):
    wj = pc.get_wj(self.l, self.k)
    R = self.radial(pc.graph.edata['dist'][:,None])
    R = R.reshape((-1,
                   self.channels,
                   self.k+self.l-abs(self.k-self.l)+1,
                   2*self.l+1,
                   2*self.k+1))
    w = torch.einsum('ecjlk,ejlk->eclk',R,wj)
    return w

class TFNLayer(nn.Module):
  def __init__(self, k, l, channels=1):
    super(TFNLayer, self).__init__()
    self.k = k
    self.l = l
    self.channels = channels
    self.wlayer = WLayer(k, l, channels)
    #self.self_int = nn.linear

  def forward(self, pc, feat):
    with pc.graph.local_scope():
      pc.graph.ndata['feat'] = feat
      pc.graph.edata['w'] = self.wlayer(pc)
      pc.graph.update_all(self.message_func, self.reduce_func)
      return pc.graph.ndata['f']

  def message_func(self, edges):
    print(edges.data['w'].size(), edges.src['feat'].size())
    return {'m': torch.einsum('eclk,eck->ecl',edges.data['w'],edges.src['feat'])}

  def reduce_func(self,nodes):
     return {'f': torch.sum(nodes.mailbox['m'], dim=1)}

class TransLayer(nn.Module):
  def __init__(self, k, l, channels=1):
    super(TransLayer, self).__init__()
    self.k = k
    self.l = l
    self.channels = channels

    self.wklayer = WLayer(k, l, channels)
    self.wvlayer = WLayer(k, l, channels)

  def forward(self, pc, feat):
    with pc.graph.local_scope():
      pc.graph.ndata['feat'] = feat
      pc.graph.edata['wk'] = self.wklayer(pc)
      pc.graph.edata['wv'] = self.wvlayer(pc)
      pc.graph.ndata['q'] = torch.sum(feat,-1)
      pc.graph.update_all(self.attn_message, self.attn_reduce)
      pc.graph.update_all(self.message_func, self.reduce_func)
      return pc.graph.ndata['f']

  def attn_message(self, edges):
    q = edges.dst['q']
    wk = edges.data['wk']
    feat = edges.src['feat']
    k = torch.sum(torch.einsum('eclk,eck->ecl',wk,feat),-1)
    exp = torch.exp(
      torch.einsum('ec,ec->ec',q,k))
    edges.data['exp'] = exp
    return {'exp': exp}

  def attn_reduce(self, nodes):
    # does sum over j'
    return {'s': torch.sum(nodes.mailbox['exp'], dim=1)}

  def message_func(self, edges):
    attn = edges.data['exp']/edges.dst['s']
    wv = edges.data['wv']
    feat = edges.src['feat']

    c = torch.einsum('ec,eclk,eck->ecl',attn,wv,feat)
    si = edges.dst['feat']
    return {'c': c+si}

  def reduce_func(self,nodes):
     return {'f': torch.sum(nodes.mailbox['c'], dim=1)}

# Create a test graph
num_pts = 10
pos = torch.rand(num_pts,3)
pc = PointCloud(pos)

feat = torch.rand(num_pts,2,1)

# TFN layer
layer1 = TFNLayer(0,1,channels=2)
layer1(pc, feat)

# Transformer layer
layer = TransLayer(0,1,channels=2)
layer(pc, feat).size()