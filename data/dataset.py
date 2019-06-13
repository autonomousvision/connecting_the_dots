import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import json
import time
from pathlib import Path
import collections
import cv2
import sys
import os
import time
import glob

import torchext
import renderer
import co
from .commons import get_patterns, augment_image

from mpl_toolkits.mplot3d import Axes3D

class TrackSynDataset(torchext.BaseDataset):
  '''
  Load locally saved synthetic dataset
  Please run ./create_syn_data.sh to generate the dataset
  '''
  def __init__(self, settings_path, sample_paths, track_length=2, train=True, data_aug=False):
    super().__init__(train=train)

    self.settings_path = settings_path
    self.sample_paths = sample_paths
    self.data_aug = data_aug
    self.train = train
    self.track_length=track_length
    assert(track_length<=4)

    with open(str(settings_path), 'rb') as f:
      settings = pickle.load(f)
    self.imsizes = settings['imsizes']
    self.patterns = settings['patterns']
    self.focal_lengths = settings['focal_lengths']
    self.baseline = settings['baseline']
    self.K = settings['K']

    self.scale = len(self.imsizes)

    self.max_shift=0
    self.max_blur=0.5
    self.max_noise=3.0
    self.max_sp_noise=0.0005

  def __len__(self):
    return len(self.sample_paths)

  def __getitem__(self, idx):
    if not self.train:
      rng = self.get_rng(idx)
    else:
      rng = np.random.RandomState()
    sample_path = self.sample_paths[idx]

    if self.train:
      track_ind = np.random.permutation(4)[0:self.track_length]
    else:
      track_ind = [0]

    ret = {}
    ret['id'] = idx

    # load imgs, at all scales
    for sidx in range(len(self.imsizes)):
      imgs = []
      ambs = []
      grads = []
      for tidx in track_ind:
        imgs.append(np.load(os.path.join(sample_path,f'im{sidx}_{tidx}.npy')))
        ambs.append(np.load(os.path.join(sample_path,f'ambient{sidx}_{tidx}.npy')))
        grads.append(np.load(os.path.join(sample_path,f'grad{sidx}_{tidx}.npy')))
      ret[f'im{sidx}'] = np.stack(imgs, axis=0)
      ret[f'ambient{sidx}'] = np.stack(ambs, axis=0)
      ret[f'grad{sidx}'] = np.stack(grads, axis=0)

    # load disp and grad only at full resolution
    disps = []
    R = []
    t = []
    for tidx in track_ind:
      disps.append(np.load(os.path.join(sample_path,f'disp0_{tidx}.npy')))
      R.append(np.load(os.path.join(sample_path,f'R_{tidx}.npy')))
      t.append(np.load(os.path.join(sample_path,f't_{tidx}.npy')))
    ret[f'disp0'] = np.stack(disps, axis=0)
    ret['R'] = np.stack(R, axis=0)
    ret['t'] = np.stack(t, axis=0)

    blend_im = np.load(os.path.join(sample_path,'blend_im.npy'))
    ret['blend_im'] = blend_im.astype(np.float32)

    #### apply data augmentation at different scales seperately, only work for max_shift=0
    if self.data_aug:
      for sidx in range(len(self.imsizes)):
        if sidx==0:
          img = ret[f'im{sidx}']
          disp = ret[f'disp{sidx}']
          grad = ret[f'grad{sidx}']
          img_aug = np.zeros_like(img)
          disp_aug = np.zeros_like(img)
          grad_aug = np.zeros_like(img)
          for i in range(img.shape[0]):
            img_aug_, disp_aug_, grad_aug_ = augment_image(img[i,0],rng,
                    disp=disp[i,0],grad=grad[i,0],
                    max_shift=self.max_shift, max_blur=self.max_blur, 
                    max_noise=self.max_noise, max_sp_noise=self.max_sp_noise)
            img_aug[i] = img_aug_[None].astype(np.float32)
            disp_aug[i] = disp_aug_[None].astype(np.float32)
            grad_aug[i] = grad_aug_[None].astype(np.float32)
          ret[f'im{sidx}'] = img_aug
          ret[f'disp{sidx}'] = disp_aug
          ret[f'grad{sidx}'] = grad_aug
        else:
          img = ret[f'im{sidx}']
          img_aug = np.zeros_like(img)
          for i in range(img.shape[0]):
            img_aug_, _, _ = augment_image(img[i,0],rng,
                    max_shift=self.max_shift, max_blur=self.max_blur, 
                    max_noise=self.max_noise, max_sp_noise=self.max_sp_noise)
            img_aug[i] = img_aug_[None].astype(np.float32)
          ret[f'im{sidx}'] = img_aug

    if len(track_ind)==1:
      for key, val in ret.items():
        if key!='blend_im' and key!='id':
          ret[key] = val[0]


    return ret

  def getK(self, sidx=0):
    K = self.K.copy() / (2**sidx)
    K[2,2] = 1
    return K

        

if __name__ == '__main__':
  pass

