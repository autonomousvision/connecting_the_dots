import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys

import hyperdepth as hd

sys.path.append('../')
import dataset


def get_data(n, row_from, row_to, train):
  imsizes = [(256,384)]
  focal_lengths = [160]
  dset = dataset.SynDataset(n, imsizes=imsizes, focal_lengths=focal_lengths, train=train)
  ims = np.empty((n, row_to-row_from, imsizes[0][1]), dtype=np.uint8)
  disps = np.empty((n, row_to-row_from, imsizes[0][1]), dtype=np.float32)
  for idx in range(n):
    print(f'load sample {idx} train={train}')
    sample = dset[idx]
    ims[idx] = (sample['im0'][0,row_from:row_to] * 255).astype(np.uint8)
    disps[idx] = sample['disp0'][0,row_from:row_to]
  return ims, disps



params = hd.TrainParams(
  n_trees=4,
  max_tree_depth=,
  n_test_split_functions=50,
  n_test_thresholds=10,
  n_test_samples=4096,
  min_samples_to_split=16,
  min_samples_for_leaf=8)

n_disp_bins = 20
depth_switch = 0

row_from = 100
row_to = 108
n_train_samples = 1024
n_test_samples = 32

train_ims, train_disps = get_data(n_train_samples, row_from, row_to, True)
test_ims, test_disps = get_data(n_test_samples, row_from, row_to, False)

for tree_depth in [8,10,12,14,16]:
  depth_switch = tree_depth - 4

  prefix = f'td{tree_depth}_ds{depth_switch}'
  prefix = Path(f'./forests/{prefix}/')
  prefix.mkdir(parents=True, exist_ok=True)

  hd.train_forest(params, train_ims, train_disps, n_disp_bins=n_disp_bins, depth_switch=depth_switch, forest_prefix=str(prefix / 'fr'))

  es = hd.eval_forest(test_ims, test_disps, n_disp_bins=n_disp_bins, depth_switch=depth_switch, forest_prefix=str(prefix / 'fr')) 

  np.save(str(prefix / 'ta.npy'), test_disps)
  np.save(str(prefix / 'es.npy'), es)

  # plt.figure(); 
  # plt.subplot(2,1,1); plt.imshow(test_disps[0], vmin=0, vmax=4);
  # plt.subplot(2,1,2); plt.imshow(es[0], vmin=0, vmax=4);
  # plt.show()  
