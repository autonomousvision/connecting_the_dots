import torch
import torch.utils.data
import numpy as np

class TestSet(object):
  def __init__(self, name, dset, test_frequency=1):
    self.name = name
    self.dset = dset
    self.test_frequency = test_frequency

class TestSets(list):
  def append(self, name, dset, test_frequency=1):
    super().append(TestSet(name, dset, test_frequency))



class MultiDataset(torch.utils.data.Dataset):
  def __init__(self, *datasets):
    self.current_epoch = 0

    self.datasets = []
    self.cum_n_samples = [0]

    for dataset in datasets:
      self.append(dataset)

  def append(self, dataset):
    self.datasets.append(dataset)
    self.__update_cum_n_samples(dataset)

  def __update_cum_n_samples(self, dataset):
    n_samples = self.cum_n_samples[-1] + len(dataset)
    self.cum_n_samples.append(n_samples)

  def dataset_updated(self):
    self.cum_n_samples = [0]
    for dset in self.datasets:
      self.__update_cum_n_samples(dset)

  def __len__(self):
    return self.cum_n_samples[-1]

  def __getitem__(self, idx):
    didx = np.searchsorted(self.cum_n_samples, idx, side='right') - 1
    sidx = idx - self.cum_n_samples[didx]
    return self.datasets[didx][sidx]



class BaseDataset(torch.utils.data.Dataset):
  def __init__(self, train=True, fix_seed_per_epoch=False):
    self.current_epoch = 0
    self.train = train
    self.fix_seed_per_epoch = fix_seed_per_epoch

  def get_rng(self, idx):
    rng = np.random.RandomState()
    if self.train:
      if self.fix_seed_per_epoch:
        seed = 1 * len(self) + idx
      else:
        seed = (self.current_epoch + 1) * len(self) + idx
      rng.seed(seed)
    else:
      rng.seed(idx)
    return rng
