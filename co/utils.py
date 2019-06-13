import numpy as np
import pandas as pd
import time
from collections import OrderedDict
import argparse
import os
import re
import pickle
import subprocess

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class StopWatch(object):
  def __init__(self):
    self.timings = OrderedDict()
    self.starts = {}

  def start(self, name):
    self.starts[name] = time.time()

  def stop(self, name):
    if name not in self.timings:
      self.timings[name] = []
    self.timings[name].append(time.time() - self.starts[name])

  def get(self, name=None, reduce=np.sum):
    if name is not None:
      return reduce(self.timings[name])
    else:
      ret = {}
      for k in self.timings:
        ret[k] = reduce(self.timings[k])
      return ret

  def __repr__(self):
    return ', '.join(['%s: %f[s]' % (k,v) for k,v in self.get().items()])
  def __str__(self):
    return ', '.join(['%s: %f[s]' % (k,v) for k,v in self.get().items()])

class ETA(object):
  def __init__(self, length):
    self.length = length
    self.start_time = time.time()
    self.current_idx = 0
    self.current_time = time.time()

  def update(self, idx):
    self.current_idx = idx
    self.current_time = time.time()

  def get_elapsed_time(self):
    return self.current_time - self.start_time

  def get_item_time(self):
    return self.get_elapsed_time() / (self.current_idx + 1)

  def get_remaining_time(self):
    return self.get_item_time() * (self.length - self.current_idx + 1)

  def format_time(self, seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    hours = int(hours)
    minutes = int(minutes)
    return f'{hours:02d}:{minutes:02d}:{seconds:05.2f}'

  def get_elapsed_time_str(self):
    return self.format_time(self.get_elapsed_time())

  def get_remaining_time_str(self):
    return self.format_time(self.get_remaining_time())

def git_hash(cwd=None):
  ret = subprocess.run(['git', 'describe', '--always'], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  hash = ret.stdout
  if hash is not None and 'fatal' not in hash.decode():
    return hash.decode().strip()
  else:
    return None

