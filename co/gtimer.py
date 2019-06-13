import numpy as np

from . import utils

class StopWatch(utils.StopWatch):
  def __del__(self):
    print('='*80)
    print('gtimer:')
    total = ', '.join(['%s: %f[s]' % (k,v) for k,v in self.get(reduce=np.sum).items()])
    print(f'  [total]  {total}')
    mean = ', '.join(['%s: %f[s]' % (k,v) for k,v in self.get(reduce=np.mean).items()])
    print(f'  [mean]   {mean}')
    median = ', '.join(['%s: %f[s]' % (k,v) for k,v in self.get(reduce=np.median).items()])
    print(f'  [median] {median}')
    print('='*80)

GTIMER = StopWatch()

def start(name):
  GTIMER.start(name)
def stop(name):
  GTIMER.stop(name)

class Ctx(object):
  def __init__(self, name):
    self.name = name

  def __enter__(self):
    start(self.name)

  def __exit__(self, *args):
    stop(self.name)
