import numpy as np
from . import geometry

def _process_inputs(estimate, target, mask):
  if estimate.shape != target.shape:
    raise Exception('estimate and target have to be same shape')
  if mask is None:
    mask = np.ones(estimate.shape, dtype=np.bool)
  else:
    mask = mask != 0
  if estimate.shape != mask.shape:
    raise Exception('estimate and mask have to be same shape')
  return estimate, target, mask

def mse(estimate, target, mask=None):
  estimate, target, mask = _process_inputs(estimate, target, mask)
  m = np.sum((estimate[mask] - target[mask])**2) / mask.sum()
  return m

def rmse(estimate, target, mask=None):
  return np.sqrt(mse(estimate, target, mask))

def mae(estimate, target, mask=None):
  estimate, target, mask = _process_inputs(estimate, target, mask)
  m = np.abs(estimate[mask] - target[mask]).sum() / mask.sum()
  return m

def outlier_fraction(estimate, target, mask=None, threshold=0):
  estimate, target, mask = _process_inputs(estimate, target, mask)
  diff = np.abs(estimate[mask] - target[mask])
  m = (diff > threshold).sum() / mask.sum()
  return m


class Metric(object):
  def __init__(self, str_prefix=''):
    self.str_prefix = str_prefix
    self.reset()

  def reset(self):
    pass

  def add(self, es, ta, ma=None):
    pass

  def get(self):
    return {}

  def items(self):
    return self.get().items()

  def __str__(self):
    return ', '.join([f'{self.str_prefix}{key}={value:.5f}' for key, value in self.get().items()])

class MultipleMetric(Metric):
  def __init__(self, *metrics, **kwargs):
    self.metrics = [*metrics]
    super().__init__(**kwargs)

  def reset(self):
    for m in self.metrics:
      m.reset()

  def add(self, es, ta, ma=None):
    for m in self.metrics:
      m.add(es, ta, ma)

  def get(self):
    ret = {}
    for m in self.metrics:
      vals = m.get()
      for k in vals:
        ret[k] = vals[k]
    return ret

  def __str__(self):
    return '\n'.join([str(m) for m in self.metrics])

class BaseDistanceMetric(Metric):
  def __init__(self, name='', **kwargs):
    super().__init__(**kwargs)
    self.name = name

  def reset(self):
    self.dists = []

  def add(self, es, ta, ma=None):
    pass

  def get(self):
    dists = np.hstack(self.dists)
    return {
      f'dist{self.name}_mean': float(np.mean(dists)),
      f'dist{self.name}_std': float(np.std(dists)),
      f'dist{self.name}_median': float(np.median(dists)),
      f'dist{self.name}_q10': float(np.percentile(dists, 10)),
      f'dist{self.name}_q90': float(np.percentile(dists, 90)),
      f'dist{self.name}_min': float(np.min(dists)),
      f'dist{self.name}_max': float(np.max(dists)),
    }

class DistanceMetric(BaseDistanceMetric):
  def __init__(self, vec_length, p=2, **kwargs):
    super().__init__(name=f'{p}', **kwargs)
    self.vec_length = vec_length
    self.p = p

  def add(self, es, ta, ma=None):
    if es.shape != ta.shape or es.shape[1] != self.vec_length or es.ndim != 2:
      print(es.shape, ta.shape)
      raise Exception('es and ta have to be of shape Nxdim')
    if ma is not None:
      es = es[ma != 0]
      ta = ta[ma != 0]
    dist = np.linalg.norm(es - ta, ord=self.p, axis=1)
    self.dists.append( dist )

class OutlierFractionMetric(DistanceMetric):
  def __init__(self, thresholds, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.thresholds = thresholds

  def get(self):
    dists = np.hstack(self.dists)
    ret = {}
    for t in self.thresholds:
      ma = dists > t
      ret[f'of{t}'] = float(ma.sum() / ma.size)
    return ret

class RelativeDistanceMetric(BaseDistanceMetric):
  def __init__(self, vec_length, p=2, **kwargs):
    super().__init__(name=f'rel{p}', **kwargs)
    self.vec_length = vec_length
    self.p = p

  def add(self, es, ta, ma=None):
    if es.shape != ta.shape or es.shape[1] != self.vec_length or es.ndim != 2:
      raise Exception('es and ta have to be of shape Nxdim')
    dist = np.linalg.norm(es - ta, ord=self.p, axis=1)
    denom = np.linalg.norm(ta, ord=self.p, axis=1)
    dist /= denom
    if ma is not None:
      dist = dist[ma != 0]
    self.dists.append( dist )

class RotmDistanceMetric(BaseDistanceMetric):
  def __init__(self, type='identity', **kwargs):
    super().__init__(name=type, **kwargs)
    self.type = type

  def add(self, es, ta, ma=None):
    if es.shape != ta.shape or es.shape[1] != 3 or es.shape[2] != 3 or es.ndim != 3:
      print(es.shape, ta.shape)
      raise Exception('es and ta have to be of shape Nx3x3')
    if ma is not None:
      raise Exception('mask is not implemented')
    if self.type == 'identity':
      self.dists.append( geometry.rotm_distance_identity(es, ta) )
    elif self.type == 'geodesic':
      self.dists.append( geometry.rotm_distance_geodesic_unit_sphere(es, ta) )
    else:
      raise Exception('invalid distance type')

class QuaternionDistanceMetric(BaseDistanceMetric):
  def __init__(self, type='angle', **kwargs):
    super().__init__(name=type, **kwargs)
    self.type = type

  def add(self, es, ta, ma=None):
    if es.shape != ta.shape or es.shape[1] != 4 or es.ndim != 2:
      print(es.shape, ta.shape)
      raise Exception('es and ta have to be of shape Nx4')
    if ma is not None:
      raise Exception('mask is not implemented')
    if self.type == 'angle':
      self.dists.append( geometry.quat_distance_angle(es, ta) )
    elif self.type == 'mineucl':
      self.dists.append( geometry.quat_distance_mineucl(es, ta) )
    elif self.type == 'normdiff':
      self.dists.append( geometry.quat_distance_normdiff(es, ta) )
    else:
      raise Exception('invalid distance type')


class BinaryAccuracyMetric(Metric):
  def __init__(self, thresholds=np.linspace(0.0, 1.0, num=101, dtype=np.float64)[:-1], **kwargs):
    self.thresholds = thresholds
    super().__init__(**kwargs)

  def reset(self):
    self.tps = [0 for wp in self.thresholds]
    self.fps = [0 for wp in self.thresholds]
    self.fns = [0 for wp in self.thresholds]
    self.tns = [0 for wp in self.thresholds]
    self.n_pos = 0
    self.n_neg = 0

  def add(self, es, ta, ma=None):
    if ma is not None:
      raise Exception('mask is not implemented')
    es = es.ravel()
    ta = ta.ravel()
    if es.shape[0] != ta.shape[0]:
      raise Exception('invalid shape of es, or ta')
    if es.min() < 0 or es.max() > 1:
      raise Exception('estimate has wrong value range')
    ta_p = (ta == 1)
    ta_n = (ta == 0)
    es_p = es[ta_p]
    es_n = es[ta_n]
    for idx, wp in enumerate(self.thresholds):
      wp = np.asscalar(wp)
      self.tps[idx] += (es_p > wp).sum()
      self.fps[idx] += (es_n > wp).sum()
      self.fns[idx] += (es_p <= wp).sum()
      self.tns[idx] += (es_n <= wp).sum()
    self.n_pos += ta_p.sum()
    self.n_neg += ta_n.sum()

  def get(self):
    tps = np.array(self.tps).astype(np.float32)
    fps = np.array(self.fps).astype(np.float32)
    fns = np.array(self.fns).astype(np.float32)
    tns = np.array(self.tns).astype(np.float32)
    wp = self.thresholds

    ret = {}

    precisions = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=tps + fps != 0)
    recalls = np.divide(tps, tps + fns, out=np.zeros_like(tps), where=tps + fns != 0) # tprs
    fprs = np.divide(fps, fps + tns, out=np.zeros_like(tps), where=fps + tns != 0)

    precisions = np.r_[0, precisions, 1]
    recalls = np.r_[1, recalls, 0]
    fprs = np.r_[1, fprs, 0]

    ret['auc'] = float(-np.trapz(recalls, fprs))
    ret['prauc'] = float(-np.trapz(precisions, recalls))
    ret['ap'] = float(-(np.diff(recalls) * precisions[:-1]).sum())

    accuracies = np.divide(tps + tns, tps + tns + fps + fns)
    aacc = np.mean(accuracies)
    for t in np.linspace(0,1,num=11)[1:-1]:
      idx = np.argmin(np.abs(t - wp))
      ret[f'acc{wp[idx]:.2f}'] = float(accuracies[idx])

    return ret
