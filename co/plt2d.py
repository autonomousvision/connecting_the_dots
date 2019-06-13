import numpy as np
import matplotlib.pyplot as plt

from . import geometry

def image_matrix(ims, bgval=0):
  n = ims.shape[0]
  m = int( np.ceil(np.sqrt(n)) )
  h = ims.shape[1]
  w = ims.shape[2]
  mat = np.empty((m*h, m*w), dtype=ims.dtype)
  mat.fill(bgval)
  idx = 0
  for r in range(m):
    for c in range(m):
      if idx < n:
        mat[r*h:(r+1)*h, c*w:(c+1)*w] = ims[idx]
        idx += 1
  return mat

def image_cat(ims, vertical=False):
  offx = [0]
  offy = [0]
  if vertical:
    width = max([im.shape[1] for im in ims])
    offx += [0 for im in ims[:-1]]
    offy += [im.shape[0] for im in ims[:-1]]
    height = sum([im.shape[0] for im in ims])
  else:
    height = max([im.shape[0] for im in ims])
    offx += [im.shape[1] for im in ims[:-1]]
    offy += [0 for im in ims[:-1]]
    width = sum([im.shape[1] for im in ims])
  offx = np.cumsum(offx)
  offy = np.cumsum(offy)

  im = np.zeros((height,width,*ims[0].shape[2:]), dtype=ims[0].dtype)
  for im0, ox, oy in zip(ims, offx, offy):
    im[oy:oy + im0.shape[0], ox:ox + im0.shape[1]] = im0

  return im, offx, offy

def line(li, h, w, ax=None, *args, **kwargs):
  if ax is None:
    ax = plt.gca()
  xs = (-li[2] - li[1] * np.array((0, h-1))) / li[0]
  ys = (-li[2] - li[0] * np.array((0, w-1))) / li[1]
  pts = np.array([(0,ys[0]), (w-1, ys[1]), (xs[0], 0), (xs[1], h-1)])
  pts = pts[np.logical_and(np.logical_and(pts[:,0] >= 0, pts[:,0] < w), np.logical_and(pts[:,1] >= 0, pts[:,1] < h))]
  ax.plot(pts[:,0], pts[:,1], *args, **kwargs)

def depthshow(depth, *args, ax=None, **kwargs):
  if ax is None:
    ax = plt.gca()
  d = depth.copy()
  d[d < 0] = np.NaN
  ax.imshow(d, *args, **kwargs)
