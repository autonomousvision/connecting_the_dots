import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import geometry

def ax3d(fig=None):
  if fig is None:
    fig = plt.gcf()
  return fig.add_subplot(111, projection='3d')

def plot_camera(ax=None, R=np.eye(3), t=np.zeros((3,)), size=25, marker_C='.', color='b', linestyle='-', linewidth=0.1, label=None, **kwargs):
  if ax is None:
    ax = plt.gca()
  C0 = geometry.translation_to_cameracenter(R, t).ravel()
  C1 = C0 + R.T.dot( np.array([[-size],[-size],[3*size]], dtype=np.float32) ).ravel()
  C2 = C0 + R.T.dot( np.array([[-size],[+size],[3*size]], dtype=np.float32) ).ravel()
  C3 = C0 + R.T.dot( np.array([[+size],[+size],[3*size]], dtype=np.float32) ).ravel()
  C4 = C0 + R.T.dot( np.array([[+size],[-size],[3*size]], dtype=np.float32) ).ravel()

  if marker_C != '':
    ax.plot([C0[0]], [C0[1]], [C0[2]], marker=marker_C, color=color, label=label, **kwargs)
  ax.plot([C0[0], C1[0]], [C0[1], C1[1]], [C0[2], C1[2]], color=color, label='_nolegend_', linestyle=linestyle, linewidth=linewidth, **kwargs)
  ax.plot([C0[0], C2[0]], [C0[1], C2[1]], [C0[2], C2[2]], color=color, label='_nolegend_', linestyle=linestyle, linewidth=linewidth, **kwargs)
  ax.plot([C0[0], C3[0]], [C0[1], C3[1]], [C0[2], C3[2]], color=color, label='_nolegend_', linestyle=linestyle, linewidth=linewidth, **kwargs)
  ax.plot([C0[0], C4[0]], [C0[1], C4[1]], [C0[2], C4[2]], color=color, label='_nolegend_', linestyle=linestyle, linewidth=linewidth, **kwargs)
  ax.plot([C1[0], C2[0], C3[0], C4[0], C1[0]], [C1[1], C2[1], C3[1], C4[1], C1[1]], [C1[2], C2[2], C3[2], C4[2], C1[2]], color=color, label='_nolegend_', linestyle=linestyle, linewidth=linewidth, **kwargs)

def axis_equal(ax=None):
  if ax is None:
    ax = plt.gca()
  extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
  sz = extents[:,1] - extents[:,0]
  centers = np.mean(extents, axis=1)
  maxsize = max(abs(sz))
  r = maxsize/2
  for ctr, dim in zip(centers, 'xyz'):
    getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
