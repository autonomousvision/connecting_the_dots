import numpy as np
import matplotlib as mpl
from matplotlib import _pylab_helpers
from matplotlib.rcsetup import interactive_bk as _interactive_bk
import matplotlib.pyplot as plt
import os
import time

def save(path, remove_axis=False, dpi=300, fig=None):
  if fig is None:
    fig = plt.gcf()
  dirname = os.path.dirname(path)
  if dirname != '' and not os.path.exists(dirname):
    os.makedirs(dirname)
  if remove_axis:
    for ax in fig.axes:
      ax.axis('off')
      ax.margins(0,0)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    for ax in fig.axes:
      ax.xaxis.set_major_locator(plt.NullLocator())
      ax.yaxis.set_major_locator(plt.NullLocator())
  fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)

def color_map(im_, cmap='viridis', vmin=None, vmax=None):
  cm = plt.get_cmap(cmap)
  im = im_.copy()
  if vmin is None:
    vmin = np.nanmin(im)
  if vmax is None:
    vmax = np.nanmax(im)
  mask = np.logical_not(np.isfinite(im))
  im[mask] = vmin
  im = (im.clip(vmin, vmax) - vmin) / (vmax - vmin)
  im = cm(im)
  im = im[...,:3]
  for c in range(3):
    im[mask, c] = 1
  return im

def interactive_legend(leg=None, fig=None, all_axes=True):
  if leg is None:
    leg = plt.legend()
  if fig is None:
    fig = plt.gcf()
  if all_axes:
    axs = fig.get_axes()
  else:
    axs = [fig.gca()]

  # lined = dict()
  # lines = ax.lines
  # for legline, origline in zip(leg.get_lines(), ax.lines):
  #   legline.set_picker(5)
  #   lined[legline] = origline
  lined = dict()
  for lidx, legline in enumerate(leg.get_lines()):
    legline.set_picker(5)
    lined[legline] = [ax.lines[lidx] for ax in axs]

  def onpick(event):
    if event.mouseevent.dblclick:
      tmp = [(k,v) for k,v in lined.items()]
    else:
      tmp = [(event.artist, lined[event.artist])]

    for legline, origline in tmp:
      for ol in origline:
        vis = not ol.get_visible()
        ol.set_visible(vis)
      if vis:
        legline.set_alpha(1.0)
      else:
        legline.set_alpha(0.2)
    fig.canvas.draw()

  fig.canvas.mpl_connect('pick_event', onpick)

def non_annoying_pause(interval, focus_figure=False):
  # https://github.com/matplotlib/matplotlib/issues/11131
  backend = mpl.rcParams['backend']
  if backend in _interactive_bk:
    figManager = _pylab_helpers.Gcf.get_active()
    if figManager is not None:
      canvas = figManager.canvas
      if canvas.figure.stale:
        canvas.draw()
      if focus_figure:
        plt.show(block=False)
      canvas.start_event_loop(interval)
      return
  time.sleep(interval)

def remove_all_ticks(fig=None):
  if fig is None:
    fig = plt.gcf()
  for ax in fig.axes:
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
