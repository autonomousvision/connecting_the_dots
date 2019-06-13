import torch
import numpy as np
import time
from pathlib import Path
import logging
import sys
import itertools
import json
import matplotlib.pyplot as plt
import co
import torchext
from model import networks
from data import dataset

class Worker(torchext.Worker):
  def __init__(self, args, num_workers=18, train_batch_size=8, test_batch_size=8, save_frequency=1, **kwargs):
    super().__init__(args.output_dir, args.exp_name, epochs=args.epochs, num_workers=num_workers, train_batch_size=train_batch_size, test_batch_size=test_batch_size, save_frequency=save_frequency, **kwargs)

    self.ms = args.ms
    self.pattern_path = args.pattern_path
    self.lcn_radius = args.lcn_radius
    self.dp_weight = args.dp_weight
    self.data_type = args.data_type 

    self.imsizes = [(480,640)]
    for iter in range(3):
      self.imsizes.append((int(self.imsizes[-1][0]/2), int(self.imsizes[-1][1]/2)))

    with open('config.json') as fp:
      config = json.load(fp)
      data_root = Path(config['DATA_ROOT'])
    self.settings_path = data_root / self.data_type / 'settings.pkl'
    sample_paths = sorted((data_root / self.data_type).glob('0*/'))

    self.train_paths = sample_paths[2**10:]
    self.test_paths = sample_paths[:2**8]

    # supervise the edge encoder with only 2**8 samples
    self.train_edge = len(self.train_paths) - 2**8

    self.lcn_in = networks.LCN(self.lcn_radius, 0.05)
    self.disparity_loss = networks.DisparityLoss()
    self.edge_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.1]).to(self.train_device))

    # evaluate in the region where opencv Block Matching has valid values
    self.eval_mask = np.zeros(self.imsizes[0])
    self.eval_mask[13:self.imsizes[0][0]-13, 140:self.imsizes[0][1]-13]=1
    self.eval_mask = self.eval_mask.astype(np.bool)
    self.eval_h = self.imsizes[0][0]-2*13
    self.eval_w = self.imsizes[0][1]-13-140

  def get_train_set(self):
    train_set = dataset.TrackSynDataset(self.settings_path, self.train_paths, train=True, data_aug=True, track_length=1)

    return train_set

  def get_test_sets(self):
    test_sets = torchext.TestSets()
    test_set = dataset.TrackSynDataset(self.settings_path, self.test_paths, train=False, data_aug=True, track_length=1)
    test_sets.append('simple', test_set, test_frequency=1)

    # initialize photometric loss modules according to image sizes
    self.losses = []
    for imsize, pat in zip(test_set.imsizes, test_set.patterns):
      pat = pat.mean(axis=2)
      pat = torch.from_numpy(pat[None][None].astype(np.float32))
      pat = pat.to(self.train_device)
      self.lcn_in = self.lcn_in.to(self.train_device)
      pat,_ = self.lcn_in(pat)
      pat = torch.cat([pat for idx in range(3)], dim=1)
      self.losses.append( networks.RectifiedPatternSimilarityLoss(imsize[0],imsize[1], pattern=pat) )

    return test_sets

  def copy_data(self, data, device, requires_grad, train):
    self.lcn_in = self.lcn_in.to(device)

    self.data = {}
    for key, val in data.items():
      grad = 'im' in key and requires_grad
      self.data[key] = val.to(device).requires_grad_(requires_grad=grad)

      # apply lcn to IR input
      # concatenate the normalized IR input and the original IR image
      if 'im' in key and 'blend' not in key:
        im = self.data[key]
        im_lcn,im_std = self.lcn_in(im)
        im_cat = torch.cat((im_lcn, im), dim=1)
        key_std = key.replace('im','std')
        self.data[key]=im_cat
        self.data[key_std] = im_std.to(device).detach()

  def net_forward(self, net, train):
    out = net(self.data['im0'])
    return out

  def loss_forward(self, out, train):
    out, edge = out
    if not(isinstance(out, tuple) or isinstance(out, list)):
      out = [out]
    if not(isinstance(edge, tuple) or isinstance(edge, list)):
      edge = [edge]

    vals = []

    # apply photometric loss
    for s,l,o in zip(itertools.count(), self.losses, out):
      val, pattern_proj = l(o, self.data[f'im{s}'][:,0:1,...], self.data[f'std{s}'])
      if s == 0: 
        self.pattern_proj = pattern_proj.detach()
      vals.append(val)

    # apply disparity loss
    # 1-edge as ground truth edge if inversed
    edge0 = 1-torch.sigmoid(edge[0])
    val = self.disparity_loss(out[0], edge0)
    if self.dp_weight>0:
      vals.append(val * self.dp_weight)

    # apply edge loss on a subset of training samples
    for s,e in zip(itertools.count(), edge):
      # inversed ground truth edge where 0 means edge
      grad = self.data[f'grad{s}']<0.2
      grad = grad.to(torch.float32)
      ids = self.data['id']
      mask = ids>self.train_edge
      if mask.sum()>0:
        val = self.edge_loss(e[mask], grad[mask])
      else:
        val = torch.zeros_like(vals[0]) 
      if s == 0:
        self.edge = e.detach()
        self.edge = torch.sigmoid(self.edge)
        self.edge_gt = grad.detach() 
      vals.append(val)

    return vals

  def numpy_in_out(self, output):
    output, edge = output
    if not(isinstance(output, tuple) or isinstance(output, list)):
      output = [output]
    es = output[0].detach().to('cpu').numpy()
    gt = self.data['disp0'].to('cpu').numpy().astype(np.float32)
    im = self.data['im0'][:,0:1,...].detach().to('cpu').numpy()

    ma = gt>0
    return es, gt, im, ma

  def write_img(self, out_path, es, gt, im, ma):
    logging.info(f'write img {out_path}')
    u_pos, _ = np.meshgrid(range(es.shape[1]), range(es.shape[0]))

    diff = np.abs(es - gt)

    vmin, vmax = np.nanmin(gt), np.nanmax(gt)
    vmin = vmin - 0.2*(vmax-vmin)
    vmax = vmax + 0.2*(vmax-vmin)

    pattern_proj = self.pattern_proj.to('cpu').numpy()[0,0]
    im_orig = self.data['im0'].detach().to('cpu').numpy()[0,0]
    pattern_diff = np.abs(im_orig - pattern_proj)


    fig = plt.figure(figsize=(16,16))
    es_ = co.cmap.color_depth_map(es, scale=vmax)
    gt_ = co.cmap.color_depth_map(gt, scale=vmax)
    diff_ = co.cmap.color_error_image(diff, BGR=True)

    # plot disparities, ground truth disparity is shown only for reference
    ax = plt.subplot(3,3,1); plt.imshow(es_[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'Disparity Est. {es.min():.4f}/{es.max():.4f}')
    ax = plt.subplot(3,3,2); plt.imshow(gt_[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'Disparity GT {np.nanmin(gt):.4f}/{np.nanmax(gt):.4f}')
    ax = plt.subplot(3,3,3); plt.imshow(diff_[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'Disparity Err. {diff.mean():.5f}')

    # plot edges
    edge = self.edge.to('cpu').numpy()[0,0]
    edge_gt = self.edge_gt.to('cpu').numpy()[0,0]
    edge_err = np.abs(edge - edge_gt)
    ax = plt.subplot(3,3,4); plt.imshow(edge, cmap='gray'); plt.xticks([]); plt.yticks([]); ax.set_title(f'Edge Est. {edge.min():.5f}/{edge.max():.5f}')
    ax = plt.subplot(3,3,5); plt.imshow(edge_gt, cmap='gray'); plt.xticks([]); plt.yticks([]); ax.set_title(f'Edge GT {edge_gt.min():.5f}/{edge_gt.max():.5f}')
    ax = plt.subplot(3,3,6); plt.imshow(edge_err, cmap='gray'); plt.xticks([]); plt.yticks([]); ax.set_title(f'Edge Err. {edge_err.mean():.5f}')

    # plot normalized IR input and warped pattern 
    ax = plt.subplot(3,3,7); plt.imshow(im, vmin=im.min(), vmax=im.max(), cmap='gray'); plt.xticks([]); plt.yticks([]); ax.set_title(f'IR input {im.mean():.5f}/{im.std():.5f}')
    ax = plt.subplot(3,3,8); plt.imshow(pattern_proj, vmin=im.min(), vmax=im.max(), cmap='gray'); plt.xticks([]); plt.yticks([]); ax.set_title(f'Warped Pattern {pattern_proj.mean():.5f}/{pattern_proj.std():.5f}')
    im_std = self.data['std0'].to('cpu').numpy()[0,0]
    ax = plt.subplot(3,3,9); plt.imshow(im_std, cmap='gray'); plt.xticks([]); plt.yticks([]); ax.set_title(f'IR std {im_std.min():.5f}/{im_std.max():.5f}')

    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close(fig)


  def callback_train_post_backward(self, net, errs, output, epoch, batch_idx, masks=[]):
    if batch_idx % 512 == 0:
      out_path = self.exp_out_root / f'train_{epoch:03d}_{batch_idx:04d}.png'
      es, gt, im, ma = self.numpy_in_out(output)
      self.write_img(out_path, es[0,0], gt[0,0], im[0,0], ma[0,0])


  def callback_test_start(self, epoch, set_idx):
    self.metric = co.metric.MultipleMetric(
        co.metric.DistanceMetric(vec_length=1),
        co.metric.OutlierFractionMetric(vec_length=1, thresholds=[0.1, 0.5, 1, 2, 5]) 
      )

  def callback_test_add(self, epoch, set_idx, batch_idx, n_batches, output, masks=[]):
    es, gt, im, ma = self.numpy_in_out(output)

    if batch_idx % 8 == 0:
      out_path = self.exp_out_root / f'test_{epoch:03d}_{batch_idx:04d}.png'
      self.write_img(out_path, es[0,0], gt[0,0], im[0,0], ma[0,0])

    es, gt, im, ma = self.crop_output(es, gt, im, ma)

    es = es.reshape(-1,1)
    gt = gt.reshape(-1,1)
    ma = ma.ravel()
    self.metric.add(es, gt, ma)

  def callback_test_stop(self, epoch, set_idx, loss):
    logging.info(f'{self.metric}')
    for k, v in self.metric.items():
      self.metric_add_test(epoch, set_idx, k, v)

  def crop_output(self, es, gt, im, ma):
    bs = es.shape[0]
    es = np.reshape(es[:,:,self.eval_mask], [bs, 1, self.eval_h, self.eval_w])
    gt = np.reshape(gt[:,:,self.eval_mask], [bs, 1, self.eval_h, self.eval_w])
    im = np.reshape(im[:,:,self.eval_mask], [bs, 1, self.eval_h, self.eval_w])
    ma = np.reshape(ma[:,:,self.eval_mask], [bs, 1, self.eval_h, self.eval_w])
    return es, gt, im, ma



if __name__ == '__main__':
  pass
