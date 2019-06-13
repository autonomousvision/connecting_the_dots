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
    self.ge_weight = args.ge_weight
    self.track_length = args.track_length
    self.data_type = args.data_type 
    assert(self.track_length>1)

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
    train_set = dataset.TrackSynDataset(self.settings_path, self.train_paths, train=True, data_aug=True, track_length=self.track_length)
    return train_set

  def get_test_sets(self):
    test_sets = torchext.TestSets()
    test_set = dataset.TrackSynDataset(self.settings_path, self.test_paths, train=False, data_aug=True, track_length=1)
    test_sets.append('simple', test_set, test_frequency=1)

    self.ph_losses = []
    self.ge_losses = []
    self.d2ds = []

    self.lcn_in = self.lcn_in.to('cuda')
    for sidx in range(len(test_set.imsizes)):
      imsize = test_set.imsizes[sidx]
      pat = test_set.patterns[sidx]
      pat = pat.mean(axis=2)
      pat = torch.from_numpy(pat[None][None].astype(np.float32)).to('cuda')
      pat,_ = self.lcn_in(pat)
      pat = torch.cat([pat for idx in range(3)], dim=1)
      ph_loss = networks.RectifiedPatternSimilarityLoss(imsize[0],imsize[1], pattern=pat)

      K = test_set.getK(sidx)
      Ki = np.linalg.inv(K)
      K = torch.from_numpy(K)
      Ki = torch.from_numpy(Ki)
      ge_loss = networks.ProjectionDepthSimilarityLoss(K, Ki, imsize[0], imsize[1], clamp=0.1)

      self.ph_losses.append( ph_loss )
      self.ge_losses.append( ge_loss )
      
      d2d = networks.DispToDepth(float(test_set.focal_lengths[sidx]), float(test_set.baseline))
      self.d2ds.append( d2d )

    return test_sets

  def copy_data(self, data, device, requires_grad, train):
    self.data = {}

    self.lcn_in = self.lcn_in.to(device)
    for key, val in data.items():
      # from 
      # batch_size x track_length x ...
      # to
      # track_length x batch_size x ...
      if len(val.shape)>2:
        if train:
          val = val.transpose(0,1)
        else:
          val = val.unsqueeze(0)
      grad = 'im' in key and requires_grad
      self.data[key] = val.to(device).requires_grad_(requires_grad=grad)
      if 'im' in key and 'blend' not in key:
        im = self.data[key]
        tl = im.shape[0]
        bs = im.shape[1]
        im_lcn,im_std = self.lcn_in(im.contiguous().view(-1, *im.shape[2:]))
        key_std = key.replace('im','std')
        self.data[key_std] = im_std.view(tl, bs, *im.shape[2:]).to(device)
        im_cat = torch.cat((im_lcn.view(tl, bs, *im.shape[2:]), im), dim=2)
        self.data[key] = im_cat

  def net_forward(self, net, train):
    im0 = self.data['im0']
    tl = im0.shape[0]
    bs = im0.shape[1]
    im0 = im0.view(-1, *im0.shape[2:])
    out, edge = net(im0)
    if not(isinstance(out, tuple) or isinstance(out, list)):
        out = out.view(tl, bs, *out.shape[1:])
        edge = edge.view(tl, bs, *out.shape[1:])
    else:
        out = [o.view(tl, bs, *o.shape[1:]) for o in out]
        edge = [e.view(tl, bs, *e.shape[1:]) for e in edge]
    return out, edge

  def loss_forward(self, out, train):
    out, edge = out
    if not(isinstance(out, tuple) or isinstance(out, list)):
      out = [out]
    vals = []
    diffs = []

    # apply photometric loss
    for s,l,o in zip(itertools.count(), self.ph_losses, out):
      im = self.data[f'im{s}']
      im = im.view(-1, *im.shape[2:])
      o = o.view(-1, *o.shape[2:])
      std = self.data[f'std{s}']
      std = std.view(-1, *std.shape[2:])
      val, pattern_proj = l(o, im[:,0:1,...], std)
      vals.append(val)
      if s == 0: 
        self.pattern_proj = pattern_proj.detach()

    # apply disparity loss
    # 1-edge as ground truth edge if inversed
    edge0 = 1-torch.sigmoid(edge[0])
    edge0 = edge0.view(-1, *edge0.shape[2:])
    out0 = out[0].view(-1, *out[0].shape[2:])
    val = self.disparity_loss(out0, edge0)
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
        e = e[:,mask,:]
        grad = grad[:,mask,:]
        e = e.view(-1, *e.shape[2:])
        grad = grad.view(-1, *grad.shape[2:])
        val = self.edge_loss(e, grad)
      else:
        val = torch.zeros_like(vals[0]) 
      vals.append(val)

    if train is False:
      return vals

    # apply geometric loss
    R = self.data['R']
    t = self.data['t']
    ge_num = self.track_length * (self.track_length-1) / 2
    for sidx in range(len(out)):
      d2d = self.d2ds[sidx]
      depth = d2d(out[sidx])
      ge_loss = self.ge_losses[sidx]
      imsize = self.imsizes[sidx]
      for tidx0 in range(depth.shape[0]):
        for tidx1 in range(tidx0+1, depth.shape[0]):
          depth0 = depth[tidx0]
          R0 = R[tidx0]
          t0 = t[tidx0]
          depth1 = depth[tidx1]
          R1 = R[tidx1]
          t1 = t[tidx1]

          val = ge_loss(depth0, depth1, R0, t0, R1, t1)
          vals.append(val * self.ge_weight / ge_num)

    return vals

  def numpy_in_out(self, output):
    output, edge = output
    if not(isinstance(output, tuple) or isinstance(output, list)):
      output = [output]
    es = output[0].detach().to('cpu').numpy()
    gt = self.data['disp0'].to('cpu').numpy().astype(np.float32)
    im = self.data['im0'][:,:,0:1,...].detach().to('cpu').numpy()
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
    im_orig = self.data['im0'].detach().to('cpu').numpy()[0,0,0]
    pattern_diff = np.abs(im_orig - pattern_proj)

    fig = plt.figure(figsize=(16,16))
    es0 = co.cmap.color_depth_map(es[0], scale=vmax)
    gt0 = co.cmap.color_depth_map(gt[0], scale=vmax)
    diff0 = co.cmap.color_error_image(diff[0], BGR=True)

    # plot disparities, ground truth disparity is shown only for reference
    ax = plt.subplot(3,3,1); plt.imshow(es0[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'F0 Disparity Est. {es0.min():.4f}/{es0.max():.4f}')
    ax = plt.subplot(3,3,2); plt.imshow(gt0[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'F0 Disparity GT {np.nanmin(gt0):.4f}/{np.nanmax(gt0):.4f}')
    ax = plt.subplot(3,3,3); plt.imshow(diff0[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'F0 Disparity Err. {diff0.mean():.5f}')

    # plot disparities of the second frame in the track if exists
    if es.shape[0]>=2:
      es1 = co.cmap.color_depth_map(es[1], scale=vmax)
      gt1 = co.cmap.color_depth_map(gt[1], scale=vmax)
      diff1 = co.cmap.color_error_image(diff[1], BGR=True)
      ax = plt.subplot(3,3,4); plt.imshow(es1[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'F1 Disparity Est. {es1.min():.4f}/{es1.max():.4f}')
      ax = plt.subplot(3,3,5); plt.imshow(gt1[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'F1 Disparity GT {np.nanmin(gt1):.4f}/{np.nanmax(gt1):.4f}')
      ax = plt.subplot(3,3,6); plt.imshow(diff1[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'F1 Disparity Err. {diff1.mean():.5f}')

    # plot normalized IR inputs
    ax = plt.subplot(3,3,7); plt.imshow(im[0], vmin=im.min(), vmax=im.max(), cmap='gray'); plt.xticks([]); plt.yticks([]); ax.set_title(f'F0 IR input {im[0].mean():.5f}/{im[0].std():.5f}')
    if es.shape[0]>=2:
      ax = plt.subplot(3,3,8); plt.imshow(im[1], vmin=im.min(), vmax=im.max(), cmap='gray'); plt.xticks([]); plt.yticks([]); ax.set_title(f'F1 IR input {im[1].mean():.5f}/{im[1].std():.5f}')
    
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close(fig)

  def callback_train_post_backward(self, net, errs, output, epoch, batch_idx, masks):
    if batch_idx % 512 == 0:
      out_path = self.exp_out_root / f'train_{epoch:03d}_{batch_idx:04d}.png'
      es, gt, im, ma = self.numpy_in_out(output)
      masks = [ m.detach().to('cpu').numpy() for m in masks ]
      self.write_img(out_path, es[:,0,0], gt[:,0,0], im[:,0,0], ma[:,0,0])

  def callback_test_start(self, epoch, set_idx):
    self.metric = co.metric.MultipleMetric(
        co.metric.DistanceMetric(vec_length=1),
        co.metric.OutlierFractionMetric(vec_length=1, thresholds=[0.1, 0.5, 1, 2, 5]) 
      )

  def callback_test_add(self, epoch, set_idx, batch_idx, n_batches, output, masks):
    es, gt, im, ma = self.numpy_in_out(output)

    if batch_idx % 8 == 0:
      out_path = self.exp_out_root / f'test_{epoch:03d}_{batch_idx:04d}.png'
      self.write_img(out_path, es[:,0,0], gt[:,0,0], im[:,0,0], ma[:,0,0])

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
    tl = es.shape[0]
    bs = es.shape[1]
    es = np.reshape(es[...,self.eval_mask], [tl*bs, 1, self.eval_h, self.eval_w])
    gt = np.reshape(gt[...,self.eval_mask], [tl*bs, 1, self.eval_h, self.eval_w])
    im = np.reshape(im[...,self.eval_mask], [tl*bs, 1, self.eval_h, self.eval_w])
    ma = np.reshape(ma[...,self.eval_mask], [tl*bs, 1, self.eval_h, self.eval_w])
    return es, gt, im, ma

if __name__ == '__main__':
  pass
