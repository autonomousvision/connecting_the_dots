import numpy as np
import torch
import random
import logging
import datetime
from pathlib import Path
import argparse
import subprocess
import socket
import sys
import os
import gc
import json
import matplotlib.pyplot as plt
import time
from collections import OrderedDict


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

class Worker(object):
  def __init__(self, out_root, experiment_name, epochs=10, seed=42, train_batch_size=8, test_batch_size=16, num_workers=16, save_frequency=1, train_device='cuda:0', test_device='cuda:0', max_train_iter=-1):
    self.out_root = Path(out_root)
    self.experiment_name = experiment_name
    self.epochs = epochs
    self.seed = seed
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size
    self.num_workers = num_workers
    self.save_frequency = save_frequency
    self.train_device = train_device
    self.test_device = test_device
    self.max_train_iter = max_train_iter

    self.errs_list=[]

    self.setup_experiment()

  def setup_experiment(self):
    self.exp_out_root = self.out_root / self.experiment_name
    self.exp_out_root.mkdir(parents=True, exist_ok=True)

    if logging.root: del logging.root.handlers[:]
    logging.basicConfig(
      level=logging.INFO,
      handlers=[
        logging.FileHandler( str(self.exp_out_root / 'train.log') ),
        logging.StreamHandler()
      ],
      format='%(relativeCreated)d:%(levelname)s:%(process)d-%(processName)s: %(message)s'
    )

    logging.info('='*80)
    logging.info(f'Start of experiment: {self.experiment_name}')
    logging.info(socket.gethostname())
    self.log_datetime()
    logging.info('='*80)

    self.metric_path = self.exp_out_root / 'metrics.json'
    if self.metric_path.exists():
      with open(str(self.metric_path), 'r') as fp:
        self.metric_data = json.load(fp)
    else:
      self.metric_data = {}

    self.init_seed()

  def metric_add_train(self, epoch, key, val):
    epoch = str(epoch)
    key = str(key)
    if epoch not in self.metric_data:
      self.metric_data[epoch] = {}
    if 'train' not in self.metric_data[epoch]:
      self.metric_data[epoch]['train'] = {}
    self.metric_data[epoch]['train'][key] = val

  def metric_add_test(self, epoch, set_idx, key, val):
    epoch = str(epoch)
    set_idx = str(set_idx)
    key = str(key)
    if epoch not in self.metric_data:
      self.metric_data[epoch] = {}
    if 'test' not in self.metric_data[epoch]:
      self.metric_data[epoch]['test'] = {}
    if set_idx not in self.metric_data[epoch]['test']:
      self.metric_data[epoch]['test'][set_idx] = {}
    self.metric_data[epoch]['test'][set_idx][key] = val

  def metric_save(self):
    with open(str(self.metric_path), 'w') as fp:
      json.dump(self.metric_data, fp, indent=2)

  def init_seed(self, seed=None):
    if seed is not None:
      self.seed = seed
    logging.info(f'Set seed to {self.seed}')
    np.random.seed(self.seed)
    random.seed(self.seed)
    torch.manual_seed(self.seed)
    torch.cuda.manual_seed(self.seed)

  def log_datetime(self):
    logging.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

  def mem_report(self):
    for obj in gc.get_objects():
      if torch.is_tensor(obj):
          print(type(obj), obj.shape)

  def get_net_path(self, epoch, root=None):
    if root is None:
      root = self.exp_out_root
    return root / f'net_{epoch:04d}.params'

  def get_do_parser_cmds(self):
    return ['retrain', 'resume', 'retest', 'test_init']

  def get_do_parser(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd', type=str, default='resume', choices=self.get_do_parser_cmds())
    parser.add_argument('--epoch', type=int, default=-1)
    return parser

  def do_cmd(self, args, net, optimizer, scheduler=None):
    if args.cmd == 'retrain':
      self.train(net, optimizer, resume=False, scheduler=scheduler)
    elif args.cmd == 'resume':
      self.train(net, optimizer, resume=True, scheduler=scheduler)
    elif args.cmd == 'retest':
      self.retest(net, epoch=args.epoch)
    elif args.cmd == 'test_init':
      test_sets = self.get_test_sets()
      self.test(-1, net, test_sets)
    else:
      raise Exception('invalid cmd')

  def do(self, net, optimizer, load_net_optimizer=None, scheduler=None):
    parser = self.get_do_parser()
    args, _ = parser.parse_known_args()

    if load_net_optimizer is not None and args.cmd not in ['schedule']:
      net, optimizer = load_net_optimizer()

    self.do_cmd(args, net, optimizer, scheduler=scheduler)

  def retest(self, net, epoch=-1):
    if epoch < 0:
      epochs = range(self.epochs)
    else:
      epochs = [epoch]

    test_sets = self.get_test_sets()

    for epoch in epochs:
      net_path = self.get_net_path(epoch)
      if net_path.exists():
        state_dict = torch.load(str(net_path))
        net.load_state_dict(state_dict)
        self.test(epoch, net, test_sets)

  def format_err_str(self, errs, div=1):
    err = sum(errs)
    if len(errs) > 1:
      err_str = f'{err/div:0.4f}=' + '+'.join([f'{e/div:0.4f}' for e in errs])
    else:
      err_str = f'{err/div:0.4f}'
    return err_str

  def write_err_img(self):
    err_img_path = self.exp_out_root / 'errs.png'
    fig = plt.figure(figsize=(16,16))
    lines=[]
    for idx,errs in enumerate(self.errs_list):
      line,=plt.plot(range(len(errs)), errs, label=f'error{idx}')
      lines.append(line)
    plt.tight_layout()
    plt.legend(handles=lines)
    plt.savefig(str(err_img_path))
    plt.close(fig)


  def callback_train_new_epoch(self, epoch, net, optimizer):
    pass

  def train(self, net, optimizer, resume=False, scheduler=None):
    logging.info('='*80)
    logging.info('Start training')
    self.log_datetime()
    logging.info('='*80)

    train_set = self.get_train_set()
    test_sets = self.get_test_sets()

    net = net.to(self.train_device)

    epoch = 0
    min_err = {ts.name: 1e9 for ts in test_sets}

    state_path = self.exp_out_root / 'state.dict'
    if resume and state_path.exists():
      logging.info('='*80)
      logging.info(f'Loading state from {state_path}')
      logging.info('='*80)
      state = torch.load(str(state_path))
      epoch = state['epoch'] + 1
      if 'min_err' in state:
        min_err = state['min_err']

      curr_state = net.state_dict()
      curr_state.update(state['state_dict'])
      net.load_state_dict(curr_state)


      try:
        optimizer.load_state_dict(state['optimizer'])
      except:
        logging.info('Warning: cannot load optimizer from state_dict')
        pass
      if 'cpu_rng_state' in state:
        torch.set_rng_state(state['cpu_rng_state'])
      if 'gpu_rng_state' in state:
        torch.cuda.set_rng_state(state['gpu_rng_state'])

    for epoch in range(epoch, self.epochs):
      self.callback_train_new_epoch(epoch, net, optimizer)

      # train epoch
      self.train_epoch(epoch, net, optimizer, train_set)

      # test epoch
      errs = self.test(epoch, net, test_sets)

      if (epoch + 1) % self.save_frequency == 0:
        net = net.to(self.train_device)

        # store state
        state_dict = {
            'epoch': epoch,
            'min_err': min_err,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'cpu_rng_state': torch.get_rng_state(),
            'gpu_rng_state': torch.cuda.get_rng_state(),
        }
        logging.info(f'save state to {state_path}')
        state_path = self.exp_out_root / 'state.dict'
        torch.save(state_dict, str(state_path))

        for test_set_name in errs:
          err = sum(errs[test_set_name])
          if err < min_err[test_set_name]:
            min_err[test_set_name] = err
            state_path = self.exp_out_root / f'state_set{test_set_name}_best.dict'
            logging.info(f'save state to {state_path}')
            torch.save(state_dict, str(state_path))

        # store network
        net_path = self.get_net_path(epoch)
        logging.info(f'save network to {net_path}')
        torch.save(net.state_dict(), str(net_path))

      if scheduler is not None:
        scheduler.step()

    logging.info('='*80)
    logging.info('Finished training')
    self.log_datetime()
    logging.info('='*80)

  def get_train_set(self):
    # returns train_set
    raise NotImplementedError()

  def get_test_sets(self):
    # returns test_sets
    raise NotImplementedError()

  def copy_data(self, data, device, requires_grad, train):
    raise NotImplementedError()

  def net_forward(self, net, train):
    raise NotImplementedError()

  def loss_forward(self, output, train):
    raise NotImplementedError()

  def callback_train_post_backward(self, net, errs, output, epoch, batch_idx, masks):
  #   err = False
  #   for name, param in net.named_parameters():
  #     if not torch.isfinite(param.grad).all():
  #       print(name)
  #       err = True
  #   if err:
  #     import ipdb; ipdb.set_trace()
    pass

  def callback_train_start(self, epoch):
    pass

  def callback_train_stop(self, epoch, loss):
    pass

  def train_epoch(self, epoch, net, optimizer, dset):
    self.callback_train_start(epoch)
    stopwatch = StopWatch()

    logging.info('='*80)
    logging.info('Train epoch %d' % epoch)

    dset.current_epoch = epoch
    train_loader = torch.utils.data.DataLoader(dset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=False)

    net = net.to(self.train_device)
    net.train()

    mean_loss = None

    n_batches = self.max_train_iter if self.max_train_iter > 0 else len(train_loader)
    bar = ETA(length=n_batches)

    stopwatch.start('total')
    stopwatch.start('data')
    for batch_idx, data in enumerate(train_loader):
      if self.max_train_iter > 0 and batch_idx > self.max_train_iter: break
      self.copy_data(data, device=self.train_device, requires_grad=True, train=True)
      stopwatch.stop('data')

      optimizer.zero_grad()

      stopwatch.start('forward')
      output = self.net_forward(net, train=True)
      if 'cuda' in self.train_device: torch.cuda.synchronize()
      stopwatch.stop('forward')

      stopwatch.start('loss')
      errs = self.loss_forward(output, train=True)
      if isinstance(errs, dict):
          masks = errs['masks']
          errs = errs['errs']
      else:
          masks = []
      if not isinstance(errs, list) and not isinstance(errs, tuple):
        errs = [errs]
      err = sum(errs)
      if 'cuda' in self.train_device: torch.cuda.synchronize()
      stopwatch.stop('loss')

      stopwatch.start('backward')
      err.backward()
      self.callback_train_post_backward(net, errs, output, epoch, batch_idx, masks)
      if 'cuda' in self.train_device: torch.cuda.synchronize()
      stopwatch.stop('backward')

      stopwatch.start('optimizer')
      optimizer.step()
      if 'cuda' in self.train_device: torch.cuda.synchronize()
      stopwatch.stop('optimizer')

      bar.update(batch_idx)
      if (epoch <= 1 and batch_idx < 128) or batch_idx % 16 == 0:
        err_str = self.format_err_str(errs)
        logging.info(f'train e{epoch}: {batch_idx+1}/{len(train_loader)}: loss={err_str} | {bar.get_elapsed_time_str()} / {bar.get_remaining_time_str()}')
        #self.write_err_img()


      if mean_loss is None:
        mean_loss = [0 for e in errs]
      for erridx, err in enumerate(errs):
        mean_loss[erridx] += err.item()

      stopwatch.start('data')
    stopwatch.stop('total')
    logging.info('timings: %s' % stopwatch)

    mean_loss = [l / len(train_loader) for l in mean_loss]
    self.callback_train_stop(epoch, mean_loss)
    self.metric_add_train(epoch, 'loss', mean_loss)

    # save metrics
    self.metric_save()

    err_str = self.format_err_str(mean_loss)
    logging.info(f'avg train_loss={err_str}')
    return mean_loss

  def callback_test_start(self, epoch, set_idx):
    pass

  def callback_test_add(self, epoch, set_idx, batch_idx, n_batches, output, masks):
    pass

  def callback_test_stop(self, epoch, set_idx, loss):
    pass

  def test(self, epoch, net, test_sets):
    errs = {}
    for test_set_idx, test_set in enumerate(test_sets):
      if (epoch + 1) % test_set.test_frequency == 0:
        logging.info('='*80)
        logging.info(f'testing set {test_set.name}')
        err = self.test_epoch(epoch, test_set_idx, net, test_set.dset)
        errs[test_set.name] = err
    return errs

  def test_epoch(self, epoch, set_idx, net, dset):
    logging.info('-'*80)
    logging.info('Test epoch %d' % epoch)
    dset.current_epoch = epoch
    test_loader = torch.utils.data.DataLoader(dset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False, pin_memory=False)

    net = net.to(self.test_device)
    net.eval()

    with torch.no_grad():
      mean_loss = None

      self.callback_test_start(epoch, set_idx)

      bar = ETA(length=len(test_loader))
      stopwatch = StopWatch()
      stopwatch.start('total')
      stopwatch.start('data')
      for batch_idx, data in enumerate(test_loader):
        # if batch_idx == 10: break
        self.copy_data(data, device=self.test_device, requires_grad=False, train=False)
        stopwatch.stop('data')

        stopwatch.start('forward')
        output = self.net_forward(net, train=False)
        if 'cuda' in self.test_device: torch.cuda.synchronize()
        stopwatch.stop('forward')

        stopwatch.start('loss')
        errs = self.loss_forward(output, train=False)
        if isinstance(errs, dict):
            masks = errs['masks']
            errs = errs['errs']
        else:
            masks = []
        if not isinstance(errs, list) and not isinstance(errs, tuple):
          errs = [errs]

        bar.update(batch_idx)
        if batch_idx % 25 == 0:
          err_str = self.format_err_str(errs)
          logging.info(f'test e{epoch}: {batch_idx+1}/{len(test_loader)}: loss={err_str} | {bar.get_elapsed_time_str()} / {bar.get_remaining_time_str()}')

        if mean_loss is None:
          mean_loss = [0 for e in errs]
        for erridx, err in enumerate(errs):
          mean_loss[erridx] += err.item()
        stopwatch.stop('loss')

        self.callback_test_add(epoch, set_idx, batch_idx, len(test_loader), output, masks)

        stopwatch.start('data')
      stopwatch.stop('total')
      logging.info('timings: %s' % stopwatch)

      mean_loss = [l / len(test_loader) for l in mean_loss]
      self.callback_test_stop(epoch, set_idx, mean_loss)
      self.metric_add_test(epoch, set_idx, 'loss', mean_loss)

      # save metrics
      self.metric_save()

      err_str = self.format_err_str(mean_loss)
      logging.info(f'test epoch {epoch}: avg test_loss={err_str}')
      return mean_loss
