import os
import torch
from model import exp_synph
from model import exp_synphge
from model import networks
from co.args import parse_args


# parse args
args = parse_args()

# loss types
if args.loss=='ph':
  worker = exp_synph.Worker(args)
elif args.loss=='phge':
  worker = exp_synphge.Worker(args)

# concatenation of original image and lcn image
channels_in=2 

# set up network
net = networks.DispEdgeDecoders(channels_in=channels_in, max_disp=args.max_disp, imsizes=worker.imsizes, output_ms=worker.ms)

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

# start the work
worker.do(net, optimizer)

