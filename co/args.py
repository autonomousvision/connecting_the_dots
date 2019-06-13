import argparse
import os
from .utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--output_dir',
                        help='Output directory',
                        default='./output', type=str)
    parser.add_argument('--loss',
                        help='Train with \'ph\' for the first stage without geometric loss, \
                              train with \'phge\' for the second stage with geometric loss',
                        default='ph', choices=['ph','phge'], type=str)
    parser.add_argument('--data_type',
                        default='syn', choices=['syn'], type=str)
    #
    parser.add_argument('--cmd', 
                        help='Start training or test', 
                        default='resume', choices=['retrain', 'resume', 'retest', 'test_init'], type=str)
    parser.add_argument('--epoch', 
                        help='If larger than -1, retest on the specified epoch',
                        default=-1, type=int)
    parser.add_argument('--epochs',
                        help='Training epochs',
                        default=100, type=int)

    # 
    parser.add_argument('--ms',
                        help='If true, use multiscale loss',
                        default=True, type=str2bool)
    parser.add_argument('--pattern_path',
                        help='Path of the pattern image',
                        default='./data/kinect_patttern.png', type=str)
    #
    parser.add_argument('--dp_weight',
                        help='Weight of the disparity loss',
                        default=0.02, type=float)
    parser.add_argument('--ge_weight',
                        help='Weight of the geometric loss',
                        default=0.1, type=float)
    #
    parser.add_argument('--lcn_radius',
                        help='Radius of the window for LCN pre-processing',
                        default=5, type=int)
    parser.add_argument('--max_disp',
                        help='Maximum disparity',
                        default=128, type=int)
    #
    parser.add_argument('--track_length',
                        help='Track length for geometric loss',
                        default=2, type=int)
    #
    parser.add_argument('--blend_im',
                        help='Parameter for adding texture',
                        default=0.6, type=float)
    
    args = parser.parse_args()

    args.exp_name = get_exp_name(args)

    return args


def get_exp_name(args):
    name = f"exp_{args.data_type}"
    return name



