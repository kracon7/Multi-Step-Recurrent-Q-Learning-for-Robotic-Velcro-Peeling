import os
import argparse
import pickle
import random
import datetime as dt

from networks.multistep_pomdp import POMDP


# Train either using MDP or POMDP
def main(args):
    # Set random seed for reproducability
    random.seed(1345)
    pomdp = POMDP(args)
    result = pomdp.train_POMDP()
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Tactile Training')
    ablation = parser.add_mutually_exclusive_group(required=True)
    ablation.add_argument('--none', action='store_true', help='include position, shear and tactile in observation')
    ablation.add_argument('--position', action='store_true', help='remove position from observation')
    ablation.add_argument('--force', action='store_true', help='remove tactile from observation')

    parser.add_argument('--model_path', required=True, help='XML model to load')
    parser.add_argument('--output_dir', default='.', help='path where to save')
    parser.add_argument('--sim', action='store_true', help='whether to run in simulation mode or on a real robot')
    parser.add_argument('--render', action='store_true', help='turn on rendering')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save_freq', default=50, type=int, help='frequency to save checkpoints')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--indim', default=234, type=int, help='observation space size')
    parser.add_argument('--outdim', default=6, type=int, help='action space size')
    parser.add_argument('--ftdim', default=150, type=int, help='tactile feature size')
    parser.add_argument('-b', '--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('-t', '--time_step', default=4, type=int, help='time steps used for pomdp')
    parser.add_argument('-j', '--gamma', default=0.96, type=float, help='future reward decay')
    parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
    parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
    parser.add_argument('--max_iter', default=200, type=float, help='max number of iterations per epoch')
    parser.add_argument('--grip_force', default=300, type=float, help='gripping force')
    parser.add_argument('--hap_sample', default=30, type=int, help='number of haptics samples feedback in each action excution')
    parser.add_argument('--len_ub', default=15, type=int, help='upper bound of multistep agent takes')
    parser.add_argument('--quiet', action='store_true', help='wether to print episodes or not')

    parser.add_argument('--weight_policy', default=None, help='checkpoint file to load to resume training')
    parser.add_argument('--weight_tactile', default=None, help='normalizer file to load to resume training')
    parser.add_argument('--normalizer_file', default=None, help='normalizer file to load to resume training')
    parser.add_argument('--memory', default=None, help='path to load memory')
    parser.add_argument('--savefig_path', default=None, help='path to save training results figures')

    args = parser.parse_args()
    print("Running model multistep_pomdp")
    main(args)
