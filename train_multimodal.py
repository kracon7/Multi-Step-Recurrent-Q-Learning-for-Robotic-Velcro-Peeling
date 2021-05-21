import os
import argparse
import pickle
import random
import datetime as dt

from networks.mdp import MDP
from networks.multimodal_pomdp import POMDP

# import cProfile

def write_results(path, results):
    f = open(path, 'wb')
    pickle.dump(results, f)
    f.close()


# Train either using MDP or POMDP
def main(args):
    # Set random seed for reproducability
    random.seed(1345)

    if args.mdp:
        mdp = MDP(args)
        result = mdp.train_MDP()
        now = dt.datetime.now()
        write_results(os.path.join(args.output_dir, 'results_mdp_{}_{}_{}.pkl'.format(now.month, now.day, now.hour)), result)
    elif args.pomdp:
        pomdp = POMDP(args)
        result = pomdp.train_POMDP()
        now = dt.datetime.now()
        write_results(os.path.join(args.output_dir, 'results_pomdp_{}_{}_{}.pkl'.format(now.month, now.day, now.hour)), result)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Tactile Training')
   
    method = parser.add_mutually_exclusive_group(required=True)
    method.add_argument('--mdp', action='store_true', help='train model using MDP')
    method.add_argument('--pomdp', action='store_true', help='train model using POMDP')

    parser.add_argument('--model_path', default='models/flat_velcro.xml', help='XML model to load')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--ftdim', default=100, type=int, help='image feature size, or size of conv_net output')
    parser.add_argument('--indim', default=496, type=int, help='observation space size')
    parser.add_argument('--outdim', default=6, type=int, help='action space size')
    parser.add_argument('-b', '--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('-t', '--time_step', default=4, type=int, help='time steps used for pomdp')
    parser.add_argument('-j', '--gamma', default=0.995, type=float, help='future reward decay')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--num-steps', type=int, default=20, metavar='NS', help='number of forward steps in A3C (default: 20)')
    parser.add_argument('--lr', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--tau', default=0.01, type=float, help='copy ratio of double dqn')
    parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
    parser.add_argument('--max_iter', default=200, type=float, help='max number of iterations per epoch')
    parser.add_argument('--grip_force', default=300, type=float, help='gripping force')
    parser.add_argument('--output_dir', default='.', help='path where to save')
    parser.add_argument('--save_freq', default=50, type=int, help='frequency to save checkpoints')
    parser.add_argument('--quiet', action='store_true', help='wether to print episodes or not')
    parser.add_argument('--render', action='store_true', help='turn on rendering')
    parser.add_argument('--checkpoint_file', default=None, help='checkpoint file to load to resume training')
    parser.add_argument('--normalizer_file', default=None, help='normalizer file to load to resume training')
    parser.add_argument('--weight_conv', default=None, help='weight file to load fot conv_net')
    parser.add_argument('--memory', default=None, help='path to load memory')
    parser.add_argument('--savefig_path', default=None, help='path to save training results figures')
    parser.add_argument('--img_w', default=200, type=int, help='observation image width')
    parser.add_argument('--img_h', default=200, type=int, help='observation image height')
    parser.add_argument('--depth', default=True, type=bool, help='use depth from rendering as input')
    parser.add_argument('--hap_sample', default=30, type=int, help='number of haptics samples feedback in each action excution')

    args = parser.parse_args()

    model = 'MDP'
    if args.pomdp:
        model = 'POMDP'
    print("Running model {}".format(model))
    # cProfile.run('main(args)')
    main(args)
