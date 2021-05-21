import os
import argparse
import pickle
import random
import datetime as dt

from networks.a2c_pomdp import A2C_POMDP


def write_results(path, results):
    f = open(path, 'wb')
    pickle.dump(results, f)
    f.close()


# Train either using MDP or POMDP
def main(args):
    # Set random seed for reproducability
    random.seed(1345)
    pomdp = A2C_POMDP(args)
    result = pomdp.train_POMDP()
    now = dt.datetime.now()
    write_results(os.path.join(args.output_dir, 'results_pomdp_{}_{}_{}.pkl'.format(now.month, now.day, now.hour)), result)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Tactile Training')
    
    ablation = parser.add_mutually_exclusive_group(required=True)
    ablation.add_argument('--none', action='store_true', help='include position, shear and tactile in observation')
    ablation.add_argument('--position', action='store_true', help='remove position from observation')
    ablation.add_argument('--shear', action='store_true', help='remove shear from observation')
    ablation.add_argument('--force', action='store_true', help='remove tactile from observation')

    parser.add_argument('--model_path', required=True, help='XML model to load')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--indim', default=24, type=int, help='observation space size')
    parser.add_argument('--outdim', default=6, type=int, help='action space size')
    parser.add_argument('-b', '--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('-t', '--time_step', default=4, type=int, help='time steps used for pomdp')
    parser.add_argument('-j', '--gamma', default=0.995, type=float, help='future reward decay')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--num-steps', type=int, default=20, metavar='NS', help='number of forward steps in A3C (default: 20)')
    parser.add_argument('--lr', default=2e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--tau', default=0.01, type=float, help='copy ratio of double dqn')
    parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
    parser.add_argument('--max_iter', default=300, type=int, help='max number of iterations per epoch')
    parser.add_argument('--grip_force', default=300, type=float, help='gripping force')
    parser.add_argument('--output_dir', default='.', help='path where to save')
    parser.add_argument('--save_freq', default=50, type=int, help='frequency to save checkpoints')
    parser.add_argument('--quiet', action='store_true', help='wether to print episodes or not')
    parser.add_argument('--render', action='store_true', help='turn on rendering')
    parser.add_argument('--checkpoint_file', default=None, help='checkpoint file to load to resume training')
    parser.add_argument('--normalizer_file', default=None, help='normalizer file to load to resume training')
    parser.add_argument('--savefig_path', default=None, help='path to save training results figures')
    parser.add_argument('--hap_sample', default=30, type=int, help='number of haptics samples feedback in each action excution')

    args = parser.parse_args()
    ablation = 'None'
    if args.position:
        ablation = 'position'
    if args.shear:
        ablation = 'shear'
    if args.force:
        ablation = 'force'
    model = 'POMDP'
    print("Running model {} with ablation {}".format(model, ablation))
    main(args)
