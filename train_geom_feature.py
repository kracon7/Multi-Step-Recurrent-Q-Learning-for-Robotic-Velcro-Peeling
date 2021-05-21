import os
import argparse
import pickle
import random
import datetime as dt
import torch
import torch.nn as nn
import torchvision


class ConvNet(nn.Module):

    def __init__(self, out_dim, depth):
        super(ConvNet, self).__init__()

        self.model_ft = torchvision.models.resnet18(pretrained=True)
        if depth:
            self.model_ft.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model_ft.fc.out_features

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(num_ftrs, num_ftrs)
        self.fc2 = nn.Linear(num_ftrs, out_dim)

    def forward(self, x):
        lin_out = self.model_ft(x)
        lin_out = self.relu(self.fc1(lin_out))
        lin_out = self.relu(self.fc2(lin_out))

        return lin_out

def write_results(path, results):
    f = open(path, 'wb')
    pickle.dump(results, f)
    f.close()

def organize_data(data):
    # seperate data into images and labels
    images = []
    labels = []
    for item in data:
        velcro_param, image = item
        geom_type, origin_offset, euler, radius = velcro_param

        label = origin_offset+euler
        label.append(radius)
        labels.append(label)

        images.append(image)
    return images, labels

# Train either using MDP or POMDP
def main(args):
    # Set random seed for reproducability
    random.seed(1345)
    data = pickle.load(open(args.data_path, 'rb'))
    images, labels = organize_data(data)

    n_batch = 

    for ii in range(epochs):

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Tactile Training')
    parser.add_argument('--model_path', required=True, help='XML model to load')
    parser.add_argument('--data_path', required=True, help='training data to load')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--indim', default=396, type=int, help='observation space size')
    parser.add_argument('--outdim', default=6, type=int, help='action space size')
    parser.add_argument('-b', '--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('-j', '--gamma', default=0.995, type=float, help='future reward decay')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--max_iter', default=200, type=float, help='max number of iterations per epoch')
    parser.add_argument('--output_dir', default='.', help='path where to save')
    parser.add_argument('--save_freq', default=50, type=int, help='frequency to save checkpoints')
    parser.add_argument('--sim', action='store_true', help='whether to run in simulation mode or on a real robot')
    parser.add_argument('--checkpoint_file', default=None, help='checkpoint file to load to resume training')
    parser.add_argument('--normalizer_file', default=None, help='normalizer file to load to resume training')
    parser.add_argument('--memory', default=None, help='path to load memory')
    parser.add_argument('--savefig_path', default=None, help='path to save training results figures')
    
    args = parser.parse_args()
    main(args)
