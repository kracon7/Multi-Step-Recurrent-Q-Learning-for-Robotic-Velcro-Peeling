import sys
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.input_size = in_dim
        self.output_size = out_dim
        n_filters = 1024
        self.model = nn.Sequential(nn.Linear(in_dim, n_filters),
                                  nn.ReLU(),
                                  nn.Linear(n_filters, n_filters // 2),
                                  nn.ReLU(),
                                  nn.Linear(n_filters // 2, n_filters // 4),
                                  nn.ReLU(),
                                  nn.Linear(n_filters // 4, n_filters // 4),
                                  nn.ReLU(),
                                  nn.Linear(n_filters // 4, out_dim))

    def forward(self, x):
        out = self.model(x)
        return out


def write_results(path, results):
    f = open(path, 'wb')
    pickle.dump(results, f)
    f.close()

class VisionInputDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_name = os.path.join(self.root_dir, 'tactile',
                                self.data_frame.iloc[idx, 0])
        label_name = os.path.join(self.root_dir, 'labels',
                                self.data_frame.iloc[idx, 1])
        tactile = pickle.load(open(data_name, 'rb'))
        label = pickle.load(open(label_name, 'rb'))
        sample = {'tactile': tactile.flatten(), 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        tactile, label = sample['tactile'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'tactile': torch.from_numpy(tactile).float(),
                'label': torch.from_numpy(label).float()}

def my_loss(output, label, epsilon=1e-3):

    ######################## cosine similarity for the direction vectors ###########################
    # breaking boundary normal loss
    pred_boundary_normals = output[:, :3].clone() / torch.norm(output[:, :2], p=2, dim=1).unsqueeze(1)
    true_boundary_normals = label[:, :3]
    
    normal_similarity = (pred_boundary_normals * true_boundary_normals).sum(1).clamp(-1+epsilon, 1-epsilon).acos()
    loss = normal_similarity.mean()

    # breaking direction loss
    pred_breaking_dir = output[:, 3:].clone() / torch.norm(output[:, 3:], p=2, dim=1).unsqueeze(1)
    true_breaking_dir = label[:, 3:]
    
    breaking_dir_similarity = (pred_breaking_dir * true_breaking_dir).sum(1).clamp(-1+epsilon, 1-epsilon).acos()
    loss += breaking_dir_similarity.mean()
    
    return loss


def main(args):
    # Create the output directory if it does not exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    csv_path = os.path.join(args.root_dir, 'train.csv')
    train_set = VisionInputDataset(csv_file=csv_path, 
                                    root_dir=args.root_dir,
                                    transform=transforms.Compose([
                                               ToTensor()]))

    csv_path = os.path.join(args.root_dir, 'test.csv')
    test_set = VisionInputDataset(csv_file=csv_path, 
                                        root_dir=args.root_dir,
                                        transform=transforms.Compose([
                                                   ToTensor()]))

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    train_set_size = len(train_set)
    test_set_size = len(test_set)
    print('train set size: ', train_set_size, '  test set size: ', test_set_size)

    net = Net(args.indim, args.outdim).to(args.device)
    # optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    print_variables = {'train_loss': [], 'test_loss': []}
    start_epochs = 1
    if args.checkpoint_file:
        if os.path.exists(args.checkpoint_file):
            checkpoint = torch.load(args.checkpoint_file)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epochs = checkpoint['epoch']
            with open(os.path.join(os.path.dirname(args.checkpoint_file), 'results_convnet.pkl'), 'rb') as file:
                plot_dict = pickle.load(file)
                print_variables['train_loss'] = plot_dict['train_loss']
                print_variables['test_loss'] = plot_dict['test_loss']

   

    for ii in range(start_epochs, start_epochs + args.epochs):
        print('================================================')

        net.train()
        epoch_loss = 0
        num_sample = 0
        for i_batch, sample in enumerate(train_dataloader):
            torch_images = sample['tactile'].to(args.device)
            torch_labels = sample['label'].to(args.device)
            output = net.forward(torch_images)

            if (output != output).any():
                a = 1

            loss = my_loss(output, torch_labels)
            optimizer.zero_grad()
            loss.backward()
            for param in net.parameters():
                param.grad.data.clamp_(-0.5, 0.5)
            
            optimizer.step()

            with torch.no_grad():
                detached_loss = loss.detach()#.cpu()
                num_sample += args.batch_size
                epoch_loss += detached_loss
                ave_epoch_loss = epoch_loss / num_sample

            if not args.quiet and i_batch % 20 == 0:
                print('Training epoch {}, batch :{}, average loss: {}.'.format(ii, i_batch, ave_epoch_loss))

        print_variables['train_loss'].append(ave_epoch_loss)
            

        net.eval()
        eval_loss = 0
        num_sample = 0
        for i_batch, sample in enumerate(test_dataloader):
            torch_images = sample['tactile'].to(args.device)
            torch_labels = sample['label'].to(args.device)
            output = net.forward(torch_images)
            # if torch.isnan(output).max() == 1:
            #     import pdb; pdb.set_trace()
            loss = my_loss(output, torch_labels)
            detached_loss = loss.detach().cpu()
            num_sample += args.batch_size
            eval_loss += detached_loss
            ave_eval_loss = eval_loss / num_sample

            if not args.quiet and i_batch % 20 == 0:
                print('Testing epoch {}, batch :{}, average loss: {}.'.format(ii, i_batch, ave_eval_loss))

        print_variables['test_loss'].append(ave_eval_loss)

        # Save checkpoints every vew iterations
        if (ii+1) % args.save_freq == 0:
            save_path = os.path.join(args.output_dir, 'checkpoint_model_' + str(ii+1) + '.pth')
            torch.save({
                       'epoch': ii+1,
                       'net': net.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       }, save_path)
            write_results(os.path.join(args.output_dir, 'results_convnet.pkl'), print_variables)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Tactile Test')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--root_dir', default='.', help='path where to load data')
    parser.add_argument('--output_dir', default='.', help='path where to save')
    parser.add_argument('--indim', default=3660, type=int, help='action space size')
    parser.add_argument('--outdim', default=6, type=int, help='action space size')
    parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
    parser.add_argument('--quiet', action='store_true', help='wether to print episodes or not')
    parser.add_argument('-b', '--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save_freq', default=50, type=int, help='frequency to save checkpoints')
    parser.add_argument('--checkpoint_file', default=None, help='checkpoint file to load to resume training')
    
    args = parser.parse_args()
    main(args)
