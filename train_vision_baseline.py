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
from networks.conv_net import ConvNet 

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

        img_name = os.path.join(self.root_dir, 'images',
                                self.data_frame.iloc[idx, 0])
        label_name = os.path.join(self.root_dir, 'labels',
                                self.data_frame.iloc[idx, 1])
        image = np.load(img_name)
        label = np.load(label_name)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).float()}

def my_loss(output, label, epsilon=1e-3):
    # breaking boundary center loss
    fl_center_error = torch.norm(output[:, :3] - label[:,:3], dim=1)
    # loss = torch.sum(fl_center_error)
    loss = fl_center_error.mean()
    # n = fl_center_error.shape[0]
    # for i in range(n):
    #     loss -= torch.dot(output[i, 3:6], label[i,3:6])
    #     loss -= torch.dot(output[i, 6:9], label[i,6:9]) 

    ######################## cosine similarity for the direction vectors ###########################
    # breaking boundary normal loss
    pred_boundary_normals = output[:, 3:6].clone() / torch.norm(output[:, 3:6], p=2, dim=1).unsqueeze(1)
    true_boundary_normals = label[:, 3:6]
    
    normal_similarity = (pred_boundary_normals * true_boundary_normals).sum(1).clamp(-1+epsilon, 1-epsilon).acos()
    loss += normal_similarity.mean()

    # breaking direction loss
    pred_breaking_dir = output[:, 6:9].clone() / torch.norm(output[:, 6:9], p=2, dim=1).unsqueeze(1)
    true_breaking_dir = label[:, 6:9]
    
    breaking_dir_similarity = (pred_breaking_dir * true_breaking_dir).sum(1).clamp(-1+epsilon, 1-epsilon).acos()
    loss += breaking_dir_similarity.mean()
    
    # gripper position loss
    loss += torch.sum(torch.norm(output[:, 9:] - label[:,9:], dim=1))
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

    conv_net = ConvNet(args.outdim, args.depth).to(args.device)
    # optimizer = optim.RMSprop(conv_net.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(conv_net.parameters(), lr=args.lr)
    
    print_variables = {'train_loss': [], 'test_loss': []}
    start_epochs = 1
    if args.checkpoint_file:
        if os.path.exists(args.checkpoint_file):
            checkpoint = torch.load(args.checkpoint_file)
            conv_net.load_state_dict(checkpoint['conv_net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epochs = checkpoint['epoch']
            with open(os.path.join(os.path.dirname(args.checkpoint_file), 'results_convnet.pkl'), 'rb') as file:
                plot_dict = pickle.load(file)
                print_variables['train_loss'] = plot_dict['train_loss']
                print_variables['test_loss'] = plot_dict['test_loss']

   

    for ii in range(start_epochs, start_epochs + args.epochs):
        print('================================================')

        conv_net.train()
        epoch_loss = 0
        num_sample = 0
        for i_batch, sample in enumerate(train_dataloader):
            torch_images = sample['image'].to(args.device)
            torch_labels = sample['label'].to(args.device)
            output = conv_net.forward(torch_images)

            if (output != output).any():
                a = 1

            loss = my_loss(output, torch_labels)
            optimizer.zero_grad()
            loss.backward()
            for param in conv_net.parameters():
                param.grad.data.clamp_(-0.01, 0.01)
            
            optimizer.step()

            with torch.no_grad():
                detached_loss = loss.detach()#.cpu()
                num_sample += args.batch_size
                epoch_loss += detached_loss
                ave_epoch_loss = epoch_loss / num_sample

            if not args.quiet and i_batch % 20 == 0:
                print('Training epoch {}, batch :{}, average loss: {}.'.format(ii, i_batch, ave_epoch_loss))

        print_variables['train_loss'].append(ave_epoch_loss)
            

        conv_net.eval()
        eval_loss = 0
        num_sample = 0
        for i_batch, sample in enumerate(test_dataloader):
            torch_images = sample['image'].to(args.device)
            torch_labels = sample['label'].to(args.device)
            output = conv_net.forward(torch_images)
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
        if ii % args.save_freq == 0:
            save_path = os.path.join(args.output_dir, 'checkpoint_model_' + str(ii) + '.pth')
            torch.save({
                       'epoch': ii,
                       'conv_net': conv_net.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       }, save_path)
            write_results(os.path.join(args.output_dir, 'results_convnet.pkl'), print_variables)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Tactile Test')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--root_dir', default='.', help='path where to load data')
    parser.add_argument('--output_dir', default='.', help='path where to save')
    parser.add_argument('--outdim', default=12, type=int, help='action space size')
    parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
    parser.add_argument('--quiet', action='store_true', help='wether to print episodes or not')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save_freq', default=50, type=int, help='frequency to save checkpoints')
    parser.add_argument('--checkpoint_file', default=None, help='checkpoint file to load to resume training')
    parser.add_argument('--img_w', default=250, type=int, help='observation image width')
    parser.add_argument('--img_h', default=250, type=int, help='observation image height')
    parser.add_argument('--depth', default=True, type=bool, help='use depth from rendering as input')
    
    args = parser.parse_args()
    main(args)
