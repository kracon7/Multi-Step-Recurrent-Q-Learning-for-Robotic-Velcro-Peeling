import torch
import torch.nn as nn
import torchvision

class ConvNet(nn.Module):

    def __init__(self, out_dim, depth):
        super(ConvNet, self).__init__()

        self.model_ft = torchvision.models.resnet18(pretrained=False)
        if depth:
            self.model_ft.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model_ft.fc.out_features

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(num_ftrs, num_ftrs)
        self.fc2 = nn.Linear(num_ftrs, num_ftrs)
        self.fc3 = nn.Linear(num_ftrs, num_ftrs)
        self.fc4 = nn.Linear(num_ftrs, out_dim)

    def forward(self, x):
        lin_out = self.model_ft(x)
        lin_out = self.relu(self.fc1(lin_out))
        lin_out = self.relu(self.fc2(lin_out))
        lin_out = self.relu(self.fc3(lin_out))
        lin_out = self.fc4(lin_out)

        return lin_out