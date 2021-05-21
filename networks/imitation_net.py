import torch
import torch.nn as nn
import torch.nn.functional as F


class ImNet(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ImNet, self).__init__()
        self.input_size = in_dim
        self.output_size = out_dim
        self.n_filters = 512

        self.linear1 = nn.Linear(in_dim, self.n_filters)
        self.linear2 = nn.Linear(self.n_filters, self.n_filters // 4)
        self.lstm_layer = nn.LSTM(input_size=self.n_filters // 4, hidden_size=self.n_filters // 4, 
                                    num_layers=1, batch_first=True)
        self.bnorm1 = nn.BatchNorm1d(self.n_filters)
        self.bnorm2 = nn.BatchNorm1d(self.n_filters // 4)
        self.relu = nn.ReLU()
        self.adv = nn.Linear(self.n_filters // 4, out_dim)

    def forward(self, x, batch_size, time_step, hidden_state, cell_state):
        # Reshape the tensor to have [batch_size * time_step, 1, input, 1] size
        x = x.view(batch_size * time_step, self.input_size)

        # Pass the data through all the linear layers
        lin_out = self.relu(self.linear1(x))
        lin_out = self.bnorm1(lin_out)
        lin_out = self.relu(self.linear2(lin_out))
        lin_out = self.bnorm2(lin_out)

        # Reshape the tensor into [batch_size, time_step, 128]
        lin_out = lin_out.view(batch_size, time_step, self.n_filters // 4)

        # Compute the LSTM output
        lstm_out = self.lstm_layer(lin_out, (hidden_state, cell_state))
        out = lstm_out[0][:, time_step - 1, :]
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]

        out = self.adv(out)
        return F.softmax(out), (h_n, c_n)

    def init_hidden_states(self, batch_size, device):
        h = torch.zeros(1, batch_size, self.n_filters // 4).float().to(device)
        c = torch.zeros(1, batch_size, self.n_filters // 4).float().to(device)

        return h, c