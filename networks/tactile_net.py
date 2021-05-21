import torch
import torch.nn as nn

class TactileNet(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(TactileNet, self).__init__()
        self.input_size = in_dim
        self.output_size = out_dim
        self.n_filters = 512

        self.lstm_layer = nn.LSTM(input_size=in_dim, hidden_size=in_dim, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(in_dim, self.n_filters)
        self.linear2 = nn.Linear(self.n_filters, self.n_filters // 4)
        self.linear3 = nn.Linear(self.n_filters // 4, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, hidden_state, cell_state):
        lstm_out, _ = self.lstm_layer(x, (hidden_state, cell_state))

        # Pass the data through all the linear layers
        lin_out = self.relu(self.linear1(lstm_out[:, -1, :]))
        lin_out = self.relu(self.linear2(lin_out))
        lin_out = self.relu(self.linear3(lin_out))

        return lin_out

    def init_hidden_states(self, device):
        h = torch.zeros(1, 1, self.input_size).float().to(device)
        c = torch.zeros(1, 1, self.input_size).float().to(device)

        return h, c
