import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

class ActorCritic(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ActorCritic, self).__init__()
        self.input_size = in_dim
        self.output_size = out_dim
        self.n_filters = 1024
        self.lstm_size = 256

        self.linear1 = nn.Linear(in_dim, self.n_filters)
        self.linear2 = nn.Linear(self.n_filters, self.lstm_size)
        self.lstm_layer = nn.LSTM(input_size=self.lstm_size, hidden_size=self.lstm_size, num_layers=1)
        self.relu = nn.ReLU()
        self.critic_1 = nn.Linear(self.lstm_size, self.lstm_size)
        self.critic_2 = nn.Linear(self.lstm_size, 1)
        self.actor_1 = nn.Linear(self.lstm_size, self.lstm_size)
        self.actor_2 = nn.Linear(self.lstm_size, out_dim)

    def forward(self, x, hidden_state, cell_state):
        # # Reshape the tensor to have [batch_size * time_step, 1, input, 1] size
        # x = x.view(batch_size * time_step, self.input_size)

        # Pass the data through all the linear layers
        lin_out = self.relu(self.linear1(x))
        lin_out = self.relu(self.linear2(lin_out))

        # Reshape the tensor into [batch_size, time_step, 128]
        lin_out = lin_out.view(1, 1, self.lstm_size)
        hidden_state = hidden_state.view(1, 1, self.lstm_size)
        cell_state = cell_state.view(1, 1, self.lstm_size)

        # Compute the LSTM output
        out, (h_n, c_n) = self.lstm_layer(lin_out, (hidden_state, cell_state))
        out = out[0,0]


        value = self.critic_2(self.relu(self.critic_1(out)))
        policy_dist = F.softmax(self.actor_2(self.relu(self.actor_1(out))), dim=0)

        return value, policy_dist, (h_n, c_n)

    def init_hidden_states(self, device):
        h = torch.zeros(1, 1, self.lstm_size).float().to(device)
        c = torch.zeros(1, 1, self.lstm_size).float().to(device)

        return h, c

