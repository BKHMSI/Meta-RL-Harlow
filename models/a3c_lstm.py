import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class A3C_LSTM(nn.Module):

    def __init__(self, config, num_actions):
        super(A3C_LSTM, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(8, 8), stride=(4, 4)),  # output: (16, 20, 20)
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)), # output: (32, 9, 9)
            nn.Flatten(),
            nn.Linear(32*9*9, 256),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(config["mem-units"], num_actions)
        self.critic = nn.Linear(config["mem-units"], 1)
        self.working_memory = nn.LSTM(256+4, config["mem-units"])
        
        # intialize actor and critic weights
        T.nn.init.orthogonal_(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight.data, 1)
        self.critic.bias.data.fill_(0)

    def forward(self, obs, p_input, mem_state=None):

        if mem_state is None:
            mem_state = self.get_init_states()

        feats = self.encoder(obs.unsqueeze(0))

        # import pdb; pdb.set_trace()
        mem_input = T.cat((feats, *p_input), dim=-1)
        if len(mem_input.size()) == 2:
            mem_input = mem_input.unsqueeze(0)

        h_t, mem_state = self.working_memory(mem_input, mem_state)

        action_dist = F.softmax(self.actor(h_t), dim=-1)
        value_estimate = self.critic(h_t)

        return action_dist, value_estimate, mem_state

    def get_init_states(self, device='cpu'):
        h0 = T.zeros(1, 1, self.working_memory.hidden_size).float().to(device)
        c0 = T.zeros(1, 1, self.working_memory.hidden_size).float().to(device)
        return (h0, c0)

class A3C_StackedLSTM(nn.Module):

    def __init__(self, config, num_actions):
        super(A3C_StackedLSTM, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(8, 8), stride=(4, 4)),  # output: (16, 20, 20)
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)), # output: (32, 9, 9)
            nn.Flatten(),
            nn.Linear(32*9*9, 256),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(64, num_actions)
        self.critic = nn.Linear(64, 1)
        self.lstm_1 = nn.LSTM(256+1, config["mem-units"])
        self.lstm_2 = nn.LSTM(256+config["mem-units"]+3, 64)
        
        # intialize actor and critic weights
        T.nn.init.orthogonal_(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight.data, 1)
        self.critic.bias.data.fill_(0)

    def forward(self, obs, p_input, state_1=None, state_2=None):

        p_action, p_reward = p_input

        if state_1 is None:
            state_1 = self.get_init_states(layer=1)

        if state_2 is None:
            state_2 = self.get_init_states(layer=2)

        feats = self.encoder(obs.unsqueeze(0))

        input_1 = T.cat((feats, p_reward), dim=-1)
        if len(input_1.size()) == 2:
            input_1 = input_1.unsqueeze(0)

        output_1, state_1 = self.lstm_1(input_1, state_1)

        input_2 = T.cat((feats, output_1.squeeze(0), p_action), dim=-1)
        if len(input_2.size()) == 2:
            input_2 = input_2.unsqueeze(0)

        output_2, state_2 = self.lstm_2(input_2, state_2)
        
        action_dist = F.softmax(self.actor(output_2), dim=-1)
        value_estimate = self.critic(output_2)

        return action_dist, value_estimate, state_1, state_2

    def get_init_states(self, layer, device='cpu'):
        hsize = self.lstm_1.hidden_size if layer == 1 else self.lstm_2.hidden_size
        h0 = T.zeros(1, 1, hsize).float().to(device)
        c0 = T.zeros(1, 1, hsize).float().to(device)
        return (h0, c0)