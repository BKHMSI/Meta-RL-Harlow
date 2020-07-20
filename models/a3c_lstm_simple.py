import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class A3C_LSTM(nn.Module):

    def __init__(self, config, num_actions):
        super(A3C_LSTM, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(config["mem-units"], num_actions)
        self.critic = nn.Linear(config["mem-units"], 1)
        self.working_memory = nn.LSTM(64+4, config["mem-units"])
        
        # intialize actor and critic weights
        T.nn.init.orthogonal_(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight.data, 1)
        self.critic.bias.data.fill_(0)

    def forward(self, obs, p_input, mem_state=None):

        if mem_state is None:
            mem_state = self.get_init_states()

        feats = self.encoder(obs.unsqueeze(0))

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
