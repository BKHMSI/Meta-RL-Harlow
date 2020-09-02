import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.rgu import RGUnit

CELLS = {
    'lstm': nn.LSTM,
    'gru': nn.GRU,
    'rgu': RGUnit
}

class A3C_LSTM(nn.Module):

    def __init__(self, input_dim, hidden_size, num_actions, cell_type="lstm"):
        super(A3C_LSTM, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        rnn = CELLS[cell_type]
        self.cell_type = cell_type

        self.working_memory = rnn(128+num_actions+1, hidden_size)
        
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        
        # intialize actor and critic weights
        T.nn.init.orthogonal_(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight.data, 1)
        self.critic.bias.data.fill_(0)

    def forward(self, obs, p_input, mem_state=None):

        if mem_state is None:
            mem_state = self.get_init_states()

        feats = self.encoder(obs)
        mem_input = T.cat((feats, *p_input), dim=-1)
        if len(mem_input.size()) == 2:
            mem_input = mem_input.unsqueeze(0)

        h_t, mem_state = self.working_memory(mem_input, mem_state)

        action_logits  = self.actor(h_t)
        value_estimate = self.critic(h_t)

        return action_logits, value_estimate, mem_state

    def get_init_states(self, device='cpu'):
        h0 = T.zeros(1, 1, self.working_memory.hidden_size).float().to(device)
        c0 = T.zeros(1, 1, self.working_memory.hidden_size).float().to(device)
        return (h0, c0) if self.cell_type in ["lstm", "rgu"] else h0


class A3C_StackedLSTM(nn.Module):

    def __init__(self, 
            input_dim, 
            hidden_dim, 
            num_actions,
            device="cpu",
    ):
        super(A3C_StackedLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device 

        feat_dim = 128

        self.encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim),
            nn.ReLU(),
        )

        # short-term memory
        # self.lstm_1 = nn.LSTM(feat_dim+1, hidden_dim)
        # self.lstm_2 = nn.LSTM(feat_dim+num_actions+hidden_dim, hidden_dim // 2)
        
        self.lstm_1 = nn.LSTM(feat_dim, hidden_dim)
        self.lstm_2 = nn.LSTM(hidden_dim+1+num_actions, hidden_dim)

        self.actor  = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        T.nn.init.orthogonal_(self.actor.weight, gain=0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight, gain=1.0)
        self.critic.bias.data.fill_(0)

    def forward(self, obs, p_input, state_1, state_2):

        p_action, p_reward = p_input
        
        feats = self.encoder(obs)
        # x_t1 = T.cat((feats, p_reward), dim=-1).unsqueeze(1)
        # x_t1 = T.cat((feats, p_action, p_reward), dim=-1).unsqueeze(1)
    
        _, (h_t1, c_t1) = self.lstm_1(feats.unsqueeze(1), state_1)

        x_t2 = T.cat((h_t1.squeeze(0), p_reward, p_action), dim=-1).unsqueeze(1)

        _, (h_t2, c_t2) = self.lstm_2(x_t2, state_2) 

        action_logits  = self.actor(h_t2)
        value_estimate = self.critic(h_t2)

        return action_logits, value_estimate, (h_t1, c_t1), (h_t2, c_t2)

    def get_init_states(self, layer=1):
        hidden_size = self.lstm_1.hidden_size if layer == 1 else self.lstm_2.hidden_size
        h0 = T.zeros(1, 1, hidden_size).float().to(self.device)
        c0 = T.zeros(1, 1, hidden_size).float().to(self.device)
        return (h0, c0)
