import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.dnd import DND
from models.ep_lstm import EpLSTM

class A3C_LSTM(nn.Module):

    def __init__(self, input_dim, hidden_size, num_actions):
        super(A3C_LSTM, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        self.working_memory = nn.LSTM(64+4, hidden_size)
        
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
        return (h0, c0)

class A3C_DND_LSTM(nn.Module):

    def __init__(self, 
            input_dim, 
            hidden_dim, 
            num_actions,
            dict_key_dim,
            dict_len,
            kernel='l2', 
            bias=True,
            device="cpu",
    ):
        super(A3C_DND_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.device = device 

        self.encoder = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        # long-term memory 
        self.dnd = DND(dict_len, dict_key_dim, hidden_dim, kernel)

        # short-term memory
        self.ep_lstm = EpLSTM(
            input_size=64+4,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False
        )
        
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # reset lstm parameters
        self.ep_lstm.reset_parameters()
        # reset dnd 
        self.reset_memory()
        # intialize actor and critic weights
        T.nn.init.orthogonal_(self.actor.weight, gain=0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight, gain=1.0)
        self.critic.bias.data.fill_(0)

    def forward(self, obs, p_input, mem_state, cue=None):

        feats = self.encoder(obs)
        x_t = T.cat((feats, *p_input), dim=-1)

        if cue is None:
            m_t = self.dnd.get_memory(feats).to(self.device)
        else:
            m_t = self.dnd.get_memory(cue).to(self.device)
    
        _, (h_t, c_t) = self.ep_lstm((x_t.unsqueeze(1), m_t.unsqueeze(1)), mem_state)

        action_logits = self.actor(h_t)
        value_estimate = self.critic(h_t)

        return action_logits, value_estimate, (h_t, c_t), feats

    def get_init_states(self):
        h0 = T.zeros(1, 1, self.ep_lstm.hidden_size).float().to(self.device)
        c0 = T.zeros(1, 1, self.ep_lstm.hidden_size).float().to(self.device)
        return (h0, c0)

    def turn_off_encoding(self):
        self.dnd.encoding_off = True

    def turn_on_encoding(self):
        self.dnd.encoding_off = False

    def turn_off_retrieval(self):
        self.dnd.retrieval_off = True

    def turn_on_retrieval(self):
        self.dnd.retrieval_off = False

    def reset_memory(self):
        self.dnd.reset_memory()

    def save_memory(self, mem_key, mem_val):
        self.dnd.save_memory(mem_key, mem_val, replace_similar=True, threshold=0.9)

    def retrieve_memory(self, query_key):
        return self.dnd.get_memory(query_key)

    def get_all_mems(self):
        n_mems = len(self.dnd.keys)
        K = [self.dnd.keys[i] for i in range(n_mems)]
        V = [self.dnd.vals[i] for i in range(n_mems)]
        return K, V
