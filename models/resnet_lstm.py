import numpy as np

import torch as T
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        original_model = torchvision.models.resnet18(pretrained=False)
        self.features = T.nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(1, -1)

class ResNet_LSTM(nn.Module):

    def __init__(self, config, num_actions, pretrained=True):
        super(ResNet_LSTM, self).__init__()
    
        self.encoder = Encoder()

        self.cell_type = config["cell-type"]

        # for param in self.encoder.parameters():
        #     param.requires_grad = False 

        self.lstm = nn.LSTM(512+1+num_actions, config["mem-units"])
        self.actor = nn.Linear(config["mem-units"], num_actions)
        self.critic = nn.Linear(config["mem-units"], 1)
        
        # intialize actor and critic weights
        T.nn.init.orthogonal_(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight.data, 1)
        self.critic.bias.data.fill_(0)

    def forward(self, obs, p_input, state):

        feats = self.encoder(obs)

        x_t = T.cat((feats, *p_input), dim=-1).unsqueeze(0)

        output, state_out = self.lstm(x_t, state)
        
        action_logits = self.actor(output)
        value_estimate = self.critic(output)

        return action_logits, value_estimate, state_out

    def get_init_states(self, device='cuda'):
        hsize = self.lstm.hidden_size 
        h0 = T.zeros(1, 1, hsize).float().to(device)
        c0 = T.zeros(1, 1, hsize).float().to(device)
        return (h0, c0) if self.cell_type == "lstm" else h0