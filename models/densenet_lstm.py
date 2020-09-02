import numpy as np

import torch as T
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, freeze = True):
        super(Encoder,self).__init__()
        original_model = torchvision.models.densenet161(pretrained=True)
        self.features = T.nn.Sequential(*list(original_model.children())[:-1])
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x

class DenseNet_StackedLSTM(nn.Module):

    def __init__(self, config, num_actions, pretrained=True):
        super(DenseNet_StackedLSTM, self).__init__()
    
        self.encoder = Encoder(freeze=True)

        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)
        self.lstm_1 = nn.LSTM(2208+1, config["mem-units"])
        self.lstm_2 = nn.LSTM(2208+config["mem-units"]+num_actions, 256)
        
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

        feats = self.encoder(obs)

        input_1 = T.cat((feats, p_reward), dim=-1)
        if len(input_1.size()) == 2:
            input_1 = input_1.unsqueeze(0)

        output_1, state_1 = self.lstm_1(input_1, state_1)

        input_2 = T.cat((feats, output_1.squeeze(0), p_action), dim=-1)
        if len(input_2.size()) == 2:
            input_2 = input_2.unsqueeze(0)

        output_2, state_2 = self.lstm_2(input_2, state_2)
        
        action_logits = self.actor(output_2)
        value_estimate = self.critic(output_2)

        return action_logits, value_estimate, state_1, state_2

    def get_init_states(self, layer, device='cuda'):
        hsize = self.lstm_1.hidden_size if layer == 1 else self.lstm_2.hidden_size
        h0 = T.zeros(1, 1, hsize).float().to(device)
        c0 = T.zeros(1, 1, hsize).float().to(device)
        return (h0, c0)

    def save_featmaps(self, obs, path, layer=5):
        featmaps = self.encoder.features[:layer+1](obs)
        np.save(path, featmaps)
        