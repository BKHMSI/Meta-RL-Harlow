import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'cifar10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth',
    'cifar100': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth',
}

def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, n_channel):
        super(Encoder, self).__init__()
        cfg = [
            n_channel, 
            n_channel, 
            'M', 
            2*n_channel, 
            2*n_channel, 
            'M', 
            4*n_channel, 
            4*n_channel, 
            'M', 
            # (8*n_channel, 0), 
            # 'M'
        ]

        self.features = make_layers(cfg, batch_norm=True)

    def forward(self, inputs):
        return self.features(inputs)

class A3C_ConvLSTM(nn.Module):

    def __init__(self, config, num_actions, pretrained=True):
        super(A3C_ConvLSTM, self).__init__()
    
        self.encoder = Encoder(config["conv-nchannels"])
        if pretrained:
            m = model_zoo.load_url(model_urls['cifar100'], map_location=T.device('cpu'))
            pretrained_dict = m.state_dict() if isinstance(m, nn.Module) else m
            pretrained_dict = {
                k: v for i, (k, v) in enumerate(pretrained_dict.items()) 
                if i < 24
            }
            self.encoder.load_state_dict(pretrained_dict)

        self.last_conv = make_layers(
            [(2*config["conv-nchannels"], 0), 'M'], 
            batch_norm=True,
            in_channels=4*config["conv-nchannels"]
        )

        self.actor = nn.Linear(64, num_actions)
        self.critic = nn.Linear(64, 1)
        self.lstm_1 = nn.LSTM(4096+1, config["mem-units"])
        self.lstm_2 = nn.LSTM(4096+config["mem-units"]+3, 64)
        
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

        feats = self.last_conv(self.encoder(obs.unsqueeze(0)))
        feats = feats.view(feats.size(0), -1)

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

    def get_init_states(self, layer, device='cuda'):
        hsize = self.lstm_1.hidden_size if layer == 1 else self.lstm_2.hidden_size
        h0 = T.zeros(1, 1, hsize).float().to(device)
        c0 = T.zeros(1, 1, hsize).float().to(device)
        return (h0, c0)