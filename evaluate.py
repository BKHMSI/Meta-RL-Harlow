import os
import yaml
import pickle
import argparse
import numpy as np

import torch as T
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm 
from datetime import datetime 
from collections import namedtuple

from Harlow_Simple.harlow import HarlowSimple
from models.a3c_lstm_simple import A3C_LSTM

def run_episode(agent, env, device="cpu"):

    agent.eval()

    done = False 
    state = env.reset()
    p_action, p_reward = [0,0,0], 0
    ht, ct = agent.get_init_states(device)

    while not done:

        logit, _, (ht, ct) = agent(
            T.tensor(state).float().to(device), (
            T.tensor(p_action).unsqueeze(0).float().to(device), 
            T.tensor([p_reward]).unsqueeze(0).float().to(device)), 
            (ht, ct)
        )

        action = T.argmax(F.softmax(logit, dim=-1), -1)

        state, reward, done, _ = env.step(action)

        p_action = np.eye(env.n_actions)[action]
        p_reward = reward

    env.reset()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="Harlow_Simple/config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    load_path = config["load-path"]
    save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}.gif")

    agent = A3C_LSTM(mem_units=config["agent"]["mem-units"], num_actions=3)
    agent.load_state_dict(T.load(load_path)["state_dict"])

    env = HarlowSimple(visualize=True, save_interval=1, save_path=save_path)

    run_episode(agent, env)

    