import os
import yaml
import pickle
import argparse
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import deepmind_lab as lab 

from tqdm import tqdm 
from collections import namedtuple

from common.shared_optim import SharedAdam, SharedRMSprop
from Harlow_PsychLab.train import train, train_stacked
from Harlow_PsychLab.harlow import HarlowWrapper
from models.a3c_lstm import A3C_LSTM, A3C_StackedLSTM
from models.a3c_conv_lstm import A3C_ConvLSTM, A3C_ConvStackedLSTM

   
if __name__ == "__main__":

    mp.set_start_method("spawn")
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, 
                        default="/home/bkhmsi/Documents/Projects/lab/Meta-RL-Harlow/Harlow_PsychLab/config.yaml", 
                        help='path of config file')
    parser.add_argument('--length', type=int, default=3600,
                        help='Number of steps to run the agent')
    parser.add_argument('--width', type=int, default=84,
                        help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=84,
                        help='Vertical size of the observations')
    parser.add_argument('--fps', type=int, default=60,
                        help='Number of frames per second')
    parser.add_argument('--runfiles_path', type=str, default=None,
                        help='Set the runfiles path to find DeepMind Lab data')
    parser.add_argument('--level_script', type=str,
                        default='contributed/psychlab/harlow',
                        help='The environment level script to load')
    parser.add_argument('--record', type=str, default=None,
                        help='Record the run to a demo file')
    parser.add_argument('--demo', type=str, default=None,
                        help='Play back a recorded demo file')
    parser.add_argument('--demofiles', type=str, default=None,
                        help='Directory for demo files')
    parser.add_argument('--video', type=str, default=None,
                        help='Record the demo run as a video')

    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    task_config = {
        'fps': str(args.fps),
        'width': str(args.width),
        'height': str(args.height)
    }

    if args.record:
        task_config['record'] = args.record
    if args.demo:
        task_config['demo'] = args.demo
    if args.demofiles:
        task_config['demofiles'] = args.demofiles
    if args.video:
        task_config['video'] = args.video

    n_seeds = 1
    device = config["device"]

    ############## Start Here ##############
    print(f"> Running {config['run-title']} {config['mode']}")

    if config["mode"] == "conv-stacked":
        agent = A3C_ConvStackedLSTM(config["agent"], config["task"]["num-actions"])
    elif config["mode"] == "stacked":
        agent = A3C_StackedLSTM(config["agent"], config["task"]["num-actions"])
    elif config["mode"] == "conv-vanilla":
        agent = A3C_ConvLSTM(config["agent"], config["task"]["num-actions"])
    elif config["mode"] == "vanilla":
        agent = A3C_LSTM(config["agent"], config["task"]["num-actions"])
    else:
        raise ValueError(config["mode"])
    

    filepath = os.path.join(
        config["save-path"], 
        config["load-title"], 
        f"{config['load-title']}_{config['start-episode']:04d}.pt"
    )
    print(f"> Loading Checkpoint {filepath}")
    agent.load_state_dict(T.load(filepath, map_location=T.device(config["device"]))["state_dict"])
    
    lab_env = lab.Lab("contributed/psychlab/harlow", ['RGB_INTERLEAVED'], config=task_config)
    env = HarlowWrapper(lab_env, config, 0)
    
    print(agent)
    agent.to(config['device'])
    agent.eval()

    with T.no_grad():

        done = False 
        state = env.reset()
        p_action, p_reward = [0]*config["task"]["num-actions"], 0

        episode_reward = 0

        ht1, ct1 = agent.get_init_states(1, device)
        ht2, ct2 = agent.get_init_states(2, device)

        while not done:

            logit, value, (ht1, ct1), (ht2, ct2) = agent(
                T.tensor([state]).to(device), (
                T.tensor([p_action]).float().to(device), 
                T.tensor([[p_reward]]).float().to(device)), 
                (ht1, ct1), (ht2, ct2)
            )

            logit = logit.squeeze(0)
            prob = F.softmax(logit, dim=-1)
            action = prob.multinomial(num_samples=1).detach()

            state, reward, done, _ = env.step(int(action))

            if reward == 0.2 and config["save-featmaps"]:
                state, _, _, _ = env.step(0)
                state, _, _, _ = env.step(1)
                layer = 9
                path = f"/home/bkhmsi/Documents/Projects/lab/Meta-RL-Harlow/featmaps_{config['start-episode']:04d}_{layer}.npy"
                agent.save_featmaps(T.tensor([state]), path, layer)
                print("> Feature Maps Saved")
                exit()

            episode_reward += reward

            p_action = np.eye(env.num_actions)[int(action)]
            p_reward = reward

    env.save_frames(f"/home/bkhmsi/Documents/Projects/lab/Meta-RL-Harlow/sample_{config['start-episode']:04d}.gif")
    print(f"Episode Reward: {episode_reward}")
    
        