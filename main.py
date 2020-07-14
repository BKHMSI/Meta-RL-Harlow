import os
import yaml
import pickle
import argparse
import numpy as np

import torch as T
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm 
from collections import namedtuple

# import deepmind_lab as lab 

import shared_optim
from train import train, train_stacked
from harlow import HarlowWrapper
from models.a3c_lstm import A3C_LSTM, A3C_StackedLSTM

   
if __name__ == "__main__":

    mp.set_start_method("spawn")
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, 
                        default="/home/bkhmsi/Documents/Projects/lab/Meta-RL-Harlow/config.yaml", 
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
    base_seed = config["seed"]
    base_run_title = config["run-title"]
    for seed_idx in range(1, n_seeds + 1):
        config["run-title"] = base_run_title + f"_{seed_idx}"
        config["seed"] = base_seed * seed_idx
        
        exp_path = os.path.join(config["save-path"], config["run-title"])
        if not os.path.isdir(exp_path): 
            os.mkdir(exp_path)
        
        out_path = os.path.join(exp_path, os.path.basename(args.config))
        with open(out_path, 'w') as fout:
            yaml.dump(config, fout)

        ############## Start Here ##############
        print(f"> Running {config['run-title']}")

        if config["mode"] == "stacked":
            shared_model = A3C_StackedLSTM(config["agent"], config["task"]["num-actions"])
        else:
            shared_model = A3C_LSTM(config["agent"], config["task"]["num-actions"])
        shared_model.share_memory()

        optimizer = shared_optim.SharedAdam(shared_model.parameters(), lr=config["agent"]["lr"])
        optimizer.share_memory()
   
        processes = []
        counter = mp.Value('i', 0)
        lock = mp.Lock()
        
        T.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        T.random.manual_seed(config["seed"])

        if config["resume"]:
            filepath = os.path.join(
                config["save-path"], 
                config["run-title"], 
                f"{config['run-title']}_{config['start-episode']}.pt"
            )
            shared_model.load_state_dict(T.load(filepath)["state_dict"])

        train_target = train_stacked if config["mode"] == "stacked" else train
        for rank in range(config["agent"]["n-workers"]):
            p = mp.Process(target=train_target, args=(
                config,
                shared_model,
                optimizer,
                rank,
                lock,
                counter,
                task_config
            ))
            p.start()
            processes += [p]

        for p in processes:
            p.join()



    