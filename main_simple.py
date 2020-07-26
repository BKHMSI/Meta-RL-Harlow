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

from common.shared_optim import SharedAdam, SharedRMSprop
from Harlow_Simple.train import train
from models.a3c_lstm_simple import A3C_LSTM

   
if __name__ == "__main__":

    mp.set_start_method("spawn")
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="Harlow_Simple/config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    n_seeds = 10
    base_seed = config["seed"]
    base_run_title = config["run-title"]
    for seed_idx in range(2, n_seeds + 1):
        config["run-title"] = base_run_title + f"_{seed_idx}"
        config["seed"] = base_seed * seed_idx
        
        exp_path = os.path.join(config["save-path"], config["run-title"])
        if not os.path.isdir(exp_path): 
            os.mkdir(exp_path)
        
        out_path = os.path.join(exp_path, os.path.basename(args.config))
        with open(out_path, 'w') as fout:
            yaml.dump(config, fout)

        ############## Start Here ##############
        print(f"> Running {config['run-title']} {config['mode']} using {config['optimizer']}")

        if config["mode"] == "vanilla":
            shared_model = A3C_LSTM(
                config["task"]["input-dim"],
                config["agent"]["mem-units"], 
                config["task"]["num-actions"],
            )
        else:
            raise ValueError(config["mode"])

        shared_model.share_memory()
        shared_model.to(config['device'])

        optim_class = SharedAdam if config["optimizer"] == "adam" else SharedRMSprop
        optimizer = optim_class(shared_model.parameters(), lr=config["agent"]["lr"])
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
                config["load-title"], 
                f"{config['load-title']}_{config['start-episode']}.pt"
            )
            print(f"> Loading Checkpoint {filepath}")
            shared_model.load_state_dict(T.load(filepath)["state_dict"])

        for rank in range(config["agent"]["n-workers"]):
            p = mp.Process(target=train, args=(
                config,
                shared_model,
                optimizer,
                rank,
                lock,
                counter,
            ))
            p.start()
            processes += [p]

        for p in processes:
            p.join()



    