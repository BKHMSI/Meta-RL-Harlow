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
        print(f"> Running {config['run-title']} {config['mode']} using {config['optimizer']}")

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
        
        print(agent)
        agent.to(config['device'])

        optimizer = T.optim.RMSprop(agent.parameters(), lr=config["agent"]["lr"])
        
        T.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        T.random.manual_seed(config["seed"])
        update_counter = 0

        if config["resume"]:
            filepath = os.path.join(
                config["save-path"], 
                config["load-title"], 
                f"{config['load-title']}_{config['start-episode']}.pt"
            )
            print(f"> Loading Checkpoint {filepath}")
            model_data = T.load(filepath, map_location=T.device(config["device"]))
            update_counter = model_data["update_counter"]
            agent.load_state_dict(model_data["state_dict"])
        
        lab_env = lab.Lab("contributed/psychlab/harlow", ['RGB_INTERLEAVED'], config=task_config)
        env = HarlowWrapper(lab_env, config, 0)
        
        agent.train()

        ### hyper-parameters ###
        gamma = config["agent"]["gamma"]
        gae_lambda = config["agent"]["gae-lambda"]
        val_coeff = config["agent"]["value-loss-weight"]
        entropy_coeff = config["agent"]["entropy-weight"]
        n_step_update = config["agent"]["n-step-update"]

        writer = SummaryWriter(log_dir=os.path.join(config["log-path"], config["run-title"]))
        save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}")
        save_interval = config["save-interval"]

        done = True 
        state = env.reset()
        p_action, p_reward = [0]*config["task"]["num-actions"], 0

        episode_reward = 0
        total_rewards = []

        while True:

            if done:
                ht1, ct1 = agent.get_init_states(1, device)
                ht2, ct2 = agent.get_init_states(2, device)
            else:
                ht1, ct1 = ht1.detach(), ct1.detach()
                ht2, ct2 = ht2.detach(), ct2.detach()

            values = []
            log_probs = []
            rewards = []
            entropies = []

            for _ in range(n_step_update):

                logit, value, (ht1, ct1), (ht2, ct2) = agent(
                    T.tensor([state]).float().to(device), (
                    T.tensor([p_action]).float().to(device), 
                    T.tensor([[p_reward]]).float().to(device)), 
                    (ht1, ct1), (ht2, ct2)
                )

                logit = logit.squeeze(0)

                prob = F.softmax(logit, dim=-1)
                log_prob = F.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies += [entropy]
                action = prob.multinomial(num_samples=1).detach()

                log_prob = log_prob.gather(1, action)

                state, reward, done, _ = env.step(int(action))

                episode_reward += reward

                p_action = np.eye(env.num_actions)[int(action)]
                p_reward = reward
                
                log_probs += [log_prob]
                values += [value]
                rewards += [reward]

                if done:
                    state = env.reset()
                    total_rewards += [episode_reward]
                    
                    avg_reward_100 = np.array(total_rewards[-100:]).mean()
                    writer.add_scalar("perf/reward_t", episode_reward, env.episode_num)
                    writer.add_scalar("perf/avg_reward_100", avg_reward_100, env.episode_num)
        
                    episode_reward = 0
                    if env.episode_num % save_interval == 0:
                        T.save({
                            "state_dict": agent.state_dict(),
                            "avg_reward_100": avg_reward_100,
                            "update_counter": update_counter
                        }, save_path.format(epi=env.episode_num) + ".pt")

                    break 

            R = T.zeros(1, 1).to(device)
            if not done:
                _, value, _, _ = agent(
                    T.tensor([state]).float().to(device), (
                    T.tensor([p_action]).float().to(device), 
                    T.tensor([[p_reward]]).float().to(device)), 
                    (ht1, ct1), (ht2, ct2)
                )
                R = value.detach()
            
            values += [R]

            policy_loss = 0
            value_loss = 0
            gae = T.zeros(1, 1).to(device)

            for i in reversed(range(len(rewards))):
                R = gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)
                
                # Generalized Advantage Estimation
                delta_t = rewards[i] + gamma * values[i + 1] - values[i]
                gae = gae * gamma * gae_lambda + delta_t

                policy_loss = policy_loss - \
                    log_probs[i] * gae.detach() - entropy_coeff * entropies[i]

            loss = policy_loss + val_coeff * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_counter += 1
            writer.add_scalar("losses/total_loss", loss.item(), update_counter)



        