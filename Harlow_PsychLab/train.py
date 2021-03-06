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

import deepmind_lab as lab 

from Harlow_PsychLab.harlow import HarlowWrapper
from models.a3c_lstm import A3C_LSTM, A3C_StackedLSTM
from models.a3c_conv_lstm import A3C_ConvLSTM, A3C_ConvStackedLSTM
from models.resnet_lstm import ResNet_LSTM

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), 
                                shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
        
def train(config, 
    shared_model, 
    optimizer, 
    rank, 
    task_config,
    counter,
):

    T.manual_seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)
    T.random.manual_seed(config["seed"] + rank)
    device = config["device"]

    lab_env = lab.Lab("contributed/psychlab/harlow", ['RGB_INTERLEAVED'], config=task_config)
    env = HarlowWrapper(lab_env, config, rank)

    if config["mode"] == "resnet":
        agent = ResNet_LSTM(config["agent"], env.num_actions)
    elif config["mode"] == "conv-vanilla":
        agent = A3C_ConvLSTM(config["agent"], env.num_actions)
    elif config["mode"] == "vanilla":
        agent = A3C_LSTM(config["agent"], env.num_actions)
    else:
        raise ValueError(config["mode"])

    agent.to(device)
    agent.train()

    ### hyper-parameters ###
    gamma = config["agent"]["gamma"]
    gae_lambda = config["agent"]["gae-lambda"]
    val_coeff = config["agent"]["value-loss-weight"]
    entropy_coeff = config["agent"]["entropy-weight"]
    n_step_update = config["agent"]["n-step-update"]

    writer = SummaryWriter(log_dir=os.path.join(config["log-path"], config["run-title"] + f"_{rank}"))
    save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}")
    save_interval = config["save-interval"]

    done = True 
    state = env.reset()
    p_action, p_reward = [0]*config["task"]["num-actions"], 0

    print('='*50)
    print(f"Starting Worker {rank}")
    print('='*50)

    episode_reward = 0
    update_counter = counter
    total_rewards = []

    while True:

        agent.load_state_dict(shared_model.state_dict())
        if done:
            rnn_state = agent.get_init_states(device)
        else:
            if config["agent"]["cell-type"] == "lstm":
                rnn_state = rnn_state[0].detach(), rnn_state[1].detach()
            else:
                rnn_state = rnn_state.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for _ in range(n_step_update):

            logit, value, rnn_state = agent(
                T.tensor([state]).to(device), (
                T.tensor([p_action]).float().to(device), 
                T.tensor([[p_reward]]).float().to(device)), 
                rnn_state
            )

            logit = logit.squeeze(0)

            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies += [entropy]
            action = prob.multinomial(num_samples=1).detach()

            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(int(action))

            # if done: 
            #     env.save_frames(os.path.join(config["save-path"], "frames.gif"))
            #     exit()

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
                        "state_dict": shared_model.state_dict(),
                        "avg_reward_100": avg_reward_100,
                        "update_counter": update_counter
                    }, save_path.format(epi=env.episode_num) + ".pt")

                break 

        R = T.zeros(1, 1).to(device)
        if not done:
            _, value, _ = agent(
                T.tensor([state]).to(device), (
                T.tensor([p_action]).float().to(device), 
                T.tensor([[p_reward]]).float().to(device)), 
                rnn_state
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
        ensure_shared_grads(agent, shared_model)
        optimizer.step()

        update_counter += 1
        writer.add_scalar("losses/total_loss", loss.item(), update_counter)

def train_stacked(config, 
    shared_model, 
    optimizer, 
    rank, 
    task_config,
    counter,
):

    T.manual_seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)
    T.random.manual_seed(config["seed"] + rank)
    device = config["device"]

    lab_env = lab.Lab("contributed/psychlab/harlow", ['RGB_INTERLEAVED'], config=task_config)
    env = HarlowWrapper(lab_env, config, rank)
    
    if config["mode"] == "conv-stacked":
        agent = A3C_ConvStackedLSTM(config["agent"], env.num_actions)
    elif config["mode"] == "stacked":
        agent = A3C_StackedLSTM(config["agent"], env.num_actions)
    else:
        raise ValueError(config["mode"])    
    
    agent.to(device)
    agent.train()

    ### hyper-parameters ###
    gamma = config["agent"]["gamma"]
    gae_lambda = config["agent"]["gae-lambda"]
    val_coeff = config["agent"]["value-loss-weight"]
    entropy_coeff = config["agent"]["entropy-weight"]
    n_step_update = config["agent"]["n-step-update"]

    writer = SummaryWriter(log_dir=os.path.join(config["log-path"], config["run-title"] + f"_{rank}"))
    save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}")
    save_interval = config["save-interval"]

    done = True 
    state = env.reset()
    p_action, p_reward = [0]*config["task"]["num-actions"], 0

    print('='*50)
    print(f"Starting Worker {rank}")
    print('='*50)

    episode_reward = 0
    update_counter = 0
    total_rewards = []

    while True:

        agent.load_state_dict(shared_model.state_dict())
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
                T.tensor([state]).to(device), (
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
                        "state_dict": shared_model.state_dict(),
                        "avg_reward_100": avg_reward_100,
                        "update_counter": update_counter
                    }, save_path.format(epi=env.episode_num) + ".pt")

                break 

        R = T.zeros(1, 1).to(device)
        if not done:
            _, value, _, _ = agent(
                T.tensor([state]).to(device), (
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
        ensure_shared_grads(agent, shared_model)
        optimizer.step()

        update_counter += 1
        writer.add_scalar("losses/total_loss", loss.item(), update_counter)