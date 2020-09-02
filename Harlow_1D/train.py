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

from Harlow_1D.harlow import Harlow_1D
from models.a3c_lstm_simple import A3C_LSTM, A3C_StackedLSTM

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), 
                                shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
        

def train(config, 
    shared_model, 
    optimizer, 
    rank
):

    T.manual_seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)
    T.random.manual_seed(config["seed"] + rank)
    device = config["device"]

    env = Harlow_1D()
    if config["mode"] == "vanilla":
        agent = A3C_LSTM(
            config["task"]["input-dim"],
            config["agent"]["mem-units"], 
            config["task"]["num-actions"],
            config["agent"]["cell-type"]
        )
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

    cell_type = config["agent"]["cell-type"]

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
            rnn_state = agent.get_init_states(device)
        else:
            if cell_type == "lstm":
                rnn_state = rnn_state[0].detach(), rnn_state[1].detach()
            elif cell_type == "gru":
                rnn_state = rnn_state.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for _ in range(n_step_update):

            logit, value, rnn_state = agent(
                T.tensor([state]).float().to(device), (
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

            episode_reward += reward

            p_action = np.eye(env.n_actions)[int(action)]
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
                    }, save_path.format(epi=env.episode_num) + ".pt")

                break 

        R = T.zeros(1, 1).to(device)
        if not done:
            _, value, _ = agent(
                T.tensor([state]).float().to(device), (
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

        if env.episode_num > env.n_episodes:
            np.save(os.path.join(os.path.dirname(save_path), f"rewards_{rank}.npy"), env.reward_counter)
            break 

def train_stacked(config, 
    shared_model, 
    optimizer, 
    rank
):

    T.manual_seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)
    T.random.manual_seed(config["seed"] + rank)
    device = config["device"]

    env = Harlow_1D()
    agent = A3C_StackedLSTM(
        config["task"]["input-dim"],
        config["agent"]["mem-units"], 
        config["task"]["num-actions"],
        device=config["device"]
    )

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
    
        if done:
            h_t1, c_t1 = agent.get_init_states(layer=1)
            h_t2, c_t2 = agent.get_init_states(layer=2)
        else:
            h_t1, c_t1 = h_t1.detach(), c_t1.detach()
            h_t2, c_t2 = h_t2.detach(), c_t2.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for _ in range(n_step_update):

            logit, value, (h_t1, c_t1), (h_t2, c_t2) = agent(
                T.tensor([state]).float().to(device), (
                T.tensor([p_action]).float().to(device), 
                T.tensor([[p_reward]]).float().to(device)), 
                (h_t1, c_t1), (h_t2, c_t2)
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

            p_action = np.eye(env.n_actions)[int(action)]
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
                        "update_counter": update_counter,
                    }, save_path.format(epi=env.episode_num) + ".pt")

                break 

        R = T.zeros(1, 1).to(device)
        if not done:
            _, value, _, _ = agent(
                T.tensor([state]).float().to(device), (
                T.tensor([p_action]).float().to(device), 
                T.tensor([[p_reward]]).float().to(device)), 
                (h_t1, c_t1), (h_t2, c_t2)
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

        if env.episode_num > env.n_episodes:
            np.save(os.path.join(os.path.dirname(save_path), f"rewards_{rank}.npy"), env.reward_counter)
            break  



def train_episodic(config, 
    shared_model, 
    optimizer, 
    rank
):

    T.manual_seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)
    T.random.manual_seed(config["seed"] + rank)
    device = config["device"]

    env = Harlow_1D()
    if config["mode"] == "vanilla":
        agent = A3C_LSTM(
            config["task"]["input-dim"],
            config["agent"]["mem-units"], 
            config["task"]["num-actions"],
        )
    elif config["mode"] == "episodic":
        agent =A3C_DND_LSTM(
            config["task"]["input-dim"],
            config["agent"]["mem-units"], 
            config["task"]["num-actions"],
            config["agent"]["dict-len"],
            config["agent"]["dict-kernel"]
        )
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
    p_action, p_reward = [0,0,0], 0

    print('='*50)
    print(f"Starting Trainer {rank}")
    print('='*50)

    episode_reward = 0
    update_counter = 0
    total_rewards = []

    agent.turn_on_encoding()
    agent.turn_on_retrieval()
    # agent.turn_off_encoding()
    # agent.turn_off_retrieval()

    while True:

        agent.load_state_dict(shared_model.state_dict())
        if done:
            ht, ct = agent.get_init_states(device)
        else:
            ht, ct = ht.detach(), ct.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for _ in range(n_step_update):

            logit, value, rnn_state, feats = agent(
                T.tensor([state]).float().to(device), (
                T.tensor([p_action]).float().to(device), 
                T.tensor([[p_reward]]).float().to(device)), 
                (ht, ct)
            )

            logit = logit.squeeze(0)

            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies += [entropy]
            action = prob.multinomial(num_samples=1).detach()

            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(int(action))

            # if reward > 0:
            agent.save_memory(feats, ct)

            episode_reward += reward

            p_action = np.eye(env.n_actions)[int(action)]
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
                if env.episode_num % save_interval == 0 and rank % 4 == 0:
                    T.save({
                        "state_dict": shared_model.state_dict(),
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
                (ht, ct)
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

        if env.episode_num > env.n_episodes:
            if rank % 2 == 0:
                np.save(os.path.join(os.path.dirname(save_path), f"{rank}_rewards.npy"), env.reward_counter)
            break 
