import os
import yaml
import pickle
import argparse

import numpy as np
import torch as T
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm 
from collections import namedtuple

import deepmind_lab as lab 

from harlow import HarlowWrapper
from models.a3c_lstm import A3C_LSTM
# from models.a3c_dnd_lstm import A3C_DND_LSTM

Rollout = namedtuple('Rollout',
                        ('state', 'action', 'reward', 'timestep', 'done', 'policy', 'value'))

class Trainer: 
    def __init__(self, 
        config, 
        environment, 
        shared_model,
        optimizer,
        counter,
        lock,
        rank 
    ):
        self.device = 'cpu'
        self.mode = config["mode"]
        self.seed = config["seed"]

        self.counter = counter
        self.lock = lock 
        self.rank = rank
        self.shared_model = shared_model

        self.env = HarlowWrapper(environment, config["task"])
        self.agent = A3C_LSTM(config["agent"], self.env.num_actions).to(self.device) 

        if optimizer is None:
            self.optim = T.optim.RMSprop(shared_model.parameters(), lr=config["agent"]["lr"])
        else:
            self.optim = optimizer 

        self.gamma = config["agent"]["gamma"]
        self.val_coeff = config["agent"]["value-loss-weight"]
        self.entropy_coeff = config["agent"]["entropy-weight"]
        self.max_grad_norm = config["agent"]["max-grad-norm"]
        self.start_episode = 0

        self.writer = SummaryWriter(log_dir=os.path.join(config["log-path"], config["run-title"] + f"_{self.rank}"))
        self.save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+f"_{self.rank}_"+"{epi:04d}")
        self.save_interval = config["save-interval"]


    def run_episode(self, episode):
        done = False
        total_reward = 0
        p_action, p_reward = [0,0,0], 0

        state = self.env.reset()
        mem_state = self.agent.get_init_states()

        buffer = []
        while not done:

            action_dist, val_estimate, mem_state = self.agent(
                T.tensor(state), 
                (T.tensor(p_action), T.tensor(p_reward)), 
                mem_state
            )

            action_cat = T.distributions.Categorical(action_dist.squeeze())
            action = action_cat.sample()
            action_onehot = np.eye(self.env.num_actions)[action]

            new_state, reward, done, timestep  = self.env.step(int(action))

            print(f"{self.rank}: {timestep}")

            # ('state', 'action', 'reward', 'timestep', 'done', 'policy', 'value')
            buffer += [Rollout(
                state, 
                action_onehot,
                reward,
                timestep,
                done,
                action_dist,
                val_estimate
            )]

            with self.lock:
                self.counter.value += 1

            state = new_state
            p_reward = reward
            p_action = action_onehot

            total_reward += reward

        # boostrap final observation 
        _, val_estimate, _ = self.agent(
            T.tensor(state), 
            (T.tensor(p_action), T.tensor(p_reward)), 
            mem_state
        )

        buffer += [Rollout(*[None]*6, val_estimate)]

        return total_reward, buffer

    def a3c_loss(self, buffer, gamma, lambd=1.0):
        # bootstrap discounted returns with final value estimates
        _, _, _, _, _, _, last_value = buffer[-1]
        returns = last_value.data
        advantages = 0

        all_returns = T.zeros(len(buffer)-1, device=self.device)
        all_advantages = T.zeros(len(buffer)-1, device=self.device)
        # run Generalized Advantage Estimation, calculate returns, advantages
        for t in reversed(range(len(buffer) - 1)):
            # ('state', 'action', 'reward', 'timestep', 'done', 'policy', 'value')
            _, _, reward, _, done, _, value = buffer[t]

            _, _, _, _, _, _, next_value = buffer[t+1]

            mask = ~done

            returns = reward + returns * gamma * mask

            deltas = reward + next_value.data * gamma * mask - value.data
            advantages = advantages * gamma * lambd * mask + deltas

            all_returns[t] = returns 
            all_advantages[t] = advantages

        batch = Rollout(*zip(*buffer))

        policy = T.cat(batch.policy[:-1], dim=0).squeeze().to(self.device)
        action = T.tensor(batch.action[:-1], device=self.device)
        values = T.tensor(batch.value[:-1], device=self.device)
        
        logits = (policy * action).sum(1)
        policy_loss = -(T.log(logits) * all_advantages).mean()
        value_loss = 0.5 * (all_returns - values).pow(2).mean()
        entropy_reg = -(policy * T.log(policy)).mean()

        loss = self.val_coeff * value_loss + policy_loss - self.entropy_coeff * entropy_reg

        return loss 


    def train(self, max_episodes):

        T.manual_seed(self.seed + self.rank)
        np.random.seed(self.seed + self.rank)
        T.random.manual_seed(self.seed + self.rank)

        total_rewards = np.zeros(max_episodes)

        self.agent.train()
        print(f"Trainer {self.rank} starting")
        episode = self.start_episode
        while True:
                        
            # sync agent with master            
            print("Syncing")
            state_dict = self.shared_model.state_dict()
            self.agent.load_state_dict(state_dict)

            print("Running Episode")
            reward, buffer = self.run_episode(episode)
        
            self.optim.zero_grad()
            loss = self.a3c_loss(buffer, self.gamma) 
            loss.backward()
            if self.max_grad_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)

            self.ensure_shared_grads()
            self.optim.step()
            total_rewards[episode] = reward

            avg_reward_10 = total_rewards[max(0, episode-10):(episode+1)].mean()
            avg_reward_100 = total_rewards[max(0, episode-100):(episode+1)].mean()
            self.writer.add_scalar("perf/reward_t", reward, episode)
            self.writer.add_scalar("perf/avg_reward_10", avg_reward_10, episode)
            self.writer.add_scalar("perf/avg_reward_100", avg_reward_100, episode)
            self.writer.add_scalar("losses/total_loss", loss.item(), episode)
            if self.max_grad_norm > 0:
                self.writer.add_scalar("losses/grad_norm", grad_norm, episode)
            
            print(f"Worker {self.rank}: Episode {episode}/{max_episodes} | Reward: {reward} | Last 10: {avg_reward_10:.4f} | Loss: {loss.item():.4f}")
            episode += 1
        
            if (episode) % self.save_interval == 0:
                T.save({
                    "state_dict": self.shared_model.state_dict(),
                    "avg_reward_100": avg_reward_100,
                    'last_episode': episode,
                }, self.save_path.format(epi=episode+1) + ".pt")

    def ensure_shared_grads(self):
        for param, shared_param in zip(self.agent.parameters(), 
                                    self.shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad