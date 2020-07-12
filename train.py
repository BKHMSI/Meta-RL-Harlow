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
    def __init__(self, config, env):
        self.device = 'cpu'
        self.mode = config["mode"]
        self.seed = config["seed"]

        self.env = HarlowWrapper(env, config["task"], config["save-interval"])
        self.agent = A3C_LSTM(config["agent"], self.env.num_actions)

        self.optim = T.optim.RMSprop(self.agent.parameters(), lr=config["agent"]["lr"])
       
        self.gamma = config["agent"]["gamma"]
        self.val_coeff = config["agent"]["value-loss-weight"]
        self.entropy_coeff = config["agent"]["entropy-weight"]
        self.max_grad_norm = config["agent"]["max-grad-norm"]
        self.start_episode = 0

        self.writer = SummaryWriter(log_dir=os.path.join(config["log-path"], config["run-title"]))
        self.save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}")
        self.save_interval = config["save-interval"]

        self.finished_trials = 0


    def run_episode(self, episode):
        total_reward = 0
        mem_state = self.agent.get_init_states()

        buffer = []
        state = self.env.reset()
        for trial in range(self.env.num_trials):
            done = False
            p_action, p_reward = [0,0,0], 0
            while not done:

                action_dist, val_estimate, mem_state = self.agent(
                    T.tensor(state), 
                    (T.tensor(p_action).unsqueeze(0).float(), 
                    T.tensor([p_reward]).unsqueeze(0).float()), 
                    mem_state
                )

                action_cat = T.distributions.Categorical(action_dist.squeeze())
                action = action_cat.sample()
                action_onehot = np.eye(self.env.num_actions)[action]

                new_state, reward, done, timestep = self.env.step(int(action))

                if done and self.env.num_steps() < self.env.max_length:
                    self.finished_trials += 1

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

                state = new_state
                p_reward = reward
                p_action = action_onehot

                total_reward += reward

        # boostrap final observation 
        _, val_estimate, _ = self.agent(
            T.tensor(state), 
            (T.tensor(p_action).unsqueeze(0).float(), 
            T.tensor([p_reward]).unsqueeze(0).float()), 
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

        total_rewards = np.zeros(max_episodes)

        self.agent.train()
        progress = tqdm(range(self.start_episode, max_episodes))
        for episode in progress:

            reward, buffer = self.run_episode(episode)
        
            self.optim.zero_grad()
            loss = self.a3c_loss(buffer, self.gamma) 
            loss.backward()
            if self.max_grad_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
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
            
            progress.set_description(f"Episode {episode}/{max_episodes} | Reward: {reward} | Last 10: {avg_reward_10:.4f} | Loss: {loss.item():.4f} | Finished Trials: {self.finished_trials}")
        
            if (episode+1) % self.save_interval == 0:
                T.save({
                    "state_dict": self.agent.state_dict(),
                    "avg_reward_100": avg_reward_100,
                    'last_episode': episode,
                }, self.save_path.format(epi=episode+1) + ".pt")