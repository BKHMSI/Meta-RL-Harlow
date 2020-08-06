import os
import imageio
import numpy as np 

PIXELS_PER_ACTION = 1

class HarlowWrapper:
  """A gym-like wrapper environment for DeepMind Lab.
  Attributes:
      env: The corresponding DeepMind Lab environment.
      max_length: Maximum number of frames
  Args:
      env (deepmind_lab.Lab): DeepMind Lab environment.
  """
  def __init__(self, env, config, rank):
    self.env = env
    self.max_length = config["task"]["max-length"]
    self.num_trials = config["task"]["num-trials"]
    self.reward_scheme = config["task"]["reward-scheme"]
    self.save_interval = config["save-interval"]
    self.save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}.gif")
    self.rank = rank 
    self.frames = []
    self.num_actions = config["task"]["num-actions"] # {no-op, left, right}
    self.episode_num = config["start-episode"]
    self.trial_num = 0
    self.reset()

  def step(self, action, repeat=4):
    '''
      Rewards: 
        fixation      1.00
        correct image 5.00
        wrong image  -5.00
        Time Penalty -0.01
    '''
    
    action_vec = self._create_action(action)
    
    obs = self.env.observations()
    reward = self.env.step(action_vec, num_steps=repeat)
    self.frames += [obs['RGB_INTERLEAVED']]

    if reward in [-5, 5]:
      self.trial_num += 1

    reward = self._return_reward(reward) 

    timestep = self.num_steps() 
    done = not self.env.is_running() or timestep > self.max_length or self.trial_num >= self.num_trials

    return self._preprocess(obs['RGB_INTERLEAVED']), reward, done, timestep

  def reset(self):
    self.env.reset()
    obs = self.env.observations()

    if len(self.frames) > 0:
        self.episode_num += 1

    if self.episode_num > 0 and len(self.frames) > 0 and self.rank == 0 and (self.episode_num % self.save_interval) == 0:
      filepath = self.save_path.format(epi=self.episode_num)
      imageio.mimsave(filepath, self.frames)

    self.trial_num = 0
    self.frames = []
    return self._preprocess(obs['RGB_INTERLEAVED'])

  def num_steps(self):
    return self.env.num_steps()

  def snapshot(self):
    obs = self.env.observations()['RGB_INTERLEAVED']
    filepath = os.path.join(os.path.dirname(self.save_path), "snapshot.png")
    imageio.imsave(filepath, obs)

  def save_frames(self, path):
      imageio.mimsave(path, self.frames)

  def _preprocess(self, obs):
    obs = obs.astype(np.float32)
    obs = obs / 255.0
    obs = (obs - 0.5) / 0.5
    return np.einsum('ijk->kij', obs)

  def _return_reward(self, reward):
    if self.reward_scheme == 0:
      return reward / 5.
    elif self.reward_scheme == 1:
      if reward in [-5, 1]: reward == 0
      return reward / 5.
    elif self.reward_scheme == 2:
      if reward == -5: reward == 0
      return reward / 5.
    else:
      return reward

  def _create_action(self, action):
    """
      action: no-op (0), left (1), right(-1)
    """
    # map_actions = [0, PIXELS_PER_ACTION, -PIXELS_PER_ACTION]
    map_actions = [0, 2, -2, 2, -2, 3, -3]
    return np.array([map_actions[action],0,0,0,0,0,0], dtype=np.intc)