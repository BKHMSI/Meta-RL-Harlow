import numpy as np 
from skimage.color import rgb2gray


class HarlowWrapper:
  """A gym-like wrapper environment for DeepMind Lab.
  Attributes:
      env: The corresponding DeepMind Lab environment.
      max_length: Maximum number of frames
  Args:
      env (deepmind_lab.Lab): DeepMind Lab environment.
  """
  def __init__(self, env, config):
    self.env = env
    self.max_length = config["max-length"]
    self.num_trials = config["num-trials"]
    self.num_actions = 3 # {no-op, left, right}
    self.frames = []
    self.reset()

  def step(self, action, repeat=4):
    
    action_vec = self._create_action(action)
    
    obs = self.env.observations()
    reward = self.env.step(action_vec, num_steps=repeat)
    self.frames += [obs['RGB_INTERLEAVED']]

    timestep = self.env.num_steps() 
    done = not self.env.is_running() or timestep > self.max_length

    return obs['RGB_INTERLEAVED'], reward, done, timestep

  def reset(self):
    self.env.reset()
    obs = self.env.observations()
    self.frames = []
    return obs['RGB_INTERLEAVED']

  def _create_action(self, action):
    """
      action: no-op (0), left (1), right(-1)
    """
    return np.array([action,0,0,0,0,0,0])