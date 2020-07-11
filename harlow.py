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
  def __init__(self, env, max_length):
    self.env = env
    self.max_length = max_length
    self.frames = []
    self.reset()

  def step(self, action):
    tmp_obs = self.env.observations()
    reward = self.env.step(np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.intc), num_steps=1)
    self.frames.append(tmp_obs['RGB_INTERLEAVED'])
    
    done = not self.env.is_running() or self.env.num_steps() > self.max_length
    if done:
      self.reset()
    
    print("\033[34mAction Taken: " + ("Left" if action[0] > 0 else "Right") + "\033[0m")

    obs = self.env.observations()
    reward += self.env.step(action, num_steps=1)
    self.frames.append(obs['RGB_INTERLEAVED'])

    if reward > 0:
        print("\033[1;32mTrial reward: " + str(reward) + " :)\033[0m")
    else:
        print("\033[1;31mTrial reward: " + str(reward) + " :(\033[0m")
    return obs['RGB_INTERLEAVED'], reward, done, self.env.num_steps()

  def reset(self):
    self.env.reset()
    obs = self.env.observations()
    self.frames = []
    return obs['RGB_INTERLEAVED']