import numpy as np 
import imageio

PIXELS_PER_ACTION = 1

class HarlowWrapper:
  """A gym-like wrapper environment for DeepMind Lab.
  Attributes:
      env: The corresponding DeepMind Lab environment.
      max_length: Maximum number of frames
  Args:
      env (deepmind_lab.Lab): DeepMind Lab environment.
  """
  def __init__(self, env, config, save_interval=100):
    self.env = env
    self.max_length = config["max-length"]
    self.num_trials = config["num-trials"]
    self.num_actions = 3 # {no-op, left, right}
    self.save_interval = save_interval
    self.frames = []
    self.trial_num = 0
    self.reset()

  def step(self, action, repeat=4):
    '''
      Rewards: 
        fixation       1
        correct image  5
        wrong image   -5
    '''
    
    action_vec = self._create_action(action)
    
    obs = self.env.observations()
    reward = self.env.step(action_vec, num_steps=repeat)
    self.frames += [obs['RGB_INTERLEAVED']]

    timestep = self.num_steps() 
    done = not self.env.is_running() or timestep > self.max_length or reward in [-5, 5]

    return self._preprocess(obs['RGB_INTERLEAVED']), reward, done, timestep

  def reset(self):
    self.env.reset()
    obs = self.env.observations()

    if len(self.frames) > 0 and ((self.trial_num+1) % self.save_interval) == 0:
      self.trial_num += 1
      filepath = f"/home/bkhmsi/Documents/Projects/lab/Meta-RL-Harlow/samples/sample_{self.trial_num}.gif"
      imageio.mimsave(filepath, self.frames)

    self.frames = []
    return self._preprocess(obs['RGB_INTERLEAVED'])

  def num_steps(self):
    return self.env.num_steps()

  def _preprocess(self, obs):
    obs = obs.astype(np.float32)
    obs *= (1.0 / 255.0)
    return np.einsum('ijk->kij', obs)

  def _create_action(self, action):
    """
      action: no-op (0), left (1), right(-1)
    """
    map_actions = [0, PIXELS_PER_ACTION, -PIXELS_PER_ACTION]
    return np.array([map_actions[action],0,0,0,0,0,0], dtype=np.intc)