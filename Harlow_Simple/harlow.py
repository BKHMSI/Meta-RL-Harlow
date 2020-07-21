import os
import sys 
import imageio
import numpy as np 
import matplotlib.pyplot as plt 

class HarlowSimple:
    """A 1D variant of the Harlow Task.
    """  
    def __init__(self, 
        verbose = False, 
        visualize = False,
        save_path = None,
        save_interval = None 
    ):

        '''environment constants'''
        self.max_length = 250
        self.n_trials   = 6
        self.n_actions  = 3 
        self.n_objects  = 1000
        self.n_episodes = 10_000
        self.state_len  = 17 # size of state
        self.obs_length = 8  # size of receptive field
        self.obj_offset = 3  
        self.fix_reward = 0.2
        self.obj_reward = 1
        self.time_step  = 0
        self.map_action = [0, 1, -1]
        
        self.episode_num = 0
        self.verbose = verbose 
        self.visualize = visualize
        self.center = self.state_len // 2
        self.reward_counter = np.zeros((self.n_episodes,self.n_trials))

        if self.visualize:
            self.frames = []
            self._create_palette()
            self.save_path = save_path
            self.save_interval = save_interval

    @property
    def current(self):
        return self.state[self.center]

    def _create_palette(self):
        self.palette = []
        for _ in range(self.n_objects*2):
            color = list(np.random.choice(range(256), size=3))
            if color not in self.palette:
                self.palette += [color]

    def _place_objects(self):
        self.state[self.center-self.obj_offset] = self.obj_1
        self.state[self.center+self.obj_offset] = self.obj_2 
        self.state[self.center] = 0

        if np.random.rand() < 0.5:
            self.state[self.center-self.obj_offset] = self.obj_2 
            self.state[self.center+self.obj_offset] = self.obj_1

    def _place_fixation(self):
        self.state = np.zeros(self.state_len)
        if self.pointer > self.center:
            self.state[self.center - self.obj_offset] = 1
        else:
            self.state[self.center + self.obj_offset] = 1

    def observation(self):
        offset = (self.state_len - self.obs_length) // 2
        return self.state[offset:-offset] 

    def step(self, action):
        
        self.time_step += 1

        reward = 0
        self.state = np.roll(self.state, self.map_action[action])
        self.pointer -= self.map_action[action]

        if self.pointer >= self.state_len:
            self.pointer = 0 
        elif self.pointer < 0:
            self.pointer = self.state_len - 1

        if self.current == 1:
            reward = self.fix_reward
            self._place_objects()
        elif self.current == self.obj_1:
            reward = self.obj_reward if self.reward_obj else -self.obj_reward
            if self.reward_obj:
                self.reward_counter[self.episode_num-1][self.trial_num] = 1
            self.trial_num += 1
            self._place_fixation()
        elif self.current == self.obj_2:
            reward = self.obj_reward if not self.reward_obj else -self.obj_reward
            if not self.reward_obj:
                self.reward_counter[self.episode_num-1][self.trial_num] = 1
            self.trial_num += 1
            self._place_fixation()

        obs = self.observation()
        if self.verbose:
            print(f"Observation: {obs}")
            print(f"Reward: {reward} | Pointer: {self.pointer}")

        if self.visualize:
            self._add_frames(obs)

        done = self.trial_num >= self.n_trials or self.time_step >= self.max_length
        return obs, reward, done, self.time_step 


    def reset(self):
        self.trial_num = 0
        self.time_step = 0
        self.episode_num += 1
        self.pointer = self.center

        if self.visualize and len(self.frames) > 0 and self.episode_num % self.save_interval == 0:
            self._save_frames()
        self.frames = []

        # initialize state
        self.state = np.zeros(self.state_len)
        self.state[self.center] = 1

        shift = np.random.randint(-self.obj_offset, self.obj_offset)
        if shift == 0: shift = 1
        
        self.state = np.roll(self.state, shift)
        self.pointer -= shift 

        obs = self.observation()

        if self.visualize:
            self._add_frames(obs)

        if self.verbose:
            print(f"Observation: {obs}")
            print(f"Pointer: {self.pointer}")

        # episode objects
        self.obj_1, self.obj_2 = np.random.randint(
            low=2, 
            high=self.n_objects+2, 
            size=2
        )

        self.obj_1 /= self.n_objects*2 
        self.obj_2 /= self.n_objects*2

        self.reward_obj = np.random.rand() < 0.5

        return obs 

    def _visualize_obs(self, obs):
        size = 20
        bar = np.zeros((size, size*(obs.shape[0]), 3), dtype=np.uint8)
        for i, cell in enumerate(obs):
            if cell == 1:
                bar[:,i*size:i*size+size] = [255, 255, 255]
            elif cell > 0:
                idx = int(cell*self.n_objects*2)
                bar[:,i*size:i*size+size] = self.palette[idx]
        return bar 

    def _add_frames(self, obs):
        bar = self._visualize_obs(obs)
        for _ in range(10):
            self.frames += [bar]

    def _save_frames(self):
        filepath = self.save_path.format(epi=self.episode_num)
        imageio.mimsave(filepath, self.frames)


if __name__ == "__main__":
    
    env = HarlowSimple(verbose=True)

    while True:
        action = int(input("Left (1) or Right (2): "))
        
        if action <= 0:
            break 

        env.step(action)
