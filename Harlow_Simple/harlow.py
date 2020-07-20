import os
import sys 
import imageio
import numpy as np 

class HarlowSimple:
    """A 1D variant of the Harlow Task.
    """  
    def __init__(self, verbose = False):

        '''environment constants'''
        self.max_length = 75
        self.n_trials   = 6
        self.n_actions  = 3 
        self.n_objects  = 1000
        self.state_len  = 17 # size of state
        self.obs_length = 8  # size of receptive field
        self.obj_offset = 3  
        self.fix_reward = 0.2
        self.obj_reward = 1
        self.time_step  = 0
        self.map_action = [0, 1, -1]
        
        self.episode_num = 0
        self.verbose = verbose 
        self.center = self.state_len // 2
        self.reset()

    @property
    def current(self):
        return self.state[self.center]

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
            self.trial_num += 1
            self._place_fixation()
        elif self.current == self.obj_2:
            reward = self.obj_reward if not self.reward_obj else -self.obj_reward
            self.trial_num += 1
            self._place_fixation()

        obs = self.observation()
        if self.verbose:
            print(f"Observation: {obs}")
            print(f"Reward: {reward} | Pointer: {self.pointer}")

        done = self.trial_num >= self.n_trials or self.time_step >= self.max_length
        return obs, reward, done, self.time_step 


    def reset(self):
        self.trial_num = 0
        self.time_step = 0
        self.episode_num += 1
        self.pointer = self.center

        # initialize state
        self.state = np.zeros(self.state_len)
        self.state[self.center] = 1

        shift = np.random.randint(-self.obj_offset, self.obj_offset)
        if shift == 0: shift = 1
        
        self.state = np.roll(self.state, shift)
        self.pointer -= shift 

        obs = self.observation()
        if self.verbose:
            print(f"Observation: {obs}")
            print(f"Pointer: {self.pointer}")

        # episode objects
        self.obj_1, self.obj_2 = np.random.randint(
            low=2, 
            high=self.n_objects+2, 
            size=2
        )

        self.reward_obj = np.random.rand() < 0.5

        return obs 

if __name__ == "__main__":
    
    env = HarlowSimple(verbose=True)

    while True:
        action = int(input("Left (1) or Right (2): "))
        
        if action <= 0:
            break 

        env.step(action)
