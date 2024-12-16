import numpy as np
import torch as th



class DataBuffer:
    def __init__(self, nS, nA, pred_rew=True, buffer_capacity = 100000):
         # number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # its tells us num of times record() was called.
        self.buffer_counter = 0

        self.track_rew = pred_rew
        if self.track_rew:
            self.rew_buffer = np.zeros((self.buffer_capacity, 1))

        self.state_buffer = np.zeros((self.buffer_capacity, nS))
        self.action_buffer = np.zeros((self.buffer_capacity, nA))
        self.next_state_buffer = np.zeros((self.buffer_capacity, nS))

    def add(self, state, act, next_state, rew=None):
        # set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = state
        self.action_buffer[index] = act
        self.next_state_buffer[index] = next_state
        if self.track_rew:
            self.rew_buffer[index] = rew

        self.buffer_counter += 1

    def get_batch(self):

        num_samples = min(self.buffer_counter,self.buffer_capacity)
        # Convert to tensors
        state_batch = th.tensor(self.state_buffer[:num_samples])
        action_batch = th.tensor(self.action_buffer[:num_samples])
        next_state_batch = th.tensor(self.next_state_buffer[:num_samples])

        if self.track_rew:
            rew_batch = th.tensor(self.rew_buffer[:num_samples])
            return state_batch, action_batch, rew_batch, next_state_batch

        return state_batch, action_batch, next_state_batch