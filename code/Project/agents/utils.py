import torch as th
import numpy as np

class ReplayBuffer(object):

    def __init__(self, num_states, num_actions, num_agents=1, buffer_capacity=100000):
        # number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity

        # it tells us how many times record() was called.
        self.buffer_counter = 0

        # instead of list of tuples as the exp.replay concept go
        # we use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, num_agents))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, num_agents))

    @property
    def num_items(self):
        return min(self.buffer_counter, self.buffer_capacity)

    # takes (s, a, r, s') observation tuple as input
    def add(self, obs_tuple):
        # set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    def sample_batch(self, batch_size=64, replace=True):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, batch_size, replace=replace)

        # Convert to tensors (PyTorch)
        state_batch = th.tensor(self.state_buffer[batch_indices], dtype=th.float32)
        action_batch = th.tensor(self.action_buffer[batch_indices], dtype=th.float32)
        reward_batch = th.tensor(self.reward_buffer[batch_indices], dtype=th.float32)
        next_state_batch = th.tensor(self.next_state_buffer[batch_indices], dtype=th.float32)
        done_batch = th.tensor(self.done_buffer[batch_indices], dtype=th.float32)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def dump_all(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        state_batch = th.tensor(self.state_buffer[:record_range], dtype=th.float32)
        action_batch = th.tensor(self.action_buffer[:record_range], dtype=th.float32)
        reward_batch = th.tensor(self.reward_buffer[:record_range], dtype=th.float32)
        next_state_batch = th.tensor(self.next_state_buffer[:record_range], dtype=th.float32)
        done_batch = th.tensor(self.done_buffer[:record_range], dtype=th.float32)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def clear(self):
        self.buffer_counter = 0
