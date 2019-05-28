import numpy as np
import random

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    
    def push(self, state, action, next_state, reward, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position+1) % self.capacity

    
    def sample(self, batch_size):
        sampled = random.sample(self.buffer, batch_size)
        batch = list(map(lambda x: np.asarray(x), zip(*sampled)))

        return batch


    def __len__(self):
        return len(self.buffer)