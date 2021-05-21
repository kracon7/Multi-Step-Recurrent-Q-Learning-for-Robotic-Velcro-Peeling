import random
import numpy as np
import pickle
from collections import namedtuple


# A named tuple representing a single state transition.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Replay memory class
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        ''' Saves the transition. '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save_memory(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.memory, f)
            f.close()

    def load(self, file_path):
        self.memory = pickle.load(open(file_path, 'rb'))
        self.position = len(self.memory) % self.capacity

    def __len__(self):
        return len(self.memory)


class RecurrentMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, episode):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, time_step):
        sampled_episodes = random.sample(self.memory, batch_size)
        batch = []

        for episode in sampled_episodes:
            # Ignore episodes that have not enough data points
            count = 0
            while episode is None or len(episode) <= time_step + 1:
                episode = random.sample(self.memory, 1)
                count += 1
                if count >= 200:
                    return None

            point = np.random.randint(0, len(episode) - 1 - time_step)
            batch.append(episode[point:point + time_step])

        return batch

    def save_memory(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.memory, f)
            f.close()

    def load(self, file_path):
        self.memory = pickle.load(open(file_path, 'rb'))
        self.position = len(self.memory) % self.capacity

    def __len__(self):
        return len(self.memory)

class ExpertMemory(object):
    def __init__(self, pkl_file):
        pkl = pickle.load(open(pkl_file, 'rb'))
        self.memory = pkl['traj']

    def sample(self, batch_size, time_step):
        sampled_episodes = random.sample(self.memory, batch_size)
        batch = []
        batch_label = []

        for episode in sampled_episodes:
            # Ignore episodes that have not enough data points
            count = 0
            while episode is None or len(episode) <= time_step + 1:
                episode = random.sample(self.memory, 1)
                count += 1
                if count >= 200:
                    return None

            point = np.random.randint(0, len(episode) - 1 - time_step)
            observation = []
            label = []
            for i in range(time_step):
                item = episode[i]
                observation.append([item['tactile'], item['image']])
                label.append([item['action']])
            batch.append(observation)
            batch_label.append(label)
        return batch, np.array(batch_label)