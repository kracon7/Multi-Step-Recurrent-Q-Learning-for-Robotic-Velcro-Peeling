import numpy as np
import pickle


class Normalizer():
    def __init__(self, num_inputs=21, device='cuda'):
        self.device = device
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)

    def observe(self, x):
        self.n += 1
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - last_mean)
        self.var = np.maximum(self.mean_diff / self.n, 1e-2)

    def normalize(self, inputs):
        obs_std = np.sqrt(self.var)
        return (inputs - self.mean) / obs_std

    def save_state(self, path):
        dictionary = {'n': self.n, 'mean': self.mean, 'mean_diff': self.mean_diff, 'var': self.var}
        with open(path, 'wb') as file:
            pickle.dump(dictionary, file)

    def restore_state(self, path):
        with open(path, 'rb') as file:
            dictionary = pickle.load(file)

        self.n = dictionary['n']
        self.mean = dictionary['mean']
        self.mean_diff = dictionary['mean_diff']
        self.var = dictionary['var']

class Geom_Normalizer():
    def __init__(self, num_inputs=21, device='cuda'):
        self.device = device
        self.max = 1e-4 * np.ones(num_inputs)
        self.min = -1e-4 * np.ones(num_inputs)

    def observe(self, x):
        self.max = np.maximum(x, self.max)
        self.min = np.minimum(x, self.min)

    def normalize(self, inputs):
        rang = self.max - self.min
        return (inputs - self.min) / rang

    def save_state(self, path):
        dictionary = {'max': self.max, 'min': self.min}
        with open(path, 'wb') as file:
            pickle.dump(dictionary, file)

    def restore_state(self, path):
        with open(path, 'rb') as file:
            dictionary = pickle.load(file)

        self.max = dictionary['max']
        self.min = dictionary['min']

class Multimodal_Normalizer():
    '''
    All input are numpy arrays
    '''
    def __init__(self, num_inputs=21, device='cuda'):
        self.device = device
        self.max = 1e-4 * np.ones(num_inputs)
        self.min = -1e-4 * np.ones(num_inputs)

    def observe(self, x):
        self.max = np.maximum(x, self.max)
        self.min = np.minimum(x, self.min)

    def normalize(self, inputs):
        rang = self.max - self.min
        return (inputs - self.min) / rang

    def save_state(self, path):
        dictionary = {'max': self.max, 'min': self.min}
        with open(path, 'wb') as file:
            pickle.dump(dictionary, file)

    def restore_state(self, path):
        with open(path, 'rb') as file:
            dictionary = pickle.load(file)

        self.max = dictionary['max']
        self.min = dictionary['min']