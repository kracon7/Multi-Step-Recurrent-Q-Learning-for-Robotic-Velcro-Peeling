import numpy as np


class Observation():

    def __init__(self, position, shear, force):
        self.position = np.array(position)
        self.shear = np.array(shear)
        self.force = np.array(force)

    def update(self, position, shear, force):
        self.position = position
        self.shear = shear
        self.force = force

    def get_state(self):
        return np.concatenate((self.position, self.shear, self.force), axis=0)

    def get_current_position(self):
        return self.position[0:3]

class TactileObs(object):
    """docstring for MultimodalObs"""
    def __init__(self, position, force):
        self.position = np.array(position)
        self.force = np.array(force)

    def update(self, position, force):
        self.position = position
        self.force = force

    def get_state(self):
        return np.concatenate((self.position, self.force), axis=0)


# Class that describes our action space.
# We can get an action by index
class ActionSpace():

    def __init__(self, dp, df):
        self.actions = {'left': {'type': 'position', 'delta': [-dp, 0, 0, 0]},
                        'right': {'type': 'position', 'delta': [dp, 0, 0, 0]},
                        'forward': {'type': 'position', 'delta': [0, dp, 0, 0]},
                        'backward': {'type': 'position', 'delta': [0, -dp, 0, 0]},
                        'up': {'type': 'position', 'delta': [0, 0, dp, 0]},
                        'down': {'type': 'position', 'delta': [0, 0, -dp, 0]},
                        'open': {'type': 'force', 'delta': [0, 0, 0, df]},
                        'close': {'type': 'force', 'delta': [0, 0, 0, -df]}}

    def get_action(self, action_key):
        assert(action_key in self.actions.keys())
        return self.actions[action_key]


#===========================================#
# test utility function for openloop test

def get_action_sequence():
    # Generate the action sequence corresponding to different directions
    action_seq = [[0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1],
                  [0, 2, 2, 2, 2, 2],
                  [0, 3, 3, 3, 3, 3],
                  [0, 4, 4, 4, 4, 4],
                  [0, 1, 2, 0, 1, 2],
                  [0, 2, 3, 0, 2, 3],
                  [0, 3, 4, 0, 3, 4],
                  [0, 1, 4, 0, 1, 4]]
    key = np.random.randint(9)
    return action_seq[key]
