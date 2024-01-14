from abc import ABC
from enum import Enum

import numpy as np


class Environment(ABC):

    def __init__(self, actions, states):
        self.actions = actions
        self.states = states

    def action(self, action):
        pass

    def get_next_state(self, state, action):
        pass

