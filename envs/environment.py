from abc import ABC
from enum import Enum

import numpy as np


class Environment(ABC):

    def __init__(self, actions, states, terminal_states=None, obstacle_states = None):
        if terminal_states is None:
            terminal_states = list()
        self.actions = actions
        self.states = states
        if terminal_states is None:
            self.terminal_states = []
        else:
            self.terminal_states = terminal_states
        if obstacle_states is None:
            self.obstacle_states = []
        else:
            self.obstacle_states = obstacle_states


    def action(self, action):
        pass


    def get_next_state(self, state, action):
        pass

