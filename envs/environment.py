from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class Environment(ABC):

    def __init__(self, actions: list, states: list, terminal_states: list = None, obstacle_states: list = None):
        # Initialize actions and states sets
        self.actions: list = actions
        self.states: list = states

        # Initialize terminal states
        if terminal_states is None:
            self.terminal_states = []
        else:
            self.terminal_states = terminal_states

        # Initialize obstacle states
        if obstacle_states is None:
            self.obstacle_states = []
        else:
            self.obstacle_states = obstacle_states

        # Initialize state
        self.state = None

        # Initialize state-transition probabilities
        self.p = np.zeros((len(self.states), len(self.states), len(self.actions)))

    @abstractmethod
    def action(self, action):
        pass

    @abstractmethod
    def get_next_state(self, state, action):
        pass

    @abstractmethod
    def get_next_transitions(self, state, action) -> tuple:
        pass
