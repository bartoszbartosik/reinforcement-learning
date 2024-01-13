from abc import ABC
from enum import Enum


class Environment(ABC):

    def __init__(self, actions):
        self.actions = actions

    def action(self, action):
        pass

    def get_next_state(self, state, action):
        pass

