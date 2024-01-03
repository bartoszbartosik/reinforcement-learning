from abc import ABC
from enum import Enum


class Environment(ABC):

    class AgentActions(Enum):
        pass

    def action(self, action: AgentActions):
        pass

    def get_next_state(self, state, action):
        pass

