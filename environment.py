from abc import ABC
from enum import Enum


class Environment(ABC):

    class AgentActions(Enum):
        pass

    def action(self):
        pass

    def get_next_state(self):
        pass

