from abc import ABC, abstractmethod
from typing import List
from .instruments import Position, UniPool


class BaseBalancer(ABC):
    def __init__(self, positions: List[Position]):
        self.positions = positions

    @abstractmethod
    def rebalance(self, state):
        pass


class PassivePoolHolder(BaseBalancer):
    def __init__(self, positions):
        assert positions.__len__() == 1 and isinstance(positions[0].instrument, UniPool)
        super().__init__(positions)

    def rebalance(self, state):
        return 0
