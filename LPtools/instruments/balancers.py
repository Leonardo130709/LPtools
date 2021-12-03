from abc import ABC, abstractmethod
from typing import List
from .instruments import Position


class BaseBalancer(ABC):
    def __init__(self, positions: List[Position]):
        self.positions = positions

    @abstractmethod
    def rebalance(self, state: 'MarketState') -> float:
        pass


class Holder(BaseBalancer):

    def rebalance(self, state):
        pass
        return 0


class Hedger(BaseBalancer):
    def __init__(self, positions, rebalance_period):
        self.positions = positions
        self.period = rebalance_period
        self._t = 0

    def rebalance(self, state):
        costs = 0
        if self._t == 0:
            pass
        return costs
