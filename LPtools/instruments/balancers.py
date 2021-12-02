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
