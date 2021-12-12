from abc import ABC, abstractmethod
from typing import List
from .instruments import Position, Perpetual, UniPool, Bond
import numpy as np
from copy import deepcopy


class BaseBalancer(ABC):
    _types = ()

    def __init__(self, positions: List[Position], initial_amounts=None):
        self._validate_inputs(positions, initial_amounts)
        self.positions = positions
        self._init = True
        self.initial_amounts = initial_amounts

    @abstractmethod
    def rebalance(self, state: 'MarketState') -> float:
        costs = 0
        if self._init:
            for p, a in zip(self.positions, self.initial_amounts):
                costs += p.rebalance(a)
            self._init = False
        return costs

    @classmethod
    def _validate_inputs(cls, positions, initial_amounts):
        check = map(lambda tup: isinstance(tup[0].instrument, tup[1]), zip(positions, cls._types))
        assert all(check) and len(positions) == len(initial_amounts), "Wrong positions for the strategy"


class Holder(BaseBalancer):
    def __init__(self, positions, initial_amounts=None):
        super().__init__(positions, initial_amounts)

    def rebalance(self, state):
        return super().rebalance(state)


class PerpetHedger(BaseBalancer):
    _types = (UniPool, Perpetual)

    def __init__(self, postitions, initial_amounts, rebalancing_interval):
        super().__init__(postitions, initial_amounts=initial_amounts)
        self.pool, self.perpet, *self.unmanaged = postitions
        self.period = rebalancing_interval
        self._t = 1
        self.L, self.sqp_l, self.sqp_u = \
            map(lambda atr: getattr(self.pool.instrument, atr), ('L', 'sqp_l', 'sqp_u'))

    def rebalance(self, state):
        costs = super().rebalance(state)
        sprice = np.sqrt(state.token0Price)
        if self._t == 0:
            if self.sqp_l < sprice < self.sqp_u:
                new_amount = - self.L * (1 / sprice - 1 / self.sqp_u)
            else:
                new_amount = 0
            costs += self.perpet.rebalance(new_amount)

        self._t = (self._t + 1) % self.period

        return costs
