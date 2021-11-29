import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from collections import defaultdict
from dataclasses import dataclass


class Instrument(ABC):

    @abstractmethod
    def value(self, state: 'MarketState') -> float:
        "'Fair' value of the instrument estimated from the current MarketState"
        pass

    @abstractmethod
    def step(self, state: 'MarketState') -> float:
        """
        Defenition of dynamics: payments of the instrument per timestep and changes in market value
        returns instrument value, cash payment on current timestep, bool: is expired/closed
        """
        pass


class Cash:
    def __init__(self, initial_value: float):
        self.value = initial_value
        self.payments = []
        self.df = 1

    def __add__(self, v):
        self.value += v
        return self

    def value(self, state):
        return self.value

    def step(self, state, value):
        self.df *= np.exp(-state.risk_free_rate / 360.)
        self.value += value
        self.payments.append(value * state.risk_free_rate)


class Bond(Instrument):
    """
    ZeroCoupon Bond
    """

    def __init__(self, initial_value, rate):
        self.value = initial_value
        self.r = rate

    def value(self, state):
        return self.value

    def step(self, state):
        self.value *= (1 + self.r)


class Perpet(Instrument):

    def value(self, state):
        return state.token0Price

    def step(self, state):
        return state.fundingRate * (state.token0Price - state.mark)


class UniPool(Instrument):
    tick_base = 1.0001

    def __init__(self, L,
                 tick_lower,
                 tick_upper,
                 phi=.03,
                 meta: dict = None):
        assert tick_lower < tick_upper
        self.tl = tick_lower
        self.tu = tick_upper

        # sqp stands for square root of the price
        self.sqp_u = UniPool.tick_to_sprice(self.tu)
        self.sqp_l = UniPool.tick_to_sprice(self.tl)

        self.L = L
        self.phi = phi  # fee
        self._meta = meta

    def value(self, state: dict) -> float:
        # market relative price of tokens \sim pool_value -- assumumpition of the ultimate arbitrage
        p0 = state.token0Price;
        p1 = state.token1Price
        sqp = np.sqrt(p0 / p1)
        t0, t1 = self._real_reserves(sqp)
        return t0 * p1 + t1 * p1

    def step(self, state):
        return state.feesUSD * (self.L / state.liquidity)  # can be replaced by the exact calculation

    @classmethod
    def sprice_to_tick(cls, sprice):
        return np.floor(2 * np.log(sprice) / np.log(cls.tick_base))

    @classmethod
    def tick_to_sprice(cls, tick):
        return np.floor(cls.tick_base ** (tick / 2.))

    def _virtual_reserves(self, sprice):
        # returns: token0_amount, token1_amount
        sprice = np.clip(sprice, self.sqp_l + 1e-8, self.sqp_u)
        token0 = self.L * sprice
        token1 = self.L / sprice
        return token0, token1

    def _real_reserves(self, sprice):
        token0, token1 = self._virtual_reserves(sprice)
        return token0 - self.L / self.sqp_u, token1 - self.L * self.sqp_l

    @staticmethod
    def liquidity_from_mv(state, mv):
        sqprice = np.sqrt(state.token0Price / state.token1Price)
        return mv / 2 / sqprice


@dataclass
class Position:
    tag: str
    instrument: Instrument
    amount: float = 0
    transaction_fees: float = .01
    last_value: float = 0

    def value(self, state):
        self.last_value = self.instrument.value(state)
        return self.amount * self.last_value

    def rebalance(self, new_amount):
        diff, self.amount = new_amount - self.amount, new_amount
        fees = -abs(diff * self.last_value * self.transaction_fees)
        return fees

    def step(self, state):
        return self.amount * self.instrument.step(state)


class Portfolio:

    def __init__(self, balancer, cash: Cash, positions: List[Position]):
        self.cash_pool = cash
        self._portfolio = positions
        self.logger = defaultdict(list)
        self.balancer = balancer

    def step(self, state):
        cash_flow = 0

        for position in self._portfolio:
            self.logger[position.tag].append(position.instrument.value(state))
            cash_flow += position.step(state)

        costs = self.balancer.rebalance(state)

        self.logger['payments'].append(cash_flow)
        self.logger['transaction_costs'].append(costs)

        self.cash_pool.step(state, cash_flow + costs)

    def __len__(self):
        return self._portfolio.__len__()


