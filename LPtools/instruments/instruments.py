import numpy as np
from abc import ABC, abstractmethod
from typing import List
from collections import defaultdict
from dataclasses import dataclass


class Instrument(ABC):

    @abstractmethod
    def value(self, state: 'MarketState') -> float:
        """
        'Fair' value of the instrument estimated from the current MarketState
        """
        pass

    @abstractmethod
    def step(self, state: 'MarketState') -> float:
        """
        Definition of dynamics: payments of the instrument per timestamp and changes in market value
        """
        pass


class Cash:
    def __init__(self, initial_value: float):
        self._value = initial_value
        self.payments = []
        self.df = 1

    def __add__(self, v):
        self._value += v
        return self

    def value(self, state):
        return self._value

    def step(self, state, value):
        self.df *= np.exp(-state.risk_free_rate / 360. / 100.)
        self._value += value
        self.payments.append(value * self.df)  # here we can get payments

    def __repr__(self):
        return self._value.__repr__()

    @property
    def discounted_payments(self):
        return np.sum(self.payments)


class Bond(Instrument):
    """
    ZeroCoupon Bond
    """

    def __init__(self, init_state):
        self.rate = init_state.risk_free_rate / 100. / 360.  # day rate
        self._value = 1

    def value(self, state):
        return self._value

    def step(self, state):
        self._value *= (1 + self.rate)
        return 0  # no payments except on expiration


class Perpetual(Instrument):

    def value(self, state):
        return state.token0Price

    def step(self, state):
        return - state.fundingRate * (state.token0Price - state.mark)


class UniPool(Instrument):
    tick_base = 1.0001

    def __init__(self, liquidity,
                 price_lower,
                 price_upper,
                 fees=.03,
                 meta: dict = None):

        assert price_lower < price_upper
        # sqp stands for square root of the price
        self.sqp_u = np.sqrt(price_upper)
        self.sqp_l = np.sqrt(price_lower)

        self.tl = UniPool.sprice_to_tick(self.sqp_l)
        self.tu = UniPool.sprice_to_tick(self.sqp_u)

        self.L = liquidity
        # self.L = self.liquidity_from_mv(init_state, 1)
        self.fees = fees  # fee
        self._meta = meta

    def value(self, state: dict) -> float:
        # market relative price of tokens \sim pool_value -- assumumpition of the ultimate arbitrage
        p0 = state.token0Price
        sqp = np.sqrt(p0)
        t0, t1 = self._real_reserves(sqp)
        return t0 * p0 + t1 * 1

    def step(self, state):
        sqprice = np.sqrt(state.token0Price)
        # factor = 1 - .5*(sqprice / self.sqp_u + self.sqp_l / sqprice)
        # net_liquidity = self.L * factor
        indicator = self.sqp_l < sqprice < self.sqp_u
        return indicator * state.feesUSD * \
                    (self.L / state.liquidity)  # can be replaced by the exact calculation per tick

    @classmethod
    def sprice_to_tick(cls, sprice):
        return np.floor(2 * np.log(sprice) / np.log(cls.tick_base))

    @classmethod
    def tick_to_sprice(cls, tick):
        return np.floor(cls.tick_base ** (tick / 2.))

    def _virtual_reserves(self, sprice):
        # returns: token0_amount, token1_amount
        sprice = np.clip(sprice, self.sqp_l, self.sqp_u)
        token0 = self.L / sprice
        token1 = self.L * sprice
        return token0, token1

    def _real_reserves(self, sprice):
        token0, token1 = self._virtual_reserves(sprice)
        return token0 - self.L / self.sqp_u, token1 - self.L * self.sqp_l

    @staticmethod
    def liquidity_from_mv(state, mv):
        sqprice = np.sqrt(state.token0Price)
        # sqprice = np.clip(sqprice, self.sqp_l, self.sqp_u)
        return mv / 2. / sqprice


@dataclass
class Position:
    tag: str
    instrument: Instrument
    amount: float = 0
    transaction_fees: float = .01
    last_value: float = 1

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

    def finalize(self, state):
        cash = 0
        for position in self._portfolio:
            val = position.value(state)
            self.logger[f'{position.tag}_last_val'] = [val]
            cash += val

        self.cash_pool.step(state, cash)
        self.logger['total_value'] = [self.cash_pool.value(state)]

    def rollout(self, runner):
        last_state = None
        for state in runner:
            last_state = state
            self.step(state)
        self.finalize(last_state)

    @property
    def summary(self):
        for k, v in self.logger.items():
            if len(v) == 1:
                print(f'{k} = {v.pop()}')
        try:
            import pandas as pd
            return pd.DataFrame({k: v for k, v in self.logger.items() if len(v) == len(self.logger['payments'])})
        except ImportError:
            print('Omitting usage of pandas')
            return self.logger


