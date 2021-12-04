import numpy as np
from abc import ABC, abstractmethod
from typing import List
from collections import defaultdict
from dataclasses import dataclass
import pandas as pd


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
    def __init__(self, initial_value: float = 0):
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

    def __init__(self, init_state,
                 price_lower,
                 price_upper,
                 fees=.03,
                 ):

        assert price_lower < price_upper
        # sqp stands for square root of the price
        self.sqp_u = np.sqrt(price_upper)
        self.sqp_l = np.sqrt(price_lower)

        self.L = self._liquidity_from_mv(init_state) # liquidity per 1$, it's ok since payments/values are linear in L
        self.fees = fees  # fee

    def value(self, state):
        # market relative price of tokens \sim pool_value -- assumption of the ultimate arbitrage
        t0, t1 = self.virtual_reserves(state)
        return t0*state.token0Price + t1

    def step(self, state):
        sqprice = np.sqrt(state.token0Price)
        indicator = self.sqp_l < sqprice < self.sqp_u
        return indicator * state.feesUSD * \
                    (self.L / state.liquidity)  # can be replaced by the exact calculation per tick

    def _liquidity_from_mv(self, state):
        sqprice = self.clip(state)
        # # sqprice = np.sqrt(state.token0Price)
        # var = sqprice / self.sqp_u + self.sqp_l / sqprice
        # rel = 4*(1 - self.sqp_l / self.sqp_u)
        # coef = var + np.sqrt(var**2 + rel)
        # coef /= rel
        #
        # return coef / sqprice
        return 1 / 2. / sqprice

    def clip(self, state):
        sqprice = np.sqrt(state.token0Price)
        return np.clip(sqprice, self.sqp_l, self.sqp_u)

    def virtual_reserves(self, state):
        sqprice = self.clip(state)
        return self.L / sqprice, self.L * sqprice

    def real_reserves(self, state):
        t0, t1 = self.virtual_reserves(state)
        return t0 - self.L / self.sqp_u, t1 - self.L * self.sqp_l


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

    def __init__(self, balancer, positions: List[Position], cash: Cash = Cash()):
        self.cash_pool = cash
        self._portfolio = positions
        self.logger = defaultdict(list)
        self.balancer = balancer

    def step(self, state):
        cash_flow = 0

        for position in self._portfolio:
            self.logger[f'{position.tag}_value'].append(position.instrument.value(state))
            self.logger[f'{position.tag}_amount'].append(position.amount)

            cash_flow += position.step(state)

        costs = self.balancer.rebalance(state)

        self.logger['payments'].append(cash_flow)
        self.logger['transaction_costs'].append(costs)

        self.cash_pool.step(state, cash_flow + costs)

    def finalize(self, state):
        cash = 0
        for position in self._portfolio:
            val = position.value(state)
            self.logger['summary'].append((f'{position.tag}_last_val', val))
            cash += val

        self.cash_pool.step(state, cash)
        self.logger['summary'].append(('total_value', self.cash_pool.value(state)))
        self.logger['summary'].append(('discounted_value', sum(self.cash_pool.payments)))

    def rollout(self, runner):
        last_state = None
        for state in runner:
            last_state = state
            self.step(state)
        self.finalize(last_state)

    @property
    def summary(self):
        for k, v in self.logger.items():
            if k == 'summary':
                df = pd.DataFrame.from_records(self.logger['summary']).set_index(0).T
                #[print(t, x) for t, x in v]
        return df, pd.DataFrame({k: v for k, v in self.logger.items() if k != 'summary'})


