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
        self.payments = list()
        self.df = 1

    def __add__(self, v):
        self._value += v
        return self

    def value(self, state):
        return self._value

    def step(self, state, value):
        self.df *= np.exp(-state.risk_free_rate / 360. / 100.)
        self._value += value
        self.payments.append(value * self.df)

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
        return state.Open

    def step(self, state):
        return - state.fundingRate * (state.Open - state.mark)


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

        self.L = self._liquidity_from_mv(init_state) # liquidity per 1$, it's ok since payments&values are linear in L
        self.fees = fees  # fee

    def value(self, state):
        t0, t1 = self.real_reserves(state)
        return t0*state.token0Price + t1

    def step(self, state):
        sqprice = np.sqrt(state.token0Price)
        indicator = self.sqp_l < sqprice < self.sqp_u
        # it is easier to use feesUSD than fees * volumeUSD
        return state.feesUSD * indicator * \
                    (self.L / state.liquidity)  # can be replaced by the exact calculation per tick

    # def step(self, state):
    #     """
    #     Alternative fees computation based on the given formula
    #     """
    #     coef = 0.0000000576975945
    #     sqprice = np.sqrt(state.token0Price)
    #     indicator = self.sqp_l < sqprice < self.sqp_u
    #     return coef*state.tvlUSD*indicator / (???)


    def _liquidity_from_mv(self, state):
        sqprice = self.clip(state)
        var = sqprice / self.sqp_u + self.sqp_l / sqprice
        denominator = 2 * sqprice * (1 - var / 2) # this only holds if p_- < p < p_+
        return 1 / denominator
        # return 1 / 2. / sqprice # corresponds to uni v2

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
        return self.last_value * self.amount

    def rebalance(self, new_amount):
        diff = new_amount - self.amount
        self.amount = new_amount
        cash = -diff*self.last_value
        fees = -abs(cash) * self.transaction_fees
        return fees, cash

    def step(self, state):
        _ = self.value(state)
        return self.amount * self.instrument.step(state)


class Portfolio:

    def __init__(self, balancer, positions: List[Position]):
        self.cash_pool = Cash()
        self._portfolio = positions
        self.logger = defaultdict(list)
        self.balancer = balancer
        self.min_value = np.inf

    def step(self, state):
        cash_flow = 0

        for position in self._portfolio:
            cash_flow += position.step(state)

        costs, cash = self.balancer.rebalance(state)
        cash_flow += cash

        for position in self._portfolio:
            self.logger[f'{position.tag}_DV'].append(position.last_value)
            self.logger[f'{position.tag}_amount'].append(position.amount)
            self.logger[f'{position.tag}_present_value'].append(position.value(state))

        total = cash_flow + costs
        self.cash_pool.step(state, total)
        total_value = self.value(state)
        self.min_value = min(self.min_value, total_value)
        self.logger['cumulative_payments'].append(sum(self.cash_pool.payments))
        self.logger['in_cash'].append(self.cash_pool.value(state))
        self.logger['payments'].append(cash_flow)
        self.logger['transaction_costs'].append(costs)
        self.logger['total_value'].append(total_value)

        return total

    def value(self, state):
        return sum([position.value(state) for position in self._portfolio]) + self.cash_pool.value(state)

    def finalize(self, state):
        cash = 0
        for position in self._portfolio:
            val = position.value(state)  # sell position
            self.logger['summary'].append((f'{position.tag}_last_val', val))
            cash += val

        self.cash_pool.step(state, cash)
        self.min_value = min(self.min_value, self.cash_pool.value(state))
        self.logger['summary'].append(('total_value', self.cash_pool.value(state)))
        self.logger['summary'].append(('discounted_value', sum(self.cash_pool.payments)))
        self.logger['summary'].append(('min_value', self.min_value))
        annual_return = (1 + self.cash_pool.value(state))**(360 / len(self.cash_pool.payments)) - 1
        self.logger['summary'].append(('annual_return', annual_return))
        return cash

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
        return df, pd.DataFrame({k: v for k, v in self.logger.items() if k != 'summary'})

    def __len__(self):
        return len(self._portfolio)


