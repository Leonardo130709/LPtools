from .clients import Client
import pandas as pd
from copy import copy, deepcopy


class Adapter:
    def __init__(self,
                 pool_id: str,
                 symbol: str,
                 zcurve: pd.DataFrame = None,
                 base_interval='1D'
                 ):
        self.client = Client()
        self.pool_id = pool_id
        self.symbol = symbol
        self.zcurve = zcurve
        self.base_interval = base_interval

    def get_rates(self, limit):
        limit = min(limit, 1000)
        df = self.client.fundingRates(symbol=self.symbol, limit=limit)
        index = pd.date_range(df.index[0], df.index[-1], freq=self.base_interval).floor(self.base_interval)
        df = df.reindex(index, fill_value=0).fundingRate
        return df.to_frame()

    def get_kleins(self, limit):
        df = self.client.kleins(symbol=self.symbol, limit=limit, interval=self.base_interval)
        return df[['Open', 'Volume']]

    def get_pool(self, limit):
        if self.base_interval == '1d':
            df = self.client.poolDayData(self.pool_id, limit=limit)
        elif self.base_interval == '1h':
            df = self.client.poolHourData(self.pool_id, limit=limit)
        else:
            raise NotImplementedError
        # df['relative_price'] = df.token0Price / df.token1Price
        return df[::-1]

    def get_mark_prices(self, limit):
        df = self.client.markPrice(symbol=self.symbol, limit=limit, interval=self.base_interval)
        return df.Open.rename('mark').to_frame()

    def __call__(self, limit):
        # if self.zcurve is not None:
        #     limit = limit + 2  # since it is not obtained from the API thus has T+2 lag

        rates = self.get_rates(limit // 8 if self.base_interval == '1h' else limit*3)  # futures paying thrice a day
        kleins = self.get_kleins(limit)
        pool = self.get_pool(limit)
        mark = self.get_mark_prices(limit)

        return self.merge(kleins, rates, pool, mark, self.zcurve)

    @staticmethod
    def merge(*args):
        state = args[0].copy()
        #indicies = state.index
        for df in args[1:]:
            if df is not None:
                #indicies = indicies.union(df.index)
                #df = df.reindex(indicies, method='nearest')
                state = state.merge(df, left_index=True, right_index=True)
        return state


class Runner:
    def __init__(self, data, portfolios):
        self._data = data
        self._portfolios = deepcopy(portfolios)
        self.results = []

    def _run(self, portfolio):
        runner = self._data.itertuples()
        _ = next(runner)  # already initialized
        portfolio.rollout(runner)
        return portfolio.logger

    def run(self):
        for portfolio in self._portfolios:
            self.results.append(self._run(portfolio))

    def logs(self, idx):
        return self._portfolios[idx].summary[1]

    @property
    def summary(self):
        df = pd.DataFrame()
        index = []
        for i, portfolio in enumerate(self._portfolios):
            df = df.append(portfolio.summary[0])
            index.append(f'portfolio {i}')
        df.index = index
        return df[['total_value', 'discounted_value', 'min_value']]


