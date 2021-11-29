from .clients import Client
import pandas as pd


class Adapter:
    def __init__(self,
                 pool_id: str,
                 symbol: str,
                 zcurve: pd.DataFrame = None
                 ):
        self.client = Client()
        self.pool_id = pool_id
        self.symbol = symbol
        self.zcurve = zcurve

    def get_rates(self, days):
        df = self.client.fundingRates(self.symbol, days)
        df = df.resample('1D').sum().fundingRate
        return df.to_frame()

    def get_kleins(self, days):
        df = self.client.kleins(self.symbol, limit=days)
        return df[['Open', 'Volume']]

    def get_pool(self, days):
        df = self.client.poolDayData(self.pool_id, limit=days)
        return df[::-1]

    def get_mark_prices(self, days):
        df = self.client.markPrice(self.symbol, limit=days)
        return df.Open.rename('mark').to_frame()

    def __call__(self, days):
        if self.zcurve is not None:
            days = days + 2  # since it is not obtained from the API thus has T+2 lag

        rates = self.get_rates(3 * days)  # futures paying thrice a day
        kleins = self.get_kleins(days)
        pool = self.get_pool(days)
        mark = self.get_mark_prices(days)

        return self.merge(rates, kleins, pool, mark, self.zcurve)

    @staticmethod
    def merge(*args):
        state = args[0].copy()
        for df in args[1:]:
            if df is not None:
                state = state.merge(df, left_index=True, right_index=True)
        return state
