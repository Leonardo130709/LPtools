import requests
import os
import hmac
import pandas as pd
from functools import wraps
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print('Skipping dotenv')


def split_request(f):
    hard_limit = 1500

    @wraps(f)
    def wrapper(*args, **kwargs):
        df = pd.DataFrame()
        query = kwargs
        limit = kwargs['limit']
        while limit > 0:
            query['limit'] = min(limit, hard_limit)  # binance.api hard limit
            response = f(*args, **query)
            endTime = pd.Timestamp(response.index[0], unit='ms').timestamp()
            df = df.append(response)
            query.update(endTime=endTime)
            limit -= hard_limit
        return df
    return wrapper


class BinanceClient:
    api_url = 'https://fapi.binance.com'

    def __init__(self, apikey=None, secret=None):
        self.apikey = apikey or os.environ.get('binance_api_key')
        self.secret = secret or os.environ.get('binance_secret_key')
        self.headers = {'X-MBX-APIKEY': self.secret}
        self.hmac = hmac.new(self.secret.encode(), None, 'sha256')

    def _request(self, endpoint, payload, signed=False):
        query = '&'.join([f'{k}={v}' for k, v in payload.items()])
        if signed:
            h = self._signature(query)
            return requests.get(f'{self.api_url}{endpoint}?{query}&signature={h}', headers=self.headers)
        else:
            return requests.get(f'{self.api_url}{endpoint}?{query}')

    def _signature(self, msg):
        h = self.hmac.copy()
        h.update(msg.encode())
        return h.hexdigest()

    @split_request
    def kleins(self, **kwargs):
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                   'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume',
                   'Ignore']

        data = self._request('/fapi/v1/klines', kwargs, False).json()
        df = pd.DataFrame(data, columns=columns)
        df.index = pd.to_datetime(df['Open time'], unit='ms')
        return df.astype(float)

    @split_request
    def aggTrades(self, **kwargs):
        columns = ('Aggregated tradeId', 'Price', 'Quantity',
                   'First tradeId', 'Last tradeId', 'Timestamp', 'IsMaker')
        data = self._request('/fapi/v1/aggTrades', kwargs, False).json()
        df = pd.DataFrame.from_dict(data)
        df.columns = columns
        df.index = pd.to_datetime(df['Timestamp'], unit='ms')
        df = df.astype({'Price': float, 'Quantity': float})
        return df

    #@split_request max lim is already 1000
    def fundingRates(self, **kwargs):
        columns = ('Symbol', 'fundingTime', 'fundingRate')
        data = self._request('/fapi/v1/fundingRate', kwargs, False).json()
        df = pd.DataFrame.from_dict(data)
        df.index = pd.to_datetime(df.fundingTime, unit='ms')
        df.columns = columns
        df.fundingRate = df.fundingRate.astype(float)
        return df

    @split_request
    def markPrice(self, **kwargs):
        resp = self._request('/fapi/v1/markPriceKlines',
                             kwargs).json()
        columns = ('Open time', 'Open', 'High', 'Low', 'Close', 'Ignore1', 'Close time', 'Ignore2',
                   'Number of trades', 'Ignore3', 'Ignore4',
                   'Ignore5')
        df = pd.DataFrame(resp, columns=columns)
        df.index = pd.to_datetime(df['Open time'], unit='ms')
        return df


