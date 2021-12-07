import requests
import os
import hmac
import pandas as pd
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print('Skipping dotenv')


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

    def kleins(self, symbol, interval='1d', limit=500):
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                   'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume',
                   'Ignore']
        data = self._request('/fapi/v1/klines', {'symbol': symbol, 'interval': interval, 'limit': limit}, False).json()
        df = pd.DataFrame(data, columns=columns)
        df.index = pd.to_datetime(df['Open time'], unit='ms')
        return df.astype(float)

    def aggTrades(self, symbol, limit=500):
        columns = ('Aggregated tradeId', 'Price', 'Quantity',
                   'First tradeId', 'Last tradeId', 'Timestamp', 'IsMaker')
        data = self._request('/fapi/v1/aggTrades', {'symbol': symbol, 'limit': limit}, False).json()
        df = pd.DataFrame.from_dict(data)
        df.columns = columns
        df.index = pd.to_datetime(df['Timestamp'], unit='ms')
        df = df.astype({'Price': float, 'Quantity': float})
        return df

    def fundingRates(self, symbol, limit=500):
        columns = ('Symbol', 'fundingTime', 'fundingRate')
        data = self._request('/fapi/v1/fundingRate', {'symbol': symbol, 'limit': limit}, False).json()
        df = pd.DataFrame.from_dict(data)
        df.index = pd.to_datetime(df.fundingTime, unit='ms')
        df.columns = columns
        df.fundingRate = df.fundingRate.astype(float)
        return df

    def markPrice(self, symbol, interval='1d', limit=500):
        resp = self._request('/fapi/v1/markPriceKlines',
                             {'symbol': symbol, 'interval': interval, 'limit': limit}).json()
        columns = ('Open time', 'Open', 'High', 'Low', 'Close', 'Ignore1', 'Close time', 'Ignore2',
                   'Number of trades', 'Ignore3', 'Ignore4',
                   'Ignore5')
        df = pd.DataFrame(resp, columns=columns)
        df.index = pd.to_datetime(df['Open time'], unit='ms')
        return df
