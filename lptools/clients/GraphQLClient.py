import requests
import pandas as pd
from string import Template
from functools import wraps


def split_request(f):
    hard_limit = 1000

    @wraps(f)
    def wrapper(*args, **kwargs):
        df = pd.DataFrame()
        limit = kwargs['limit']
        query = kwargs
        skip = 0
        while limit > 0:
            query.update(limit=min(limit, hard_limit),
                         skip=skip)
            df = df.append(f(*args, **query))
            limit -= hard_limit
            skip += hard_limit
        return df
    return wrapper


class GraphQLClient:
    uniswap_graph_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

    def _graph_request(self, query):
        return requests.post(self.uniswap_graph_url, json={"query": query})

    def pool(self, pool_id, fields=['id']):
        fields = ' '.join(fields)
        query = f'{{pool(id : "{pool_id}") {{ {fields} }}}}'

        return self._graph_request(query).json()

    @split_request
    def poolDayData(self, pool_id, fields=None, limit=10, skip=0):
        fields = fields or ('date', 'liquidity',
                            'sqrtPrice', 'tvlUSD',
                            'volumeUSD', 'feesUSD',
                            'token0Price', 'token1Price'
                            )
        tmp = Template("""
        {
          pool(id: "$pool_id") {
            poolDayData(first: $limit, orderDirection:  desc, orderBy: date, skip: $skip) {$fields}
          }
        }
        """)
        fields = ' '.join(fields)
        query = tmp.substitute(pool_id=pool_id, fields=fields, limit=limit, skip=skip)
        resp = self._graph_request(query).json()

        df = pd.DataFrame.from_dict(resp['data']['pool']['poolDayData'])
        df.index = pd.to_datetime(df.date, unit='s')
        return df.astype(float)

    @split_request
    def poolHourData(self, pool_id, fields=None, limit=10, skip=0):
        fields = fields or ('periodStartUnix', 'liquidity', 'tvlUSD', 'feesUSD',
                            'volumeUSD', 'token0Price', 'token1Price'
                            )
        tmp = Template("""
        {
          pool(id: "$pool_id") {
            poolHourData(first: $limit, orderDirection:  desc, orderBy: periodStartUnix, skip: $skip) {$fields}
          }
        }
        """)
        fields = ' '.join(fields)
        query = tmp.substitute(pool_id=pool_id, fields=fields, limit=limit, skip=skip)
        resp = self._graph_request(query).json()

        df = pd.DataFrame.from_dict(resp['data']['pool']['poolHourData'])
        df.index = pd.to_datetime(df.periodStartUnix, unit='s')
        return df.astype(float)
