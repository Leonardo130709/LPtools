import requests
import pandas as pd
from string import Template


class GraphQLClient:
    uniswap_graph_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

    def _graph_request(self, query):
        return requests.post(self.uniswap_graph_url, json={"query": query})

    def pool(self, pool_id, fields=['id']):
        fields = ' '.join(fields)
        query = f'{{pool(id : "{pool_id}") {{ {fields} }}}}'

        return self._graph_request(query).json()

    def poolDayData(self, pool_id, fields=None, limit=10):
        fields = fields or ('date', 'liquidity',
                            'sqrtPrice', 'tvlUSD', 'feesUSD',
                            'volumeUSD', 'feesUSD',
                            'token0Price', 'token1Price'
                            )
        tmp = Template("""
        {
          pool(id: "$pool_id") {
            poolDayData(first: $limit, orderDirection:  desc, orderBy: date) {$fields}
          }
        }
        """)
        fields = ' '.join(fields)
        query = tmp.substitute(pool_id=pool_id, fields=fields, limit=limit)
        resp = self._graph_request(query).json()

        df = pd.DataFrame.from_dict(resp['data']['pool']['poolDayData'])
        df.index = pd.to_datetime(df.date, unit='s')
        return df.astype(float)
