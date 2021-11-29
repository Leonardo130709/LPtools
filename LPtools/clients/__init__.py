from .BinanceClient import BinanceClient
from .GraphQLClient import GraphQLClient


class Client(BinanceClient, GraphQLClient):
    def __init__(self, *args, **kwargs):
        super(BinanceClient).__init__(*args, **kwargs)
