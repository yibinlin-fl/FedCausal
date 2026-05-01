"""Federated training and aggregation components."""

from federated.client import Client
from federated.server import Server
from federated.aggregation import aggregate_mask_and_prototypes

__all__ = ["Client", "Server", "aggregate_mask_and_prototypes"]
