"""Offline graph loading; downloads are explicit and never happen on import."""

from .real_graphs import GraphDataset, LoadStats, load_graph
from .sources import DatasetSource, available_datasets, get_source

__all__ = [
    "DatasetSource", "GraphDataset", "LoadStats", "available_datasets",
    "get_source", "load_graph",
]