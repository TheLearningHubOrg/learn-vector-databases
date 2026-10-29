"""
learn_vector_databases: Production-Grade Vector Database Systems Engineering Engine.

Provides exact kNN, Inverted File (IVF) indexes, Hierarchical Navigable Small World (HNSW) graphs,
Scalar Quantization (SQ8), 1-bit Binary Quantization (BQ), single-stage payload filtering,
and pedagogical curriculum progress tracking.
"""

from .vectordb import (
    VectorItem,
    DistanceMetric,
    calculate_distance,
    l2_normalize,
    ExactKNNIndex,
    IVFIndex,
    HNSWIndex,
    ScalarQuantizer,
    BinaryQuantizer,
    MultiTenantVectorStore,
)
from .roadmap import VectorDatabasesRoadmap

__all__ = [
    "VectorItem",
    "DistanceMetric",
    "calculate_distance",
    "l2_normalize",
    "ExactKNNIndex",
    "IVFIndex",
    "HNSWIndex",
    "ScalarQuantizer",
    "BinaryQuantizer",
    "MultiTenantVectorStore",
    "VectorDatabasesRoadmap",
]
