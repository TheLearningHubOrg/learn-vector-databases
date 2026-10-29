"""
Pedagogical Roadmap: 7-Stage Vector Databases Masterclass.

Tracks curriculum progress across Vector Geometry, ANN Indexing, Quantization,
Engine Architecture, Filtering, Distributed Sharding, and Staff Engineering.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class StageStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


@dataclass
class Topic:
    id: str
    title: str
    summary: str
    completed: bool = False


@dataclass
class Stage:
    stage_number: int
    title: str
    description: str
    topics: List[Topic]
    status: StageStatus = StageStatus.NOT_STARTED

    @property
    def progress_percentage(self) -> float:
        if not self.topics:
            return 0.0
        completed = sum(1 for t in self.topics if t.completed)
        return (completed / len(self.topics)) * 100.0


class VectorDatabasesRoadmap:
    """Manages progress and milestones for the 7-Stage Vector Databases curriculum."""

    def __init__(self) -> None:
        self.stages: Dict[int, Stage] = {
            1: Stage(
                stage_number=1,
                title="Vector Spaces & Distance Metrics",
                description="Mathematical foundations in R^D, curse of dimensionality, Euclidean, Cosine, Dot Product, Manhattan, and L2 normalization.",
                topics=[
                    Topic("1.1", "Vector Databases & High-Dimensional R^D Spaces", "Geometry of high-dimensional embeddings and failure of spatial B/R-Trees.", True),
                    Topic("1.2", "Curse of Dimensionality", "Distance concentration and hypersphere volume concentration at boundaries.", True),
                    Topic("1.3", "Distance Metrics", "Euclidean (L2), Manhattan (L1), Cosine Distance, Dot Product, and Hamming distance.", True),
                    Topic("1.4", "Metric Invariance & L2 Normalization", "Equivalence of Cosine Similarity and Dot Product on unit spheres.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            2: Stage(
                stage_number=2,
                title="Approximate Nearest Neighbor (ANN) Indexing",
                description="Exact kNN brute-force vs IVF inverted lists and Hierarchical Navigable Small World (HNSW) skip-list graphs.",
                topics=[
                    Topic("2.1", "Exact kNN Brute-Force", "O(N*D) baseline search latency and memory bandwidth limitations.", True),
                    Topic("2.2", "Inverted File (IVF) Indexes", "Centroid k-means clustering, Voronoi cell partitioning, and nprobe tuning.", True),
                    Topic("2.3", "HNSW Graph Architecture", "Multi-layer skip-list graph, greedy traversal, and beam search.", True),
                    Topic("2.4", "HNSW Hyperparameters", "Tuning M, M0, efConstruction, and efSearch for recall vs latency tradeoffs.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            3: Stage(
                stage_number=3,
                title="Quantization & Memory Compression",
                description="Solving vector RAM crisis via Scalar Quantization (SQ8), Product Quantization (PQ), and 1-bit Binary Quantization (BQ).",
                topics=[
                    Topic("3.1", "High-Dimensional Vector RAM Sizing", "Calculating 100M 1536-dim vector FP32 footprints (600GB+ RAM).", True),
                    Topic("3.2", "Scalar Quantization (SQ8)", "Uniform affine scaling from FP32 to INT8 with min/max calibration.", True),
                    Topic("3.3", "Product Quantization (PQ)", "Sub-vector codebook generation, centroid training, and asymmetric distance tables (ADC).", True),
                    Topic("3.4", "1-Bit Binary Quantization (BQ)", "Threshold sign bit-packing, Hamming distance, and XOR/POPCNT acceleration.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            4: Stage(
                stage_number=4,
                title="Database Landscape & Engine Architectures",
                description="Dedicated engines (Qdrant, Pinecone, Milvus, Weaviate) vs relational extensions (PostgreSQL pgvector, Redis VSS).",
                topics=[
                    Topic("4.1", "Specialized vs Extended Engines", "Architectural trade-offs of Qdrant/Pinecone vs PostgreSQL pgvector/Redis.", True),
                    Topic("4.2", "Storage Engine Layouts", "In-memory, MMAP disk-backed indexes, and TurboQuant SSD segment streaming.", True),
                    Topic("4.3", "Index Build Performance", "Thread pooling, parallel graph construction, and vector ingestion throughput.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            5: Stage(
                stage_number=5,
                title="Metadata Filtering & Multi-Tenancy",
                description="Single-stage payload-filtered graph traversal to eliminate post-filtering recall collapse, plus isolation patterns.",
                topics=[
                    Topic("5.1", "Pre-Filtering vs Post-Filtering Recall Collapse", "Why naive post-filtering drops recall to near zero.", True),
                    Topic("5.2", "Single-Stage Payload-Filtered HNSW", "Evaluating boolean/range payload predicates during graph candidate traversal.", True),
                    Topic("5.3", "Multi-Tenancy Isolation", "Namespace partitioning, collection sharding, and tenant-keyed routing.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            6: Stage(
                stage_number=6,
                title="Distributed Scaling & Production Sharding",
                description="Coordinator topologies, Raft consensus shard replication, WAL durability, segment immutability, and compaction.",
                topics=[
                    Topic("6.1", "Distributed Topologies", "Query coordinators, scatter-gather scatterers, and replica sets.", True),
                    Topic("6.2", "Write Durability & Compaction", "Write-Ahead Logging (WAL), immutable segment merging, and soft deletes.", True),
                    Topic("6.3", "High Availability & Rebalancing", "Consistent hashing, dynamic shard splitting, and cluster failover.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            7: Stage(
                stage_number=7,
                title="Staff Architect Blueprints & Interview Mastery",
                description="End-to-end multi-billion scale system designs, operational runbooks, and staff interview question sets.",
                topics=[
                    Topic("7.1", "Multi-Billion Scale Design", "Designing a sub-10ms latency search cluster over 10 billion vectors.", True),
                    Topic("7.2", "Operational Runbooks", "OOM recovery, index rebuild zero-downtime cutover, and cache warmup.", True),
                    Topic("7.3", "50 Staff-Level Q&As", "Exhaustive mastery of indexing, quantization, hardware SIMD, and scaling.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
        }

    def get_stage(self, stage_number: int) -> Optional[Stage]:
        return self.stages.get(stage_number)

    def total_progress(self) -> float:
        total_topics = sum(len(s.topics) for s in self.stages.values())
        if total_topics == 0:
            return 0.0
        completed = sum(sum(1 for t in s.topics if t.completed) for s in self.stages.values())
        return (completed / total_topics) * 100.0

    def get_summary(self) -> Dict[str, object]:
        return {
            "total_stages": len(self.stages),
            "completed_stages": sum(1 for s in self.stages.values() if s.status == StageStatus.COMPLETED),
            "overall_progress_pct": round(self.total_progress(), 2),
            "stages": [
                {
                    "stage": s.stage_number,
                    "title": s.title,
                    "status": s.status.value,
                    "progress_pct": round(s.progress_percentage, 1),
                    "topics_count": len(s.topics),
                }
                for s in self.stages.values()
            ],
        }
