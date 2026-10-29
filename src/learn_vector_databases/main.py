"""
Vector Databases Masterclass: CLI & Architectural Demonstrator.

Demonstrates:
1. 7-Stage Curriculum progress and completion tracking.
2. Vector distance metric evaluation (Euclidean, Cosine, Manhattan, Dot Product).
3. Exact kNN vs Inverted File (IVF) vs HNSW Graph ANN benchmarks.
4. Memory compression: FP32 vs SQ8 (4x reduction) vs BQ (32x reduction).
5. Single-stage payload filtering preventing post-filtering recall collapse.
6. Multi-tenant collection isolation.
"""

import math
import random
import time
from typing import List, Tuple

from src.learn_vector_databases.roadmap import VectorDatabasesRoadmap
from src.learn_vector_databases.vectordb import (
    BinaryQuantizer,
    DistanceMetric,
    ExactKNNIndex,
    HNSWIndex,
    IVFIndex,
    MultiTenantVectorStore,
    ScalarQuantizer,
    VectorItem,
    calculate_distance,
    l2_normalize,
)


def generate_synthetic_vectors(count: int, dimensions: int, seed: int = 42) -> List[VectorItem]:
    """Generates synthetic embedding vectors with metadata payloads."""
    rng = random.Random(seed)
    categories = ["ai_ml", "databases", "cloud_infra", "security", "distributed_systems"]
    tenants = ["acme_corp", "globex_inc", "initech_llc"]

    items: List[VectorItem] = []
    for i in range(count):
        raw_vec = [rng.gauss(0.0, 1.0) for _ in range(dimensions)]
        unit_vec = l2_normalize(raw_vec)
        category = rng.choice(categories)
        tenant = rng.choice(tenants)
        rating = round(rng.uniform(1.0, 5.0), 1)
        items.append(
            VectorItem(
                id=f"doc_{i:04d}",
                vector=unit_vec,
                payload={
                    "category": category,
                    "tenant_id": tenant,
                    "rating": rating,
                    "active": rng.random() > 0.1,
                },
            )
        )
    return items


def run_roadmap_showcase() -> None:
    roadmap = VectorDatabasesRoadmap()
    summary = roadmap.get_summary()

    print("=" * 80)
    print("VECTOR DATABASES MASTERCLASS: 7-STAGE PEDAGOGICAL ROADMAP")
    print("=" * 80)
    print(f"Curriculum Completion: {summary['overall_progress_pct']}% | Stages: {summary['completed_stages']}/{summary['total_stages']} Completed\n")

    for s in summary["stages"]:
        status_symbol = "[x]" if s["status"] == "COMPLETED" else "[ ]"
        print(f"  {status_symbol} Stage {s['stage']}: {s['title']} ({s['topics_count']} topics, {s['progress_pct']}%)")
    print("=" * 80 + "\n")


def run_metrics_demo() -> None:
    print("-" * 80)
    print("STAGE 1: HIGH-DIMENSIONAL VECTOR DISTANCE METRICS")
    print("-" * 80)
    v1 = l2_normalize([1.0, 2.0, 3.0, 4.0, 5.0])
    v2 = l2_normalize([1.2, 1.8, 3.1, 3.9, 5.2])
    v3 = l2_normalize([-1.0, -2.0, -3.0, -4.0, -5.0])

    print("Vectors in R^5:")
    print(f"  v1: {[round(x, 3) for x in v1]}")
    print(f"  v2: {[round(x, 3) for x in v2]} (Close to v1)")
    print(f"  v3: {[round(x, 3) for x in v3]} (Opposite to v1)\n")

    metrics = [
        (DistanceMetric.EUCLIDEAN, "Euclidean (L2)"),
        (DistanceMetric.COSINE, "Cosine Distance (1 - cos)"),
        (DistanceMetric.MANHATTAN, "Manhattan (L1)"),
        (DistanceMetric.DOT_PRODUCT, "Dot Product (Negative IP)"),
    ]

    for metric, label in metrics:
        d12 = calculate_distance(v1, v2, metric)
        d13 = calculate_distance(v1, v3, metric)
        print(f"  {label:<30} -> dist(v1, v2) = {d12:7.4f} | dist(v1, v3) = {d13:7.4f}")
    print()


def run_benchmark_demo() -> None:
    print("-" * 80)
    print("STAGE 2: ANN INDEX BENCHMARK: EXACT KNN vs IVF vs HNSW GRAPH")
    print("-" * 80)

    dim = 64
    count = 600
    items = generate_synthetic_vectors(count, dim, seed=123)
    query_item = items[0]
    query_vec = query_item.vector
    k = 5

    # 1. Exact kNN
    exact_idx = ExactKNNIndex(metric=DistanceMetric.COSINE)
    exact_idx.add_batch(items)
    t0 = time.perf_counter()
    exact_results = exact_idx.search(query_vec, k=k)
    t_exact = (time.perf_counter() - t0) * 1000.0
    exact_ids = set(it.id for it, _ in exact_results)

    # 2. Inverted File (IVF)
    ivf_idx = IVFIndex(nlist=16, nprobe=4, metric=DistanceMetric.COSINE)
    ivf_idx.train_and_build(items, iterations=10)
    t0 = time.perf_counter()
    ivf_results = ivf_idx.search(query_vec, k=k)
    t_ivf = (time.perf_counter() - t0) * 1000.0
    ivf_ids = set(it.id for it, _ in ivf_results)
    ivf_recall = (len(exact_ids.intersection(ivf_ids)) / k) * 100.0

    # 3. HNSW Graph
    hnsw_idx = HNSWIndex(dimensions=dim, metric=DistanceMetric.COSINE, M=16, ef_construction=64, ef_search=32)
    for it in items:
        hnsw_idx.add(it)
    t0 = time.perf_counter()
    hnsw_results = hnsw_idx.search(query_vec, k=k)
    t_hnsw = (time.perf_counter() - t0) * 1000.0
    hnsw_ids = set(it.id for it, _ in hnsw_results)
    hnsw_recall = (len(exact_ids.intersection(hnsw_ids)) / k) * 100.0

    print(f"Dataset: {count} vectors, D={dim} | Query: k={k} nearest neighbors\n")
    print(f"  {'Index Type':<16} | {'Latency':<12} | {'Recall@5':<12} | {'Top Match':<12}")
    print("  " + "-" * 56)
    print(f"  {'Exact kNN':<16} | {t_exact:7.3f} ms   | {'100.0%':<12} | {exact_results[0][0].id}")
    print(f"  {'IVF (nprobe=4)':<16} | {t_ivf:7.3f} ms   | {ivf_recall:5.1f}%       | {ivf_results[0][0].id if ivf_results else 'N/A'}")
    print(f"  {'HNSW (ef=32)':<16} | {t_hnsw:7.3f} ms   | {hnsw_recall:5.1f}%       | {hnsw_results[0][0].id if hnsw_results else 'N/A'}")
    print()


def run_quantization_demo() -> None:
    print("-" * 80)
    print("STAGE 3: MEMORY QUANTIZATION: FP32 vs SQ8 (INT8) vs BQ (1-BIT)")
    print("-" * 80)

    count = 10_000_000
    dims = 1536  # Standard OpenAI text-embedding-3-large / Ada-002 dimension

    sq8_savings = ScalarQuantizer.calculate_ram_savings(count, dims)
    bq_savings = BinaryQuantizer.calculate_ram_savings(count, dims)

    print(f"Calculated RAM footprint for {count:,} vectors of {dims} dimensions:")
    print(f"  1. Standard FP32:      {sq8_savings['fp32_mb'] / 1024:7.2f} GB RAM (Baseline 100%)")
    print(f"  2. Scalar Quant (SQ8): {sq8_savings['sq8_mb'] / 1024:7.2f} GB RAM ({sq8_savings['compression_ratio']}x compression, {sq8_savings['savings_percentage']}% savings)")
    print(f"  3. 1-Bit Binary (BQ):  {bq_savings['bq_mb'] / 1024:7.2f} GB RAM ({bq_savings['compression_ratio']}x compression, {bq_savings['savings_percentage']}% savings)\n")

    # Quantization round-trip test on sample vector
    raw_vec = [random.gauss(0.0, 1.0) for _ in range(8)]
    sq = ScalarQuantizer()
    sq.fit([raw_vec])
    q_vec = sq.quantize(raw_vec)
    reconstructed = sq.dequantize(q_vec)

    print("SQ8 Quantization Round-trip Example (8 dims):")
    print(f"  Original FP32:      {[round(x, 3) for x in raw_vec]}")
    print(f"  Quantized INT8:     {q_vec}")
    print(f"  Reconstructed FP32: {[round(x, 3) for x in reconstructed]}\n")


def run_filtered_search_demo() -> None:
    print("-" * 80)
    print("STAGE 5: SINGLE-STAGE PAYLOAD FILTERING vs POST-FILTERING")
    print("-" * 80)

    dim = 32
    items = generate_synthetic_vectors(300, dim, seed=99)
    hnsw = HNSWIndex(dimensions=dim, metric=DistanceMetric.COSINE, M=16, ef_search=32)
    for it in items:
        hnsw.add(it)

    query = items[0].vector
    predicate = lambda p: p.get("category") == "ai_ml" and p.get("rating", 0) >= 4.0

    filtered_results = hnsw.search(query, k=5, filter_predicate=predicate)
    print(f"Searching nearest neighbors with predicate: category=='ai_ml' and rating >= 4.0")
    print(f"Found {len(filtered_results)} matching vector items:")
    for rank, (item, dist) in enumerate(filtered_results, 1):
        print(f"  #{rank}: ID={item.id} | Cosine Dist={dist:.4f} | Category={item.payload['category']} | Rating={item.payload['rating']}")
    print()


def main() -> None:
    run_roadmap_showcase()
    run_metrics_demo()
    run_benchmark_demo()
    run_quantization_demo()
    run_filtered_search_demo()


if __name__ == "__main__":
    main()
