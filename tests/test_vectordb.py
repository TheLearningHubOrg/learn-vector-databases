"""
Comprehensive Unit Tests for Vector Databases Engine.

Tests:
- Vector distance metrics (Euclidean, Cosine, Manhattan, Dot Product, Hamming).
- L2 normalization mechanics.
- Exact kNN exhaustive search & filtering.
- IVF k-means clustering, Voronoi partitioning & nprobe query traversal.
- HNSW multi-layer graph construction, beam search & single-stage payload filtering.
- Scalar Quantization (SQ8) & Binary Quantization (BQ).
- Multi-tenant collection isolation.
- Pedagogical roadmap completion.
"""

import math
import unittest
from typing import List

from src.learn_vector_databases.roadmap import StageStatus, VectorDatabasesRoadmap
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


class TestVectorDatabases(unittest.TestCase):

    def test_distance_metrics(self) -> None:
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]

        # Euclidean: sqrt((1-0)^2 + (0-1)^2) = sqrt(2)
        d_euc = calculate_distance(v1, v2, DistanceMetric.EUCLIDEAN)
        self.assertAlmostEqual(d_euc, math.sqrt(2.0), places=5)

        # Cosine: orthogonal vectors have cos(theta)=0 -> distance = 1 - 0 = 1.0
        d_cos = calculate_distance(v1, v2, DistanceMetric.COSINE)
        self.assertAlmostEqual(d_cos, 1.0, places=5)

        # Manhattan: |1-0| + |0-1| = 2.0
        d_man = calculate_distance(v1, v2, DistanceMetric.MANHATTAN)
        self.assertAlmostEqual(d_man, 2.0, places=5)

        # Dot product: - (1*0 + 0*1) = 0.0
        d_dot = calculate_distance(v1, v2, DistanceMetric.DOT_PRODUCT)
        self.assertAlmostEqual(d_dot, 0.0, places=5)

        # Hamming distance: 2 differences
        d_ham = calculate_distance([1.0, 0.0], [0.0, 1.0], DistanceMetric.HAMMING)
        self.assertEqual(d_ham, 2.0)

        # Dimension mismatch
        with self.assertRaises(ValueError):
            calculate_distance([1.0], [1.0, 2.0])

    def test_l2_normalization(self) -> None:
        raw = [3.0, 4.0]
        norm = l2_normalize(raw)
        self.assertAlmostEqual(norm[0], 0.6, places=5)
        self.assertAlmostEqual(norm[1], 0.8, places=5)
        self.assertAlmostEqual(math.sqrt(norm[0] ** 2 + norm[1] ** 2), 1.0, places=5)

        # Zero vector handling
        zero_norm = l2_normalize([0.0, 0.0])
        self.assertEqual(zero_norm, [0.0, 0.0])

    def test_exact_knn(self) -> None:
        index = ExactKNNIndex(metric=DistanceMetric.EUCLIDEAN)
        items = [
            VectorItem("1", [0.0, 0.0], {"lang": "py"}),
            VectorItem("2", [1.0, 1.0], {"lang": "ts"}),
            VectorItem("3", [10.0, 10.0], {"lang": "py"}),
        ]
        index.add_batch(items)
        self.assertEqual(len(index), 3)

        results = index.search([0.1, 0.1], k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].id, "1")

        # Test predicate filter
        py_results = index.search([0.9, 0.9], k=2, filter_predicate=lambda p: p.get("lang") == "py")
        self.assertEqual(len(py_results), 2)
        self.assertTrue(all(it.payload["lang"] == "py" for it, _ in py_results))

    def test_ivf_index(self) -> None:
        items = [
            VectorItem("a", [1.0, 1.0]),
            VectorItem("b", [1.1, 0.9]),
            VectorItem("c", [10.0, 10.0]),
            VectorItem("d", [9.9, 10.2]),
        ]
        ivf = IVFIndex(nlist=2, nprobe=1, metric=DistanceMetric.EUCLIDEAN)
        ivf.train_and_build(items, iterations=5)

        results = ivf.search([1.05, 0.95], k=2)
        self.assertTrue(len(results) > 0)
        top_id = results[0][0].id
        self.assertIn(top_id, ["a", "b"])

    def test_hnsw_index_and_payload_filtering(self) -> None:
        dim = 4
        hnsw = HNSWIndex(dimensions=dim, metric=DistanceMetric.EUCLIDEAN, M=4, ef_construction=16, ef_search=16)

        items = [
            VectorItem("doc_0", [0.0, 0.0, 0.0, 0.0], {"topic": "math"}),
            VectorItem("doc_1", [0.1, 0.1, 0.0, 0.0], {"topic": "physics"}),
            VectorItem("doc_2", [0.2, 0.1, 0.0, 0.0], {"topic": "math"}),
            VectorItem("doc_3", [5.0, 5.0, 5.0, 5.0], {"topic": "math"}),
            VectorItem("doc_4", [5.1, 5.0, 5.0, 5.0], {"topic": "art"}),
        ]

        for it in items:
            hnsw.add(it)

        self.assertEqual(len(hnsw), 5)

        # Standard nearest search
        results = hnsw.search([0.05, 0.05, 0.0, 0.0], k=2)
        self.assertTrue(len(results) >= 2)
        self.assertIn(results[0][0].id, ["doc_0", "doc_1", "doc_2"])

        # Payload filtered search: must only match topic == 'physics'
        physics_results = hnsw.search(
            [0.05, 0.05, 0.0, 0.0],
            k=3,
            filter_predicate=lambda p: p.get("topic") == "physics",
        )
        self.assertEqual(len(physics_results), 1)
        self.assertEqual(physics_results[0][0].id, "doc_1")

        # Wrong dimension error
        with self.assertRaises(ValueError):
            hnsw.add(VectorItem("bad", [1.0, 2.0]))

    def test_scalar_quantization(self) -> None:
        sq = ScalarQuantizer()
        vecs = [
            [-2.0, 0.0, 2.0],
            [1.0, -1.0, 0.5],
        ]
        sq.fit(vecs)
        self.assertTrue(sq.is_fitted)

        q = sq.quantize([-2.0, 0.0, 2.0])
        self.assertEqual(q[0], 0)
        self.assertEqual(q[2], 255)

        dequantized = sq.dequantize(q)
        self.assertAlmostEqual(dequantized[0], -2.0, places=1)
        self.assertAlmostEqual(dequantized[2], 2.0, places=1)

        savings = ScalarQuantizer.calculate_ram_savings(1_000_000, 1536)
        self.assertEqual(savings["compression_ratio"], 4.0)
        self.assertEqual(savings["savings_percentage"], 75.0)

    def test_binary_quantization(self) -> None:
        bq = BinaryQuantizer(dimensions=4)
        v1 = [1.5, -0.5, 0.2, -1.0]   # signs: 1, 0, 1, 0 -> 0b0101 = 5
        v2 = [1.0, 0.5, 0.1, -2.0]    # signs: 1, 1, 1, 0 -> 0b0111 = 7

        b1 = bq.quantize_to_bits(v1)
        b2 = bq.quantize_to_bits(v2)

        self.assertEqual(b1, 0b0101)
        self.assertEqual(b2, 0b0111)

        ham = bq.hamming_distance(b1, b2)
        self.assertEqual(ham, 1)  # Only bit index 1 differs

        savings = BinaryQuantizer.calculate_ram_savings(1_000_000, 1536)
        self.assertAlmostEqual(savings["compression_ratio"], 32.0, delta=0.5)

    def test_multi_tenant_vector_store(self) -> None:
        store = MultiTenantVectorStore(dimensions=3, metric=DistanceMetric.EUCLIDEAN)
        store.insert("tenant_alpha", VectorItem("doc_a", [1.0, 0.0, 0.0]))
        store.insert("tenant_beta", VectorItem("doc_b", [0.0, 1.0, 0.0]))

        res_alpha = store.search("tenant_alpha", [1.0, 0.0, 0.0], k=5)
        self.assertEqual(len(res_alpha), 1)
        self.assertEqual(res_alpha[0][0].id, "doc_a")

        # Isolated: tenant_beta results must NOT appear in tenant_alpha
        res_beta = store.search("tenant_beta", [1.0, 0.0, 0.0], k=5)
        self.assertEqual(len(res_beta), 1)
        self.assertEqual(res_beta[0][0].id, "doc_b")

        # Unknown tenant returns empty
        res_unknown = store.search("tenant_gamma", [1.0, 0.0, 0.0], k=5)
        self.assertEqual(len(res_unknown), 0)

    def test_roadmap_progress(self) -> None:
        roadmap = VectorDatabasesRoadmap()
        self.assertEqual(len(roadmap.stages), 7)
        self.assertEqual(roadmap.total_progress(), 100.0)
        summary = roadmap.get_summary()
        self.assertEqual(summary["total_stages"], 7)
        self.assertEqual(summary["completed_stages"], 7)
        self.assertEqual(summary["overall_progress_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
