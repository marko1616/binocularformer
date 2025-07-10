import torch
import unittest

from src.model.utils import furthest_point_sampling_disjoint_clusters, point_rope

class TestFurthestPointSamplingDisjointClusters(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_single_batch(self):
        points = torch.rand(512, 3)
        cluster_size = 64
        cluster_indices = furthest_point_sampling_disjoint_clusters(points, cluster_size)
        self.assertEqual(cluster_indices.shape, (512 // 64, 64))

        flat_indices = cluster_indices.flatten()
        self.assertEqual(len(flat_indices.unique()), 512)

        for i in range(cluster_indices.shape[0]):
            cluster = cluster_indices[i]
            self.assertEqual(len(cluster), cluster_size)
            self.assertEqual(len(cluster.unique()), cluster_size)

    def test_multi_batch(self):
        B, N, cluster_size = 4, 512, 64
        points = torch.rand(B, N, 3)
        cluster_indices = furthest_point_sampling_disjoint_clusters(points, cluster_size)

        self.assertEqual(cluster_indices.shape, (B, N // cluster_size, cluster_size))

        for b in range(B):
            flat = cluster_indices[b].flatten()
            self.assertEqual(len(flat.unique()), N)

            for i in range(cluster_indices.shape[1]):
                cluster = cluster_indices[b, i]
                self.assertEqual(len(cluster), cluster_size)
                self.assertEqual(len(cluster.unique()), cluster_size)

    def test_assert_divisible(self):
        points = torch.rand(500, 3)  # Not divisible by 64
        with self.assertRaises(AssertionError):
            furthest_point_sampling_disjoint_clusters(points, cluster_size=64)

    def test_value_range(self):
        points = torch.rand(256, 3)
        cluster_size = 32
        cluster_indices = furthest_point_sampling_disjoint_clusters(points, cluster_size)
        self.assertTrue(torch.all(cluster_indices < 256))
        self.assertTrue(torch.all(cluster_indices >= 0))

class TestPointRotaryPositionalEmbeddings(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_per_pair_l2_norm_preserved(self):
        B, N, D = 2, 64, 64
        x = torch.randn(B, N, D)
        pos = torch.rand(B, N, 3)
        out = point_rope(x.clone(), pos)

        x_pairs = x.reshape(B, N, D // 2, 2)
        out_pairs = out.reshape(B, N, D // 2, 2)

        x_norms = x_pairs.norm(dim=-1)
        out_norms = out_pairs.norm(dim=-1)

        self.assertTrue(torch.allclose(x_norms, out_norms, atol=1e-4))

if __name__ == '__main__':
    unittest.main()
