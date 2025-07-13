import unittest
import torch

from functools import reduce
from src.model.backbone import BinocularFormer

class TestBinocularFormer(unittest.TestCase):
    def test_binocularformer_forward(self):
        B = 4
        C = 3
        H = 420
        W = 560
        N = H * W

        positions = torch.rand(B, N, 3).to(torch.float16).to("cuda")  # (B, N, 3)
        image = torch.rand(B, C, H, W).to(torch.float16).to("cuda")   # (B, C, H, W)

        model_dim = 64
        encoder_layer_num = 2
        encoder_ffd_dim = 64
        detection_head_num = 4
        cluster_sizes = [28, 28, 15, 10]

        model = BinocularFormer(model_dim=model_dim,
                                encoder_layer_num=encoder_layer_num,
                                encoder_ffd_dim=encoder_ffd_dim,
                                cluster_sizes=cluster_sizes,
                                detection_head_num=4).to(torch.float16).to("cuda")

        with torch.no_grad():
            output, _ = model(positions, image)

        expected_num_clusters = reduce(lambda x, y: x // y, [N]+cluster_sizes)
        self.assertEqual(len(output), expected_num_clusters*detection_head_num)

if __name__ == '__main__':
    unittest.main()
