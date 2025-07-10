import torch
import unittest

from src.model.transformer import BinocularformerEncoder

class TestBinocularformerEncoder(unittest.TestCase):
    def test_random_input(self):
        batch_size = 2
        seq_len = 128
        d_model = 128
        encoder = BinocularformerEncoder(
            num_layers=2,
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1
        )

        position = torch.randn(batch_size, seq_len, 3)
        points = torch.randn(batch_size, seq_len, d_model)
        output = encoder(position, points)
        self.assertEqual(output.shape, (batch_size, seq_len + 1, d_model))

if __name__ == "__main__":
    unittest.main()