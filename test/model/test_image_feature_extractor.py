import torch
import unittest

from src.model.image_feature_extractor import ImageFeatureExtractor, ResidualBlock
from src.model.utils import ActivationType

class TestImageFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.img_size = 16
        self.out_channels = 32
        self.test_img = torch.randn(self.batch_size, 3, self.img_size, self.img_size)

    def test_output_shape(self):
        model = ImageFeatureExtractor(
            out_channels=self.out_channels,
            num_layers=2,
            kernel_size=3,
            padding=1,
            activation=ActivationType.RELU
        )
        output = model(self.test_img)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.img_size, self.img_size))

class TestResidualBlock(unittest.TestCase):
    def test_block(self):
        block = ResidualBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            activation=ActivationType.GELU
        )
        x = torch.randn(2, 32, 16, 16)
        self.assertEqual(block(x).shape, (2, 64, 16, 16))

if __name__ == "__main__":
    unittest.main()