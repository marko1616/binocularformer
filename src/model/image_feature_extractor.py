import torch
import torch.nn as nn

from .utils import ActivationType, init_add_activation

RGB_CHANNEL = 3

class ImageFeatureExtractor(nn.Module):
    @init_add_activation
    def __init__(self, out_channels: int, num_layers: int, kernel_size: int, padding: int, activation: ActivationType) -> None:
        super().__init__()
        
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.padding = padding

        self.initial_conv = nn.Conv2d(RGB_CHANNEL, out_channels, 
                                     kernel_size=kernel_size, 
                                     padding=padding)
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.residual_blocks.append(
                ResidualBlock(out_channels, out_channels, 
                            kernel_size=kernel_size, 
                            padding=padding,
                            activation=activation)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract image feature (B, C, H, W)
        x = self.initial_conv(x)
        for block in self.residual_blocks:
            x = block(x)
        return x

class ResidualBlock(nn.Module):
    @init_add_activation
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, 
                              padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=kernel_size, 
                              padding=padding)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 
                                    kernel_size=1)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.activation_fn(self.conv1(x))
        out = self.conv2(out)
        
        out += residual
        out = self.activation_fn(out)
        
        return out