import inspect
import torch

from typing import Callable, TypeVar, Any
from torch import nn
from torch import Tensor
from enum import Enum

class ActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    LEAKY_RELU = "leaky_relu"

T = TypeVar('T')
def init_add_activation(fn: Callable[..., T]) -> Callable[..., T]:
    def wrapper(self, *args: Any, activation=ActivationType.GELU, **kwargs: Any) -> T:
        sig = inspect.signature(fn)
        ba = sig.bind_partial(self, *args, **kwargs)
        if 'activation' in sig.parameters:
            ba.arguments['activation'] = activation
        ba.apply_defaults()
        result = fn(*ba.args, **ba.kwargs)
        if activation == ActivationType.RELU:
            self.activation_fn = nn.ReLU()
        elif activation == ActivationType.GELU:
            self.activation_fn = nn.GELU()
        elif activation == ActivationType.LEAKY_RELU:
            self.activation_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        return result
    return wrapper

def point_rope(x: Tensor, # shape: (B, N, D)
                                 pos: Tensor, # shape: (B, N, 3)
                                 base: int = 128
                                 ) -> Tensor:
    _B, _N, D = x.shape
    assert D % 4 == 0
    part_length = D // 4
    device = x.device
    dtype = x.dtype

    inv_freq = 1.0 / (base ** (torch.arange(0, part_length, 2, device=device, dtype=dtype) / part_length)) # (part_length/2,)
    theta = torch.repeat_interleave(inv_freq, 2) # (part_length,)

    out = x.clone()

    for i in range(3): # x/y/z direction
        pos_i = pos[:, :, i] # (B, N)
        angle = pos_i.unsqueeze(-1) * theta # (B, N, part_length)
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        x_part = x[:, :, i * part_length:(i + 1) * part_length]
        x_even = x_part[:, :, ::2]
        x_odd = x_part[:, :, 1::2]

        x_rotated_even = x_even * cos[:, :, ::2] - x_odd * sin[:, :, ::2]
        x_rotated_odd = x_even * sin[:, :, ::2] + x_odd * cos[:, :, ::2]

        out[:, :, i * part_length:(i + 1) * part_length][:, :, ::2] = x_rotated_even
        out[:, :, i * part_length:(i + 1) * part_length][:, :, 1::2] = x_rotated_odd

    return out

def furthest_point_sampling_disjoint_clusters(points: Tensor, cluster_size: int) -> Tensor:
    """
    Furthest point sampling with disjoint clusters of top-K nearest neighbors.

    Args:
        points: (N, 3) or (B, N, 3)
        cluster_size: number of points in each cluster

    Returns:
        cluster_indices: (num_clusters, cluster_size) or (B, num_clusters, cluster_size)
    """
    device = points.device

    def knn_indices(center, all_points, k, mask):
        dist = torch.sum((all_points - center.unsqueeze(-2)) ** 2, dim=-1)
        if mask is not None:
            dist = dist.masked_fill(mask, float('inf'))
        return torch.topk(dist, k, dim=-1, largest=False)[1]


    if points.dim() == 2:
        N = points.shape[0]
        assert N % cluster_size == 0, f"points.shape[0]:{points.shape[0]} must be divisible by cluster_size:{cluster_size}"
        num_clusters = N // cluster_size

        mask = torch.zeros(N, dtype=torch.bool, device=device)
        cluster_indices = torch.empty((num_clusters, cluster_size), dtype=torch.long, device=device)
        distances = torch.ones(N, device=device) * 1e10
        farthest = torch.randint(0, N, (1,), device=device).item()

        for i in range(num_clusters):
            while mask[farthest]:
                distances[farthest] = -1
                farthest = torch.argmax(distances).item()
            centroid = points[farthest]
            cluster = knn_indices(centroid, points, cluster_size, mask)
            cluster_indices[i] = cluster
            mask[cluster] = True
            dist = torch.sum((points - centroid) ** 2, dim=1)
            distances = torch.minimum(distances, dist)
            distances[mask] = -1

        return cluster_indices

    elif points.dim() == 3:
        B, N, _ = points.shape
        assert N % cluster_size == 0, f"points.shape[1]:{points.shape[1]} must be divisible by cluster_size:{cluster_size}"
        num_clusters = N // cluster_size

        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        cluster_indices = torch.empty((B, num_clusters, cluster_size), dtype=torch.long, device=device)
        distances = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), device=device)

        for i in range(num_clusters):
            for b in range(B):
                while mask[b, farthest[b]]:
                    distances[b, farthest[b]] = -1
                    farthest[b] = torch.argmax(distances[b])
            centroid = points[torch.arange(B), farthest, :]
            cluster = knn_indices(centroid, points, cluster_size, mask)
            cluster_indices[:, i, :] = cluster
            mask[torch.arange(B).unsqueeze(1), cluster] = True
            dist = torch.sum((points - centroid.unsqueeze(1)) ** 2, dim=2)
            distances = torch.minimum(distances, dist)
            distances[mask] = -1

        return cluster_indices

    else:
        raise ValueError("points must be (N, 3) or (B, N, 3)")