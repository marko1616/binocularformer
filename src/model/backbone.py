import torch

from torch import nn
from torch import Tensor

from .image_feature_extractor import ImageFeatureExtractor
from .head import DetectionHead
from .transformer import BinocularformerEncoder
from .utils import ActivationType, init_add_activation, furthest_point_sampling_disjoint_clusters

class BinocularFormer(nn.Module):
    @init_add_activation
    def __init__(self,
                 model_dim: int = 512,
                 cnn_layer_num: int = 4,
                 cnn_kernel_size: int = 3,
                 cnn_padding: int = 1,
                 cluster_sizes: list[int] = [4, 4, 4, 4],
                 encoder_layer_num: int = 4,
                 encoder_head_num: int = 4,
                 encoder_ffd_dim: int = 1024,
                 detection_head_num: int = 4,
                 class_num: int = 20,
                 dropout: float = 0.1,
                 activation: ActivationType = ActivationType.GELU
                 ) -> None:
        super().__init__()

        self.cluster_sizes = cluster_sizes
        self.num_hierarchies = len(cluster_sizes)

        self.image_feature_extractor = ImageFeatureExtractor(model_dim, cnn_layer_num, cnn_kernel_size, cnn_padding, activation)

        self.encoder_stages = nn.ModuleList([
            BinocularformerEncoder(encoder_layer_num, model_dim, encoder_head_num, encoder_ffd_dim, dropout, activation)
            for _ in range(self.num_hierarchies)
        ])

        self.class_num = class_num
        self.detection_head_num = detection_head_num
        self.detection_heads = nn.ModuleList([
            DetectionHead(model_dim, self.class_num)
            for _ in range(detection_head_num)
        ])

    def forward(self, positions: Tensor, image: Tensor) -> Tensor:
        B, _, H, W = image.shape
        image_feature = self.image_feature_extractor(image)
        image_feature = image_feature.view(B, -1, H * W).permute(0, 2, 1)  # (B, N, model_dim)

        feat = image_feature
        pos = positions

        for stage in range(self.num_hierarchies):
            cluster_indices = furthest_point_sampling_disjoint_clusters(pos, self.cluster_sizes[stage])  # (B, num_clusters, cluster_size)
            B, num_clusters, cluster_size = cluster_indices.shape

            new_feat_list = []
            new_pos_list = []

            # DO NOT USE GATHER HERE IF YOU DONT WANT YOUR VRAM BEING FUCKED UP
            for b in range(B):
                for c in range(num_clusters):
                    indices = cluster_indices[b, c]  # (cluster_size)
                    cluster_feat = feat[b, indices]  # (cluster_size, model_dim)
                    cluster_pos = pos[b, indices]    # (cluster_size, 3)

                    new_feat_list.append(cluster_feat)
                    new_pos_list.append(cluster_pos)

            gathered_flat = torch.stack(new_feat_list).reshape(B * num_clusters, cluster_size, -1)
            pos_flat = torch.stack(new_pos_list).reshape(B * num_clusters, cluster_size, -1)

            encoded = self.encoder_stages[stage](pos_flat, gathered_flat)
            encoded = encoded.reshape(B, num_clusters, cluster_size + 1, -1)  # (B, num_clusters, cluster_size+1, model_dim)

            feat = encoded[:, :, -1, :]  # (B, num_clusters, model_dim)

            new_pos_list = []
            for b in range(B):
                for c in range(num_clusters):
                    indices = cluster_indices[b, c]
                    cluster_pos = pos[b, indices]
                    new_pos_list.append(cluster_pos.mean(dim=0))

            pos = torch.stack(new_pos_list).reshape(B, num_clusters, -1)

        result = []
        for i in range(num_clusters):
            result += [self.detection_heads[j](pos[:,i], feat[:,i]) for j in range(self.detection_head_num)]

        return tuple(result), pos