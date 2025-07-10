import torch

from torch import nn
from torch import Tensor

class DetectionHead(nn.Module):
    def __init__(self, in_features: int, class_num: int):
        super().__init__()

        self.objectness_branch = nn.Linear(in_features, 1)

        self.class_branch = nn.Sequential(
            nn.Linear(in_features, class_num),
            nn.Softmax(dim=-1)
        )

        self.xz_branch = nn.Linear(in_features, 5)
        self.y_branch = nn.Linear(in_features, 2)
    
    def forward(self, center: Tensor, feature: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        device = center.device
        dtype = center.dtype
        if feature.dim() > 2:
            batch_size = feature.size(0)
            feature = feature.view(batch_size, -1)

        objectness_logits = self.objectness_branch(feature)
        class_pred = self.class_branch(feature)

        w, h, theta, offset = self.xz_branch(feature).split([1, 1, 1, 2], dim=-1)
        corners = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=dtype, device=device)  # (4, 2)
        corners = corners * torch.cat([w, h], dim=-1).unsqueeze(1) / 2  # (B, 4, 2)
        cos_t = torch.cos(theta).squeeze(1)
        sin_t = torch.sin(theta).squeeze(1)
        rot_mat = torch.stack([
            torch.stack([cos_t, -sin_t], dim=-1),
            torch.stack([sin_t,  cos_t], dim=-1)
        ], dim=-2)
        xz_pred = torch.einsum('bij,bjk->bik', corners, rot_mat)
        xz_pred = xz_pred + offset.unsqueeze(1)
        xz_pred = center[..., [0, 2]].unsqueeze(1) + xz_pred

        y_min, height = self.y_branch(feature).split([1, 1], dim=-1)
        y_pred = torch.stack([
            center[..., 1] + y_min.squeeze(1),
            center[..., 1] + y_min.squeeze(1) + height.squeeze(1)
        ], dim=-1)

        return objectness_logits, class_pred, xz_pred, y_pred