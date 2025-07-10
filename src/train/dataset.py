import json
import torch
import numpy as np

from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from PIL import Image
from PIL import ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUNRGBD_PATH = PROJECT_ROOT / "datasets" / "SUNRGBD"
SUNRGBD_PATH_IGNORE = [".DS_Store"]
SUNRGBD_CLASS_IGNORE = ["wall:occluded", "wall:truncated"]

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

def fitted_intrinsics(
    intr: Intrinsics,
    size_src: tuple[int, int],
    size_tgt: tuple[int, int],
    centering: tuple[float, float] = (0.5, 0.5)
) -> Intrinsics:
    scale = max(size_tgt[0] / size_src[0], size_tgt[1] / size_src[1])
    
    tx = size_tgt[0] / 2 - scale * size_src[0] * centering[0]
    ty = size_tgt[1] / 2 - scale * size_src[1] * centering[1]
    
    return Intrinsics(
        fx=scale * intr.fx,
        fy=scale * intr.fy,
        cx=scale * intr.cx + tx,
        cy=scale * intr.cy + ty
    )

class SunRgbdDataset(Dataset):
    def __init__(self, norm_param: Optional[tuple[float, float]] = None) -> None:
        super().__init__()

        all_candidate_items = []
        for sub_dataset in filter(lambda x: x.name not in SUNRGBD_PATH_IGNORE, SUNRGBD_PATH.iterdir()):
            for sub_sub_dataset in filter(lambda x: x.name not in SUNRGBD_PATH_IGNORE, sub_dataset.iterdir()):
                if sub_sub_dataset.name == "sun3ddata":
                    for sub_sub_sub_dataset in filter(lambda x: x.name not in SUNRGBD_PATH_IGNORE, sub_sub_dataset.iterdir()):
                        for sub_sub_sub_sub_dataset in filter(lambda x: x.name not in SUNRGBD_PATH_IGNORE, sub_sub_sub_dataset.iterdir()):
                            all_candidate_items += sorted(filter(lambda x: x.name not in SUNRGBD_PATH_IGNORE, sub_sub_sub_sub_dataset.iterdir()))
                    continue
                all_candidate_items += sorted(filter(lambda x: x.name not in SUNRGBD_PATH_IGNORE, sub_sub_dataset.iterdir()))

        class_counter = Counter()
        item_class_map = {}

        for item_path in all_candidate_items:
            annotation_path = item_path / "annotation3Dfinal" / "index.json"
            if not annotation_path.exists():
                continue

            try:
                with open(annotation_path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                continue

            class_names = []
            if "objects" in data:
                for obj in data["objects"]:
                    if obj is not None and not isinstance(obj, list) and obj["polygon"] and "name" in obj:
                        name = obj.get("name")
                        class_names.append(name)
                        class_counter[name] += 1

            if class_names:
                item_class_map[item_path] = class_names

        valid_classes = {class_name for class_name, cnt in class_counter.items() if cnt >= 500 and class_name not in SUNRGBD_CLASS_IGNORE}
        print("Number of classes:", len(class_counter))
        print("Filtered valid (N>=500) classes frequency:")
        for class_name in valid_classes:
            print(f"  {class_name}: {class_counter[class_name]}")
        self.class_name = tuple(sorted(valid_classes))
        self.items = [item_path for item_path, classes in item_class_map.items() 
                      if any(class_name in valid_classes for class_name in classes)]
        self.length = len(self.items)
        print(f"Filtered valid items: {self.length}")

        if norm_param:
            self.mean = norm_param[0]
            self.std = norm_param[1]
            return

        self.mean = torch.zeros(3, dtype=torch.float64)
        self.std = torch.zeros(3, dtype=torch.float64)
        n_pixels = 0

        for item_path in self.items:
            try:
                image_path = next((item_path / "image").iterdir())
                raw_image = Image.open(image_path).convert("RGB")
                image = ImageOps.fit(raw_image, (560, 420))  # Resize + center crop
                image_tensor = TF.to_tensor(image).to(torch.float64)  # (3, H, W)

                n = image_tensor.shape[1] * image_tensor.shape[2]
                self.mean += image_tensor.sum(dim=[1, 2])
                self.std += (image_tensor ** 2).sum(dim=[1, 2])
                n_pixels += n
            except Exception as e:
                print(f"Error processing {item_path}: {e}")
                continue

        self.mean /= n_pixels
        self.std = (self.std / n_pixels - self.mean ** 2).sqrt()

        print(f"Dataset Image mean: {self.mean}")
        print(f"Dataset Image std:  {self.std}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> tuple[tuple[Tensor, Tensor], Any]:
        item_path = self.items[idx]

        image_path = next((item_path / "image").iterdir())
        depth_path = next((item_path / "depth_bfx").iterdir())
        intrinsics_path = item_path / "intrinsics.txt"
        annotation_path = item_path / "annotation3Dfinal" / "index.json"

        raw_image = Image.open(image_path)
        raw_depth = Image.open(depth_path)

        target_size = (560, 420)
        image = ImageOps.fit(raw_image, target_size)
        depth = ImageOps.fit(raw_depth, target_size)

        raw_intrinsics = intrinsics_path.read_text().split()
        fx = float(raw_intrinsics[0])
        cx = float(raw_intrinsics[2])
        fy = float(raw_intrinsics[4])
        cy = float(raw_intrinsics[5])
        original_size = raw_image.size

        intrinsics = fitted_intrinsics(
            Intrinsics(fx=fx, cx=cx, fy=fy, cy=cy),
            original_size,
            target_size
        )

        depth_vis = np.array(depth, dtype=np.uint16)
        depth_inpaint = np.bitwise_or(np.right_shift(depth_vis, 3), np.right_shift(depth_vis, 16-3))
        depth_meters = torch.from_numpy(depth_inpaint.astype(np.float32) / 1000.0)
        H, W = depth_meters.shape
        z = depth_meters

        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        xs = xs.to(z.dtype)
        ys = ys.to(z.dtype)

        x = (xs - intrinsics.cx) * z / intrinsics.fx
        y = (ys - intrinsics.cy) * z / intrinsics.fy

        points = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        image = (TF.to_tensor(image) - self.mean.reshape(3, 1, 1)) / self.std.reshape(3, 1, 1)
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            
        objects = []
        for obj in data["objects"]:
            if obj == None or isinstance(obj, list) or not obj["polygon"] or obj["name"] not in self.class_name:
                continue
            class_idx = self.class_name.index(obj["name"])

            xz_coords = list(zip(obj["polygon"][0]["X"], obj["polygon"][0]["Z"]))
            xz_tensor = torch.tensor(xz_coords)  # shape: (4, 2)

            y_bounds = torch.tensor([
                obj["polygon"][0]["Ymin"],
                obj["polygon"][0]["Ymax"]
            ])  # shape: (2,)

            objects.append((class_idx, xz_tensor, y_bounds))
        return (points, image), objects

def sun_rgbd_collector(batch):
    points = torch.stack([item[0][0] for item in batch])
    images = torch.stack([item[0][1] for item in batch])
    objects_list = [item[1] for item in batch]

    return (points, images), objects_list
