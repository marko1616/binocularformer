import torch

from shapely.geometry import Polygon

def box3d_iou(bottom_box1, y_box1, bottom_box2, y_box2):
    bottom_box1_np = bottom_box1.detach().cpu().numpy()
    bottom_box2_np = bottom_box2.detach().cpu().numpy()

    poly1 = Polygon(bottom_box1_np)
    poly2 = Polygon(bottom_box2_np)

    area1 = poly1.area
    area2 = poly2.area

    height1 = float(y_box1[1] - y_box1[0])
    height2 = float(y_box2[1] - y_box2[0])

    vol1 = area1 * height1
    vol2 = area2 * height2

    if poly1.intersects(poly2):
        bottom_intersection = poly1.intersection(poly2).area
    else:
        bottom_intersection = 0.0

    y_min_inter = max(float(y_box1[0]), float(y_box2[0]))
    y_max_inter = min(float(y_box1[1]), float(y_box2[1]))
    height_intersection = max(0.0, y_max_inter - y_min_inter)

    intersection_vol = bottom_intersection * height_intersection

    union_vol = vol1 + vol2 - intersection_vol
    iou_3d = intersection_vol / union_vol if union_vol > 0 else 0.0
    
    return torch.tensor(iou_3d, device=bottom_box1.device)
