import math
import torch

from shapely.geometry import Polygon, Point
from shapely.errors import GEOSException

def unwrap_ddp(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

def box3d_iou(bottom_box1, y_box1, bottom_box2, y_box2):
    try:
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
    except GEOSException:
        print(bottom_box1)
        print(y_box1)
        print(bottom_box2)
        print(y_box2)
        return torch.tensor(0.0, device=bottom_box1.device)

def box3d_center(bottom_box, y_box):
    try:
        bottom_box_np = bottom_box.detach().cpu().numpy()

        poly = Polygon(bottom_box_np)
        bottom_center = poly.centroid.coords[0]

        y_min = float(y_box[0])
        y_max = float(y_box[1])
        y_center = (y_min + y_max) / 2.0

        center_3d = torch.tensor([bottom_center[0], bottom_center[1], y_center], 
                                device=bottom_box.device)
        
        return center_3d
    except GEOSException:
        print("Error in calculating box center:")
        print(bottom_box)
        print(y_box)
        return torch.tensor([0.0, 0.0, 0.0], device=bottom_box.device)
    
def is_point_inside_box(bottom_box, y_box, point):
    try:
        bottom_box_np = bottom_box.detach().cpu().numpy()
        point_np = point.detach().cpu().numpy()

        y_min = float(y_box[0])
        y_max = float(y_box[1])
        if not (y_min <= point_np[2] <= y_max):
            return False

        poly = Polygon(bottom_box_np)
        point_xy = Point(point_np[0], point_np[1])
        
        return poly.contains(point_xy)
        
    except GEOSException:
        print("Error in point-in-box check:")
        print(bottom_box)
        print(y_box)
        print(point)
        return False

def is_rectangle(box):
    box = box.detach().cpu().numpy()
    poly = Polygon(box)

    mbr = poly.minimum_rotated_rectangle
    return math.isclose(poly.area, mbr.area, rel_tol=1e-4)