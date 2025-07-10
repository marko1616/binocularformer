import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader

from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from rich.table import Table

from .utils import box3d_iou
from .dataset import sun_rgbd_collector

class BinocularFormerTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        batch_size=8,
        lr=1e-8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
        objectness_weight=1.0,
        class_weight=1.0,
        box_weight=1.0,
        height_weight=1.0,
        neg_pos_ratio=3.0
    ):
        self.model = model.to(dtype).to(device)
        self.dtype = dtype
        self.device = device
        self.objectness_weight = objectness_weight
        self.class_weight = class_weight
        self.box_weight = box_weight
        self.height_weight = height_weight
        self.neg_pos_ratio = neg_pos_ratio
        self.console = Console()
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=sun_rgbd_collector
        )
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
        )
        
        self.objectness_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.class_criterion = nn.CrossEntropyLoss(reduction='none')
        self.box_criterion = nn.SmoothL1Loss(reduction='none')
        self.height_criterion = nn.SmoothL1Loss(reduction='none')
        
        self.class_num = model.class_num
        self.detection_head_num = model.detection_head_num
        
    def create_progress(self, description):
        return Progress(
            TextColumn(f"[bold blue]{description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_objectness_loss = 0
        total_class_loss = 0
        total_box_loss = 0
        total_height_loss = 0
        
        with self.create_progress(f"Epoch {epoch} Training") as progress:
            task = progress.add_task(f"Epoch {epoch}", total=len(self.train_loader))
            
            for batch_idx, ((points, images), objects_list) in enumerate(self.train_loader):
                batch_size = len(points)

                points = points.to(self.dtype).to(self.device).detach()
                images = images.to(self.dtype).to(self.device).detach()

                self.optimizer.zero_grad()
                outputs = self.model(points, images)

                num_clusters = len(outputs) // self.detection_head_num
                predictions_by_head = []
                
                for i in range(self.detection_head_num):
                    head_outputs = outputs[i*num_clusters:(i+1)*num_clusters]
                    predictions_by_head.append(head_outputs)

                batch_loss = torch.tensor([0], device=self.device, dtype=self.dtype)

                batch_objectness_loss = 0.0
                batch_class_loss = 0.0
                batch_box_loss = 0.0
                batch_height_loss = 0.0
                
                for b in range(batch_size):
                    gt_objects = objects_list[b]
                    
                    if len(gt_objects) == 0:
                        for head_idx, head_preds in enumerate(predictions_by_head):
                            for cluster_idx in range(num_clusters):
                                objectness = head_preds[cluster_idx][0][b]
                                objectness_loss = self.objectness_criterion(objectness, torch.zeros_like(objectness))
                                batch_loss += objectness_loss
                                batch_objectness_loss += objectness_loss.item()
                        continue
                    assigned_gt_masks = torch.zeros(self.detection_head_num, num_clusters, device=self.device)
                    
                    for (class_idx, xz_tensor, y_bounds) in gt_objects:
                        xz_tensor = xz_tensor.to(self.dtype).to(self.device)
                        y_bounds = y_bounds.to(self.dtype).to(self.device)
                        
                        best_iou = -1
                        best_head_idx = -1
                        best_cluster_idx = -1

                        for head_idx, head_preds in enumerate(predictions_by_head):
                            for cluster_idx in range(num_clusters):
                                objectness, class_pred, xz_pred, y_pred = [
                                    tensor[b] 
                                    for tensor in head_preds[cluster_idx]
                                ]
                                iou = box3d_iou(xz_pred, y_pred, xz_tensor, y_bounds)
                                
                                if iou > best_iou:
                                    best_iou = iou
                                    best_head_idx = head_idx
                                    best_cluster_idx = cluster_idx
                        assigned_gt_masks[best_head_idx, best_cluster_idx] = 1

                        objectness, class_pred, xz_pred, y_pred = [
                            tensor[b] 
                            for tensor in predictions_by_head[best_head_idx][best_cluster_idx]
                        ]

                        objectness_loss = self.objectness_criterion(
                            objectness, torch.ones_like(objectness)
                        )
                        
                        class_loss = self.class_criterion(
                            class_pred.unsqueeze(0), torch.tensor([class_idx], device=self.device)
                        )

                        box_loss = self.box_criterion(
                            xz_pred.reshape(-1), xz_tensor.reshape(-1)
                        ).mean()

                        height_loss = self.height_criterion(
                            y_pred, y_bounds
                        ).mean()

                        total_pred_loss = (
                            self.objectness_weight * objectness_loss +
                            self.class_weight * class_loss +
                            self.box_weight * box_loss +
                            self.height_weight * height_loss
                        )
                        
                        batch_loss += total_pred_loss
                        batch_objectness_loss += objectness_loss.item()
                        batch_class_loss += class_loss.item()
                        batch_box_loss += box_loss.item()
                        batch_height_loss += height_loss.item()

                    negative_mask = (assigned_gt_masks == 0)
                    num_positive = assigned_gt_masks.sum().item()
                    num_negative = int(min(num_positive * self.neg_pos_ratio, negative_mask.sum().item()))
                    
                    if num_negative > 0 and num_positive > 0:
                        neg_objectness_scores = []
                        for head_idx in range(self.detection_head_num):
                            for cluster_idx in range(num_clusters):
                                if negative_mask[head_idx, cluster_idx]:
                                    objectness = predictions_by_head[head_idx][cluster_idx][0][b]
                                    neg_objectness_scores.append((objectness.item(), head_idx, cluster_idx))

                        neg_objectness_scores.sort(reverse=True)
                        hard_negatives = neg_objectness_scores[:num_negative]

                        for _, head_idx, cluster_idx in hard_negatives:
                            objectness = predictions_by_head[head_idx][cluster_idx][0][b]
                            objectness_loss = self.objectness_criterion(
                                objectness, torch.zeros_like(objectness)
                            )
                            batch_loss += self.objectness_weight * objectness_loss
                            batch_objectness_loss += objectness_loss.item()

                batch_loss = batch_loss / batch_size

                batch_loss.backward()
                self.optimizer.step()

                total_loss += batch_loss.item()
                total_objectness_loss += batch_objectness_loss / batch_size
                total_class_loss += batch_class_loss / batch_size
                total_box_loss += batch_box_loss / batch_size
                total_height_loss += batch_height_loss / batch_size

                progress.update(
                    task, 
                    advance=1, 
                    description=f"Epoch {epoch} [loss={total_loss/(batch_idx+1):.4f}]"
                )

                if batch_idx % 10 == 0 or batch_idx == len(self.train_loader) - 1:
                    avg_loss = total_loss / (batch_idx + 1)
                    avg_obj_loss = total_objectness_loss / (batch_idx + 1)
                    avg_cls_loss = total_class_loss / (batch_idx + 1)
                    avg_box_loss = total_box_loss / (batch_idx + 1)
                    avg_height_loss = total_height_loss / (batch_idx + 1)
                    
                    progress.console.print(
                        f"[cyan]Loss: {avg_loss:.4f} | "
                        f"Obj: {avg_obj_loss:.4f} | "
                        f"Cls: {avg_cls_loss:.4f} | "
                        f"Box: {avg_box_loss:.4f} | "
                        f"Height: {avg_height_loss:.4f}[/cyan]"
                    )

        return {
            'loss': total_loss / len(self.train_loader),
            'objectness_loss': total_objectness_loss / len(self.train_loader),
            'class_loss': total_class_loss / len(self.train_loader),
            'box_loss': total_box_loss / len(self.train_loader),
            'height_loss': total_height_loss / len(self.train_loader)
        }
    
    def train(self, num_epochs, save_path=None, scheduler=None):
        train_losses = []
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            if scheduler:
                scheduler.step()
            
            table = Table(title=f"Epoch {epoch} Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Train", style="green")
            
            table.add_row("Loss", f"{train_loss['loss']:.4f}")
            table.add_row("Objectness Loss", f"{train_loss['objectness_loss']:.4f}")
            table.add_row("Class Loss", f"{train_loss['class_loss']:.4f}")
            table.add_row("Box Loss", f"{train_loss['box_loss']:.4f}")
            table.add_row("Height Loss", f"{train_loss['height_loss']:.4f}")
            
            self.console.print(table)
            
            if save_path and epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                }, f"{save_path}_epoch{epoch}.pt")
                self.console.print(f"[bold green]Saved model checkpoint at {save_path}_epoch{epoch}.pt[/bold green]")
        
        return train_losses
