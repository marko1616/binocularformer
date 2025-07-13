import torch
import torch.nn as nn
import torch.amp as amp

from contextlib import nullcontext

from torch.utils.tensorboard import SummaryWriter

from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from rich.table import Table

from .utils import box3d_iou, box3d_center, is_point_inside_box, unwrap_ddp

class BinocularFormerTrainer:
    def __init__(
        self,
        model,
        train_loader,
        optimizer,
        scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        objectness_weight=1.0,
        class_weight=1.0,
        box_weight=1.0,
        height_weight=1.0,
        accumulation_steps=2,
        use_amp=True,
        init_scale=256.0,
        rank=None
    ):
        self.model = model.to(dtype).to(device)
        self.dtype = dtype
        self.device = device
        self.objectness_weight = objectness_weight
        self.class_weight = class_weight
        self.box_weight = box_weight
        self.height_weight = height_weight
        self.console = Console()

        self.train_loader = train_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.use_amp = use_amp
        self.scaler = amp.GradScaler(init_scale=init_scale) if use_amp else None

        self.objectness_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.class_criterion = nn.CrossEntropyLoss(reduction='none')
        self.box_criterion = nn.SmoothL1Loss(reduction='none')
        self.height_criterion = nn.SmoothL1Loss(reduction='none')

        self.class_num = unwrap_ddp(model).class_num
        self.detection_head_num = unwrap_ddp(model).detection_head_num

        self.accumulation_steps = accumulation_steps

        self.writer = SummaryWriter(log_dir="runs/binocular_former")

        self.rank = rank

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
        optim_step_obj_count = 0
        optim_step_loss = 0
        optim_step_objectness_loss = 0
        optim_step_class_loss = 0
        optim_step_box_loss = 0
        optim_step_height_loss = 0

        epoch_obj_count = 0
        epoch_loss = 0
        epoch_objectness_loss = 0
        epoch_class_loss = 0
        epoch_box_loss = 0
        epoch_height_loss = 0

        accumulation_steps = self.accumulation_steps
        accumulation_counter = 0

        progress_bar_context = self.create_progress(f"Epoch {epoch} Training") if not self.rank else nullcontext()
        with progress_bar_context as progress:
            if not self.rank:
                task = progress.add_task(f"Epoch {epoch}", total=len(self.train_loader))

            for batch_idx, ((points, images), objects_list) in enumerate(self.train_loader):
                batch_size = len(points)

                points = points.to(self.dtype).to(self.device).detach()
                images = images.to(self.dtype).to(self.device).detach()

                if accumulation_counter == 0:
                    self.optimizer.zero_grad()
                
                loss_compute_context = torch.autocast(device_type="cuda") if self.use_amp else nullcontext()
                with loss_compute_context:
                    outputs, cluster_positions = self.model(points, images)

                    num_clusters = len(outputs) // self.detection_head_num
                    predictions_by_head = []

                    for i in range(self.detection_head_num):
                        head_outputs = outputs[i::self.detection_head_num]
                        predictions_by_head.append(head_outputs)

                    batch_loss = torch.tensor([0], device=self.device, dtype=self.dtype)

                    batch_objectness_loss = 0.0
                    batch_class_loss = 0.0
                    batch_box_loss = 0.0
                    batch_height_loss = 0.0

                    for b in range(batch_size):
                        gt_objects = objects_list[b]
                        optim_step_obj_count += len(gt_objects)
                        epoch_obj_count += len(gt_objects)
                        
                        # Track which heads have been assigned to a ground truth object
                        assigned_heads = set()
                        
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
                                if head_idx in assigned_heads:
                                    continue
                                    
                                for cluster_idx in range(num_clusters):
                                    objectness, class_pred, xz_pred, y_pred = [
                                        tensor[b]
                                        for tensor in head_preds[cluster_idx]
                                    ]
                                    head_centroid = cluster_positions[b][cluster_idx]
                                    iou = box3d_iou(xz_pred, y_pred, xz_tensor, y_bounds)

                                    if iou > best_iou and is_point_inside_box(xz_tensor, y_bounds, head_centroid):
                                        best_iou = iou
                                        best_head_idx = head_idx
                                        best_cluster_idx = cluster_idx

                            # If no head was found with points inside the box, find the closest unassigned head
                            if best_head_idx == -1:
                                target_center = box3d_center(xz_tensor, y_bounds)
                                min_distance = float('inf')
                                
                                for head_idx, head_preds in enumerate(predictions_by_head):
                                    if head_idx in assigned_heads:
                                        continue
                                        
                                    for cluster_idx in range(num_clusters):
                                        objectness, class_pred, xz_pred, y_pred = [
                                            tensor[b]
                                            for tensor in head_preds[cluster_idx]
                                        ]
                                        head_centroid = cluster_positions[b][cluster_idx]
                                        iou = box3d_iou(xz_pred, y_pred, xz_tensor, y_bounds)

                                        distance = torch.norm(head_centroid - target_center)

                                        if (distance < min_distance) or (distance == min_distance and iou > best_iou):
                                            min_distance = distance
                                            best_iou = iou
                                            best_head_idx = head_idx
                                            best_cluster_idx = cluster_idx

                            if best_head_idx == -1:
                                continue

                            assigned_heads.add(best_head_idx)
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

                        # For unassigned heads, make them predict no object
                        for head_idx in range(self.detection_head_num):
                            if head_idx not in assigned_heads:
                                for cluster_idx in range(num_clusters):
                                    objectness = predictions_by_head[head_idx][cluster_idx][0][b]
                                    objectness_loss = self.objectness_criterion(
                                        objectness, torch.zeros_like(objectness)
                                    )
                                    batch_loss += self.objectness_weight * objectness_loss
                                    batch_objectness_loss += objectness_loss.item()
                                if self.rank is not None:
                                    # Add fake loss for DDP at cluster 0
                                    objectness, class_pred, xz_pred, y_pred = [
                                        tensor[b]
                                        for tensor in predictions_by_head[head_idx][0]
                                    ]
                                    batch_loss += 0.0 * class_pred.sum() * xz_pred.sum() * y_pred.sum()
                            else:
                                head_clusters_assigned = assigned_gt_masks[head_idx].nonzero().squeeze(-1)

                                # If there's only one cluster assigned for this head
                                if len(head_clusters_assigned.shape) == 0:
                                    head_clusters_assigned = head_clusters_assigned.unsqueeze(0)

                                # Get negative samples only from this head
                                for cluster_idx in range(num_clusters):
                                    if cluster_idx not in head_clusters_assigned:
                                        objectness = predictions_by_head[head_idx][cluster_idx][0][b]
                                        objectness_loss = self.objectness_criterion(
                                            objectness, torch.zeros_like(objectness)
                                        )
                                        batch_loss += self.objectness_weight * objectness_loss
                                        batch_objectness_loss += objectness_loss.item()

                    loss = batch_loss / (batch_size * accumulation_steps)
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulation_counter += 1
                if accumulation_counter % accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    accumulation_counter = 0

                optim_step_loss += batch_loss.item()
                optim_step_objectness_loss += batch_objectness_loss
                optim_step_class_loss += batch_class_loss
                optim_step_box_loss += batch_box_loss
                optim_step_height_loss += batch_height_loss

                epoch_loss += batch_loss.item()
                epoch_objectness_loss += batch_objectness_loss
                epoch_class_loss += batch_class_loss
                epoch_box_loss += batch_box_loss
                epoch_height_loss += batch_height_loss

                if not self.rank:
                    progress.update(
                        task,
                        advance=1,
                        description=f"Epoch {epoch} [loss={optim_step_loss/(batch_idx+1):.4f}]"
                    )

                if self.rank is None:
                    self.writer.add_scalar('Obj Avg Loss/Step/Total', optim_step_loss / (batch_idx + 1), epoch * len(self.train_loader) + batch_idx)
                    self.writer.add_scalar('Obj Avg Loss/Step/Objectness', optim_step_objectness_loss / (batch_idx + 1), epoch * len(self.train_loader) + batch_idx)
                    self.writer.add_scalar('Obj Avg Loss/Step/Class', optim_step_class_loss / (batch_idx + 1), epoch * len(self.train_loader) + batch_idx)
                    self.writer.add_scalar('Obj Avg Loss/Step/Box', optim_step_box_loss / (batch_idx + 1), epoch * len(self.train_loader) + batch_idx)
                    self.writer.add_scalar('Obj Avg Loss/Step/Height', optim_step_height_loss / (batch_idx + 1), epoch * len(self.train_loader) + batch_idx)                
                else:
                    self.writer.add_scalar(f'Obj Avg Loss/Rank{self.rank}/Step/Total', optim_step_loss / (batch_idx + 1), epoch * len(self.train_loader) + batch_idx)
                    self.writer.add_scalar(f'Obj Avg Loss/Rank{self.rank}/Step/Objectness', optim_step_objectness_loss / (batch_idx + 1), epoch * len(self.train_loader) + batch_idx)
                    self.writer.add_scalar(f'Obj Avg Loss/Rank{self.rank}/Step/Class', optim_step_class_loss / (batch_idx + 1), epoch * len(self.train_loader) + batch_idx)
                    self.writer.add_scalar(f'Obj Avg Loss/Rank{self.rank}/Step/Box', optim_step_box_loss / (batch_idx + 1), epoch * len(self.train_loader) + batch_idx)
                    self.writer.add_scalar(f'Obj Avg Loss/Rank{self.rank}/Step/Height', optim_step_height_loss / (batch_idx + 1), epoch * len(self.train_loader) + batch_idx)    

                if batch_idx % 5 == 0:
                    avg_loss = optim_step_loss / optim_step_obj_count
                    avg_obj_loss = optim_step_objectness_loss / optim_step_obj_count
                    avg_cls_loss = optim_step_class_loss / optim_step_obj_count
                    avg_box_loss = optim_step_box_loss / optim_step_obj_count
                    avg_height_loss = optim_step_height_loss / optim_step_obj_count

                    optim_step_obj_count = 0
                    optim_step_loss = 0
                    optim_step_objectness_loss = 0
                    optim_step_class_loss = 0
                    optim_step_box_loss = 0
                    optim_step_height_loss = 0

                    if not self.rank:
                        progress.console.print(
                            f"[cyan]Optim Loss/obj: {avg_loss:.4f} | "
                            f"Optim Loss.Obj/obj: {avg_obj_loss:.4f} | "
                            f"Optim Loss.Cls/obj: {avg_cls_loss:.4f} | "
                            f"Optim Loss.Box/obj: {avg_box_loss:.4f} | "
                            f"Optim Loss.Height/obj: {avg_height_loss:.4f} |"
                        )

        return {
            'loss': epoch_loss / epoch_obj_count,
            'objectness_loss': epoch_objectness_loss / epoch_obj_count,
            'class_loss': epoch_class_loss / epoch_obj_count,
            'box_loss': epoch_box_loss / epoch_obj_count,
            'height_loss': epoch_height_loss / epoch_obj_count
        }

    def train(self, num_epochs, start_epoch=1, save_path=None):
        train_losses = []

        for epoch in range(start_epoch, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)

            if self.scheduler:
                self.scheduler.step()

            if self.rank is None:
                self.writer.add_scalar('Obj Avg Loss/Epoch/Total', train_loss['loss'], epoch)
                self.writer.add_scalar('Obj Avg Loss/Epoch/Objectness', train_loss['objectness_loss'], epoch)
                self.writer.add_scalar('Obj Avg Loss/Epoch/Class', train_loss['class_loss'], epoch)
                self.writer.add_scalar('Obj Avg Loss/Epoch/Box', train_loss['box_loss'], epoch)
                self.writer.add_scalar('Obj Avg Loss/Epoch/Height', train_loss['height_loss'], epoch)
            else:
                self.writer.add_scalar(f'Obj Avg Loss/Rank{self.rank}/Epoch/Total', train_loss['loss'], epoch)
                self.writer.add_scalar(f'Obj Avg Loss/Rank{self.rank}/Epoch/Objectness', train_loss['objectness_loss'], epoch)
                self.writer.add_scalar(f'Obj Avg Loss/Rank{self.rank}/Epoch/Class', train_loss['class_loss'], epoch)
                self.writer.add_scalar(f'Obj Avg Loss/Rank{self.rank}/Epoch/Box', train_loss['box_loss'], epoch)
                self.writer.add_scalar(f'Obj Avg Loss/Rank{self.rank}/Epoch/Height', train_loss['height_loss'], epoch)

            if not self.rank:
                for i, group in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(f'LR/group_{i}', group['lr'], epoch)

                table = Table(title=f"Epoch {epoch} Summary")
                table.add_column("Metric", style="cyan")
                table.add_column("Train", style="green")

                table.add_row("Loss", f"{train_loss['loss']:.4f}")
                table.add_row("Objectness Loss", f"{train_loss['objectness_loss']:.4f}")
                table.add_row("Class Loss", f"{train_loss['class_loss']:.4f}")
                table.add_row("Box Loss", f"{train_loss['box_loss']:.4f}")
                table.add_row("Height Loss", f"{train_loss['height_loss']:.4f}")

                self.console.print(table)

                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                    }, f"{save_path}_epoch{epoch}.pt")
                    self.console.print(f"[bold green]Saved model checkpoint at {save_path}_epoch{epoch}.pt[/bold green]")

        return train_losses
