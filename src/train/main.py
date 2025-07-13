import argparse
import torch
import os

import torch.optim as optim
import torch.distributed as dist

from rich import print

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from ..model.backbone import BinocularFormer
from .dataset import SunRgbdDataset, sun_rgbd_collector
from .trainer import BinocularFormerTrainer

def load_checkpoint(model, path):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {path}")
    else:
        print(f"Checkpoint path {path} not found. Starting from scratch.")
    return model

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"[DDP] rank:{rank} setup")

def main(rank, world_size, args):
    if rank is not None:
        setup_ddp(rank, world_size)

    train_dataset = SunRgbdDataset(norm_param=(
        torch.tensor([0.4905, 0.4565, 0.4314], dtype=torch.float64),
        torch.tensor([0.2818, 0.2895, 0.2946], dtype=torch.float64)
    ),verbose=not rank)

    model = BinocularFormer(
        model_dim=512,
        cnn_layer_num=8,
        cnn_kernel_size=3,
        cnn_padding=1,
        cluster_sizes=[28, 28, 15, 10],
        encoder_layer_num=2,
        encoder_head_num=8,
        encoder_ffd_dim=512,
        detection_head_num=4,
        class_num=len(train_dataset.class_name),
        dropout=0.1
    )

    if rank is None:
        model.to("cuda")
    else:
        model.to(rank)
        model = DDP(model, device_ids=[rank])

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(lambda grad, name=name:
                print(f"Parameter {name} has NaN gradient") if torch.isnan(grad).any() else grad)
            param.register_hook(lambda grad, name=name:
                print(f"Parameter {name} has INF gradient") if torch.isinf(grad).any() else grad)

    if args.model_name_or_path:
        model = load_checkpoint(model, args.model_name_or_path)

    optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
        )
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=sun_rgbd_collector,
                              sampler=RandomSampler(train_dataset) if rank is None else DistributedSampler(train_dataset, num_replicas=world_size, rank=rank))

    trainer = BinocularFormerTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=rank if rank else args.device,
        dtype=torch.float32,
        objectness_weight=args.objectness_weight,
        class_weight=args.class_weight,
        box_weight=args.box_weight,
        height_weight=args.height_weight,
        accumulation_steps=args.accumulation_steps,
        use_amp=args.use_amp,
        init_scale=args.init_scale,
        rank=rank
    )

    _ = trainer.train(
        num_epochs=args.epochs,
        start_epoch=args.start_epoch,
        save_path="checkpoints/binocular_former",
    )
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BinocularFormer on SunRGBD")

    parser.add_argument('--model_name_or_path', type=str, default=None,
                        help="Path to a checkpoint to resume training from")
    parser.add_argument('--start_epoch', type=int, default=0,
                        help="Epoch to start training from (used when resuming)")
    parser.add_argument('--epochs', type=int, default=16,
                        help="Total number of epochs to train")

    parser.add_argument('--batch_size', type=int, default=1, help="Training batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument('--objectness_weight', type=float, default=1.0, help="Loss weight for objectness")
    parser.add_argument('--class_weight', type=float, default=1.0, help="Loss weight for class prediction")
    parser.add_argument('--box_weight', type=float, default=1.0, help="Loss weight for bounding box regression")
    parser.add_argument('--height_weight', type=float, default=1.0, help="Loss weight for height regression")
    parser.add_argument('--accumulation_steps', type=int, default=5, help="Gradient accumulation steps")

    parser.add_argument('--use_amp', type=bool, default=True, help="Use automatic mixed precision train")
    parser.add_argument('--init_scale', type=float, default=256.0, help="AMP Scaler init scale")

    parser.add_argument('--use_gpus', type=bool, default=False, help="Use DDP train")

    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")

    args = parser.parse_args()

    if args.use_gpus:
        import torch.multiprocessing as mp
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
    else:
        main(None, None, args)
