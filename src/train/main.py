import torch

from ..model.backbone import BinocularFormer
from .dataset import SunRgbdDataset
from .trainer import BinocularFormerTrainer

if __name__ == "__main__":
    train_dataset = SunRgbdDataset()

    model = BinocularFormer(
        model_dim=512,
        cnn_layer_num=4,
        cnn_kernel_size=3,
        cnn_padding=1,
        cluster_sizes=[28, 28, 15, 10],
        encoder_layer_num=2,
        encoder_head_num=4,
        encoder_ffd_dim=512,
        detection_head_num=4,
        class_num=len(train_dataset.class_name),
        dropout=0.1
    )

    for name, param in model.named_parameters():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: 
                    print(f"Parameter {name} has NaN gradient") if torch.isnan(grad).any() else grad)

                param.register_hook(lambda grad, name=name: 
                    print(f"Parameter {name} has INF gradient") if torch.isinf(grad).any() else grad)

    trainer = BinocularFormerTrainer(
        model=model,
        train_dataset=train_dataset,
        batch_size=2,
        lr=1e-6,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        objectness_weight=1.0,
        class_weight=1.0,
        box_weight=1.0,
        height_weight=1.0,
        neg_pos_ratio=3.0
    )

    scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=10, gamma=0.5)
    train_losses, val_losses = trainer.train(
        num_epochs=50,
        save_path="checkpoints/binocular_former.pt",
        scheduler=scheduler
    )
