import argparse
import os
from datetime import datetime
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dataset.av2_binary_seg_dataset import AV2BinarySegDataset
from utils.config_utils import load_config_with_inheritance
from utils.dataloader_utils import infinite_dataloader
from config.config import correct_datatype


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg) -> AV2BinarySegDataset:
    return AV2BinarySegDataset(
        root_dir=cfg.root_dir,
        flow_data_path=cfg.flow_data_path,
        expected_camera_shape=tuple(cfg.expected_camera_shape),
        subsequence_length=cfg.subsequence_length,
        sliding_window_step_size=cfg.sliding_window_step_size,
        with_rgb=cfg.with_rgb,
        range_crop_type=cfg.range_crop_type,
        point_size=cfg.point_size,
        with_ground=cfg.with_ground,
        use_gt_flow=cfg.use_gt_flow,
        eval_type=cfg.eval_type,
        load_flow=cfg.load_flow,
        load_boxes=cfg.load_boxes,
        min_instance_size=cfg.min_instance_size,
        cache_root=cfg.cache_root,
        log_subset=getattr(cfg, "log_subset", None),
        min_flow_threshold=cfg.min_flow_threshold,
        fg_class_ids=getattr(cfg, "fg_class_ids", None),
    )


def build_model(cfg, point_size: int, device: torch.device) -> torch.nn.Module:
    model_name = cfg.name.lower()
    num_classes = cfg.num_classes
    if model_name == "maskformer3d":
        from OGCModel.segnet_av2 import MaskFormer3D

        params = cfg.MaskFormer3D
        model = MaskFormer3D(
            n_slot=num_classes,
            n_point=point_size,
            point_feats_dim=3,
            use_xyz=True,
            n_transformer_layer=params.n_transformer_layer,
            transformer_embed_dim=params.transformer_embed_dim,
            scale=params.scale,
            transformer_input_pos_enc=params.transformer_input_pos_enc,
        )
    elif model_name == "ptv3":
        from model.ptv3_mask_predictor import PTV3MaskPredictor

        params = cfg.PTV3
        model = PTV3MaskPredictor(
            slot_num=num_classes,
            in_channels=params.in_channels,
            feat_dim=params.feat_dim,
            grid_size=params.grid_size,
            enable_flash=params.enable_flash,
            enable_rpe=params.enable_rpe,
            enc_depths=tuple(params.enc_depths),
            enc_channels=tuple(params.enc_channels),
            dec_depths=tuple(params.dec_depths),
            dec_channels=tuple(params.dec_channels),
        )
    else:
        raise ValueError(f"Unsupported model {cfg.name}")

    return model.to(device)


def model_log_probs(model, model_cfg_name: str, points: torch.Tensor) -> torch.Tensor:
    model_name = model_cfg_name.lower()
    if model_name == "maskformer3d":
        probs = model(points, points)
    else:
        probs = model({"point_cloud_first": points})

    if probs.ndim != 3:
        raise ValueError(f"Unexpected mask shape {probs.shape}")

    # Ensure [B, C, N]
    if probs.shape[1] == points.shape[1]:
        probs = probs.transpose(1, 2)

    probs = probs.clamp(min=1e-6)
    return torch.log(probs)


def ce_loss(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    b, c, n = log_probs.shape
    reshaped = log_probs.permute(0, 2, 1).reshape(-1, c)
    target = labels.reshape(-1)
    return F.nll_loss(reshaped, target)


def train_one_step(model, batch, optimizer, device, model_name, grad_clip: float | None) -> float:
    model.train()
    points = batch["points"].to(device)
    labels = batch["labels"].to(device)
    optimizer.zero_grad(set_to_none=True)
    log_probs = model_log_probs(model, model_name, points)
    loss = ce_loss(log_probs, labels)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, loader, device, model_name) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    intersections = torch.zeros(2, device=device)
    unions = torch.zeros(2, device=device)
    max_eval_steps = 10

    for i, batch in enumerate(loader):
        if i >= max_eval_steps:
            break
        points = batch["points"].to(device)
        labels = batch["labels"].to(device)
        log_probs = model_log_probs(model, model_name, points)
        loss = ce_loss(log_probs, labels)
        total_loss += loss.item()
        preds = log_probs.exp().argmax(dim=1)
        for cls in range(2):
            pred_mask = preds == cls
            gt_mask = labels == cls
            intersections[cls] += (pred_mask & gt_mask).sum()
            unions[cls] += (pred_mask | gt_mask).sum()

    valid = unions > 0
    if valid.any():
        miou = (intersections[valid] / unions[valid]).mean().item()
    else:
        miou = 1.0
    return total_loss / max(len(loader), 1), miou


def prepare_dataloaders(config):
    train_dataset = build_dataset(config.dataset.train)
    val_dataset = build_dataset(config.dataset.val)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_dataset.point_size


def main():
    parser = argparse.ArgumentParser(description="Binary FG/BG segmentation on AV2SceneFlowZoo")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args, unknown = parser.parse_known_args()

    config = load_config_with_inheritance(args.config)
    cli_cfg = OmegaConf.from_cli(unknown)
    config = OmegaConf.merge(config, cli_cfg)
    config = correct_datatype(config)

    log_dir = config.log.dir
    if log_dir == "":
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"../outputs/binary_fg_bg/{time_str}"
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    train_loader, val_loader, point_size = prepare_dataloaders(config)
    infinite_train_loader = infinite_dataloader(train_loader)
    model = build_model(config.model, point_size, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay
    )
    scheduler = None
    max_steps = getattr(config.training, "max_steps", 10000)
    if getattr(config.optim, "scheduler", None) is not None:
        sched_cfg = config.optim.scheduler
        if sched_cfg.name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_steps
            )

    best_miou = 0.0
    best_ckpt = os.path.join(log_dir, f"{config.model.name}_best.pth")
    print_interval = getattr(config.training, "print_interval", 10)
    eval_interval = getattr(config.training, "eval_interval", 100)

    step = 0
    running_loss = 0.0
    for batch in infinite_train_loader:
        step += 1
        if step > max_steps:
            break

        loss_value = train_one_step(
            model,
            batch,
            optimizer,
            device,
            config.model.name,
            getattr(config.training, "grad_clip", None),
        )
        running_loss += loss_value

        if scheduler is not None:
            scheduler.step()

        if step % eval_interval == 0:
            val_loss, val_miou = evaluate(model, val_loader, device, config.model.name)
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(
                    {
                        "step": step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_miou": val_miou,
                    },
                    best_ckpt,
                )
            steps_for_avg = step % print_interval if step % print_interval > 0 else print_interval
            avg_loss = running_loss / steps_for_avg
            running_loss = 0.0
            print(
                f"[Step {step}/{max_steps}] "
                f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_mIoU={val_miou:.4f} (best={best_miou:.4f})"
            )
        elif step % print_interval == 0:
            avg_loss = running_loss / print_interval
            running_loss = 0.0
            print(f"[Step {step}/{max_steps}] train_loss={avg_loss:.4f}")

    print(f"Training completed. Best mIoU: {best_miou:.4f}, checkpoint saved to {best_ckpt}")


if __name__ == "__main__":
    main()

