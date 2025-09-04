import argparse
import math
import os
from pathlib import Path
from typing import Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import torchvision
from torchvision import transforms

from vit import ViT

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='cifar10')
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument('--model', type=str, default='ViT')
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--warmup_epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--opt', type=str, default='adamw')
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--output', type=str, default="./output/vit_c10")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")

    return p.parse_args()
    
def set_seed(seed: int):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    CIFAR-10: 32 * 32; We upsample to 224 to match ViT-B/16 image size
    We resize the image size and normalize to [-1,1]    
    
    Args:
        img_size: image size of the dataset 
    
    Returns:
        train_tf: transform of train set
        eval_tf: transform of eval set
    """
    
    train_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return train_tf, eval_tf

def build_dataloaders(dataset: str, root: str, img_size: int, bs: int, nw: int):
    """
    
    """
    
    train_tf, eval_tf = build_transforms(img_size)
    assert dataset == "cifar10", "Only CIFAR-10 wired here (extend as needed)."
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    val_set   = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=eval_tf)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    num_classes = 10
    return train_loader, val_loader, num_classes
    
def build_model(num_classes: int, img_size: int):
    # Match your ViT signature
    model = ViT(
        image_size=img_size,
        patch_size=16,          # adjust if you want a different patch
        num_classes=num_classes,
        dim=768,                # ViT-B dim
        depth=12,
        heads=12,
        mlp_dim=3072,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    )
    return model

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total, top1, top5 = 0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        # CrossEntropyLoss expects raw logits; use F.cross_entropy for loss elsewhere. :contentReference[oaicite:5]{index=5}
        pred = torch.softmax(logits, dim=-1)
        total += y.size(0)
        top1 += (pred.argmax(dim=-1) == y).sum().item()
        # top-5
        _, top5_idx = pred.topk(5, dim=-1)
        top5 += (top5_idx == y.view(-1, 1)).any(dim=1).sum().item()
    return top1 / total * 100.0, top5 / total * 100.0

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_loader, val_loader, num_classes = build_dataloaders(
        args.dataset, args.data_root, args.img_size, args.batch_size, args.num_workers
    )

    # model
    model = build_model(num_classes, args.img_size).to(device)

    # opt
    assert args.opt.lower() == "adamw"
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # sched: Linear warmup (epochs) → Cosine annealing (epochs - warmup)
    # PyTorch built-ins: SequentialLR + LinearLR + CosineAnnealingLR. :contentReference[oaicite:6]{index=6}
    warmup_epochs = args.warmup_epochs
    cosine_epochs = max(args.epochs - warmup_epochs, 1)
    sched_warm = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs)
    sched_cos = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=0.0)
    scheduler = SequentialLR(optimizer, schedulers=[sched_warm, sched_cos], milestones=[warmup_epochs])

    # loss
    criterion = nn.CrossEntropyLoss()  # standard multi-class CE: use one hot to turn groud truth to prob dist
    
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    best_top1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(x) # B, num_classes
                assert logits.shape == (x.size(0), num_classes), f"Expected logits shape {(x.size(0), num_classes)}, got {logits.shape}"
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)

        # epoch schedulers step once per epoch (LinearLR/CosineAnnealingLR are epoch-based by default)
        scheduler.step()

        avg_loss = running_loss / len(train_loader.dataset)
        top1, top5 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1:03d}/{args.epochs} | loss {avg_loss:.4f} | top1 {top1:.2f} | top5 {top5:.2f}")

        # save
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "top1": top1,
        }
        torch.save(ckpt, Path(args.output) / "last.pth")
        if top1 > best_top1:
            best_top1 = top1
            torch.save(ckpt, Path(args.output) / "best.pth")

    print(f"Best top1: {best_top1:.2f}")


if __name__ == "__main__":
    main()
