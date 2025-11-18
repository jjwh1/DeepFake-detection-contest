import os
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
from torchvision.utils import make_grid
from PIL import Image

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm  # ✅ 추가됨

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def accuracy_and_f1_from_logits(logits: torch.Tensor, targets: torch.Tensor):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    acc = correct / targets.numel()

    num_classes = logits.size(1)
    f1_sum, cls_count = 0.0, 0
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1_c = 2 * precision * recall / (precision + recall + 1e-12)
        f1_sum += f1_c
        cls_count += 1
    f1_macro = f1_sum / max(cls_count, 1)
    return acc, f1_macro


# ---------------------------
# Custom Dataset (real=0, fake=1)
# ---------------------------
class SimpleImageDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.samples = []

        if real_dir:
            for fname in os.listdir(real_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(real_dir, fname), 0))  # real = 0

        if fake_dir:
            for fname in os.listdir(fake_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(fake_dir, fname), 1))  # fake = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------
# Dataloader Builder
# ---------------------------
def create_dataloaders(real_train_dir, fake_train_dir, real_val_dir, fake_val_dir,
                       img_size=224, batch_size=8, workers=2):

    train_tf = T.Compose([
        T.Resize(img_size),
        T.RandomHorizontalFlip(p=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    val_tf = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    train_ds = SimpleImageDataset(real_train_dir, fake_train_dir, transform=train_tf)
    val_ds   = SimpleImageDataset(real_val_dir, fake_val_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)

    classes = ["real(0)", "fake(1)"]
    return train_loader, val_loader, classes


# ---------------------------
# Model
# ---------------------------
def build_model(num_classes=2):
    model = timm.create_model('coat_lite_small', pretrained=True, num_classes=num_classes)
    return model

def save_checkpoint(state, is_best, outdir, filename='last.pth'):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(state, outdir / filename)
    if is_best:
        torch.save(state, outdir / 'best.pth')

def format_seconds(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------
# Train / Validate
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, writer, log_images=False):
    model.train()
    running_loss, running_acc, running_f1 = 0.0, 0.0, 0.0
    n_batches = len(loader)
    t0 = time.time()

    for step, (imgs, targets) in enumerate(tqdm(loader, desc=f"Train [{epoch+1}]")):
        imgs, targets = imgs.to(device, non_blocking=True), torch.tensor(targets).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            acc, f1 = accuracy_and_f1_from_logits(logits, targets)

        running_loss += loss.item()
        running_acc  += acc
        running_f1   += f1

        if log_images and step == 0:
            grid = make_grid(
                imgs[:8].cpu() * torch.tensor(IMAGENET_DEFAULT_STD).view(3,1,1)
                + torch.tensor(IMAGENET_DEFAULT_MEAN).view(3,1,1),
                nrow=4,
                normalize=False
            )
            writer.add_image("train/sample_images", grid, global_step=epoch)

    epoch_loss = running_loss / n_batches
    epoch_acc  = running_acc / n_batches
    epoch_f1   = running_f1 / n_batches

    elapsed = format_seconds(time.time() - t0)
    return epoch_loss, epoch_acc, epoch_f1, elapsed


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer, split='val'):
    model.eval()
    running_loss, running_acc, running_f1 = 0.0, 0.0, 0.0
    n_batches = len(loader)
    t0 = time.time()

    for imgs, targets in tqdm(loader, desc=f"Valid [{epoch+1}]"):
        imgs, targets = imgs.to(device, non_blocking=True), torch.tensor(targets).to(device)
        logits = model(imgs)
        loss = criterion(logits, targets)
        acc, f1 = accuracy_and_f1_from_logits(logits, targets)
        running_loss += loss.item()
        running_acc  += acc
        running_f1   += f1

    epoch_loss = running_loss / n_batches
    epoch_acc  = running_acc / n_batches
    epoch_f1   = running_f1 / n_batches
    elapsed = format_seconds(time.time() - t0)
    print("=== Prediction Distribution Check ===")
    print("Pred counts:", {0: 0, 1: 0})
    print("Target counts:", {0: 0, 1: 0})

    pred_cnt = {0: 0, 1: 0}
    tgt_cnt = {0: 0, 1: 0}

    for imgs, targets in tqdm(loader, desc="Check Dist"):
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().tolist()

        for p in preds:
            pred_cnt[p] += 1
        for t in targets.tolist():
            tgt_cnt[t] += 1

    print("Pred counts:", pred_cnt)
    print("Target counts:", tgt_cnt)

    return epoch_loss, epoch_acc, epoch_f1, elapsed


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="CoaT-Lite Small Deepfake Detection")

    # 🔥 여기만 완전히 변경됨
    parser.add_argument("--real_train_dir", type=str, default="/content/dataset/Real-img_train")
    parser.add_argument("--real_val_dir",   type=str, default="/content/dataset/Real-img_valid")

    parser.add_argument("--fake_train_dir", type=str, default="/content/dataset/Image_train")
    parser.add_argument("--fake_val_dir",   type=str, default="/content/dataset/Image_valid")

    parser.add_argument("--outdir",    type=str, default="/content/drive/MyDrive/DeepFake_detection/result/1")
    parser.add_argument("--epochs",    type=int, default=500)
    parser.add_argument("--batch_size",type=int, default=8)
    parser.add_argument("--img_size",  type=int, default=224)
    parser.add_argument("--lr",        type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    board_dir = os.path.join(args.outdir, "board")
    os.makedirs(board_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=board_dir)

    # Dataset
    train_loader, val_loader, classes = create_dataloaders(
        args.real_train_dir, args.fake_train_dir,
        args.real_val_dir, args.fake_val_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.num_workers
    )
    print(f"[Info] Classes: {classes}")

    # ============================================
    # 🔥 Train / Validation 데이터 분포 체크
    # ============================================

    print("========== DATASET DISTRIBUTION CHECK ==========")

    # Train split
    train_labels = [label for _, label in train_loader.dataset.samples]
    print("[Train]")
    print("  real(0) =", train_labels.count(0))
    print("  fake(1) =", train_labels.count(1))
    print("  total   =", len(train_labels))

    # Validation split
    val_labels = [label for _, label in val_loader.dataset.samples]
    print("[Validation]")
    print("  real(0) =", val_labels.count(0))
    print("  fake(1) =", val_labels.count(1))
    print("  total   =", len(val_labels))

    print("=================================================")

    # Model
    model = build_model(num_classes=len(classes))
    model.to(device)

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_epochs = args.epochs
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=total_epochs - args.warmup_epochs,
                                  eta_min=args.lr * 0.01)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_f1 = -1.0

    def warmup_lr(ep):
        return max(1e-8, args.lr * (ep + 1) / max(1, args.warmup_epochs))

    for epoch in range(total_epochs):
        if epoch < args.warmup_epochs:
            for g in optimizer.param_groups:
                g["lr"] = warmup_lr(epoch)
        else:
            scheduler.step()

        tr_loss, tr_acc, tr_f1, tr_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, writer, log_images=(epoch==0)
        )
        writer.add_scalar("train/loss", tr_loss, epoch)
        writer.add_scalar("train/acc", tr_acc, epoch)
        writer.add_scalar("train/f1", tr_f1, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        va_loss, va_acc, va_f1, va_time = validate(
            model, val_loader, criterion, device, epoch, writer, split='val'
        )
        writer.add_scalar("val/loss", va_loss, epoch)
        writer.add_scalar("val/acc", va_acc, epoch)
        writer.add_scalar("val/f1", va_f1, epoch)

        print(f"[Epoch {epoch+1:03d}/{total_epochs:03d}] "
              f"train: loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} ({tr_time}) | "
              f"val: loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} ({va_time}) | "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        is_best = va_f1 > best_f1
        if is_best:
            best_f1 = va_f1

        # -----------------------------
        # 🔥 10 epoch마다 체크포인트 저장
        # -----------------------------
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_f1": best_f1,
                "classes": classes,
                "args": vars(args),
            }, is_best=False, outdir=args.outdir, filename=f'epoch_{epoch + 1}.pth')

        # -----------------------------
        # 🔥 best model 저장
        # -----------------------------
        if is_best:
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_f1": best_f1,
                "classes": classes,
                "args": vars(args),
            }, is_best=True, outdir=args.outdir, filename='best.pth')

    writer.close()
    print(f"[Done] Best val F1: {best_f1:.4f} | Logs & checkpoints at: {args.outdir}")


if __name__ == "__main__":
    main()
