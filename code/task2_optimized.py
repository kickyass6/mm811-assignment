"""任务 2 实验：软二值预热 + 更宽卷积网络 + 验证集自适应学习率。"""

import argparse
import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parent
VAL_SIZE = 256
SEED = 42


class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        return torch.where(logits >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class OptimizedBinaryAutoencoder(nn.Module):
    def __init__(self, activation="silu"):
        super().__init__()
        activation_layer = nn.ReLU if activation == "relu" else nn.SiLU
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.GroupNorm(8, 64), activation_layer(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), activation_layer(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(16, 256), activation_layer(),
            nn.Flatten(), nn.Linear(256 * 7 * 7, 48),
            nn.BatchNorm1d(48, affine=False), nn.Unflatten(1, (3, 4, 4)),
        )
        self.decoder = nn.Sequential(
            nn.Flatten(), nn.Linear(48, 256 * 7 * 7), nn.Unflatten(1, (256, 7, 7)),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(16, 256), activation_layer(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GroupNorm(8, 128), activation_layer(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), activation_layer(),
            nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid(),
        )

    def encode(self, images, hard=True):
        logits = self.encoder(images)
        if not hard:
            # 与最终二值编码使用相同的 [-1, 1] 范围；等价于 tanh(5x)。
            return 2 * torch.sigmoid(10 * logits) - 1
        # 前向严格为 -1/+1；反向使用直通估计器把梯度传给编码器。
        return BinarizeSTE.apply(logits)

    def forward(self, images, hard=True):
        return self.decoder(self.encode(images, hard=hard))


@torch.no_grad()
def evaluate(model, loader, device):
    """始终用真正的硬二值 latent 计算 MSE 和逐图片平均 PSNR。"""
    model.eval()
    mse_sum = psnr_sum = 0.0
    for images, _ in loader:
        images = images.to(device)
        reconstructed = model(images, hard=True)
        image_mse = (reconstructed - images).square().mean(dim=(1, 2, 3))
        mse_sum += image_mse.sum().item()
        psnr_sum += (-10 * torch.log10(image_mse.clamp_min(1e-10))).sum().item()
    count = len(loader.dataset)
    return mse_sum / count, psnr_sum / count


def parse_args():
    parser = argparse.ArgumentParser()
    # 默认值复现实测超过 25 dB 的 256 张训练子集实验。
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--activation", choices=("silu", "relu"), default="silu")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    source = datasets.MNIST(ROOT / "data", train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(ROOT / "data", train=False, download=True, transform=transforms.ToTensor())
    max_train_size = len(source) - VAL_SIZE
    if not (2 <= args.batch_size <= args.train_size <= max_train_size):
        raise ValueError(f"需要 2 <= batch size <= train size <= {max_train_size}")
    if not (0 <= args.warmup_epochs < args.epochs):
        raise ValueError("需要 0 <= warmup epochs < epochs")

    indices = torch.randperm(len(source), generator=torch.Generator().manual_seed(SEED))
    val_indices = indices[256:256 + VAL_SIZE]
    available = torch.cat([indices[:256], indices[256 + VAL_SIZE:]])
    train_indices = available[:args.train_size]
    train_set = Subset(source, train_indices.tolist())
    val_set = Subset(source, val_indices.tolist())
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_eval_loader = DataLoader(train_set, batch_size=64)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test, batch_size=64)

    model = OptimizedBinaryAutoencoder(args.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, threshold=0.0001,
        threshold_mode="rel", min_lr=0.00001,
    )
    loss_fn = nn.MSELoss()
    best_psnr = float("-inf")
    best_weights = None
    history = []

    print(
        f"Device: {device} | Train: {args.train_size} | Validation: {VAL_SIZE} | "
        f"Parameters: {sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        use_hard_codes = epoch > args.warmup_epochs
        loss_sum = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images, hard=use_hard_codes), images)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.size(0)

        val_mse, val_psnr = evaluate(model, val_loader, device)
        scheduler.step(val_mse)
        learning_rate = optimizer.param_groups[0]["lr"]
        train_mse = loss_sum / (len(train_loader) * args.batch_size)
        history.append((epoch, "hard" if use_hard_codes else "soft", train_mse, val_mse, val_psnr, learning_rate))
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            best_weights = deepcopy(model.state_dict())
        if epoch == 1 or epoch == args.warmup_epochs or epoch % 10 == 0 or epoch == args.epochs:
            phase = "hard" if use_hard_codes else "soft warm-up"
            print(
                f"Epoch {epoch:03d}/{args.epochs} | {phase} | Train MSE: {train_mse:.6f} | "
                f"Hard-val PSNR: {val_psnr:.2f} dB | LR: {learning_rate:.6g}",
                flush=True,
            )

    # 两个模型都在查看测试集之前确定：一个按验证集选择，一个是最后一轮。
    selected = {
        "best_validation": (best_epoch, best_weights),
        "final_epoch": (args.epochs, deepcopy(model.state_dict())),
    }
    scores = {}
    for label, (epoch, weights) in selected.items():
        model.load_state_dict(weights)
        train_mse, train_psnr = evaluate(model, train_eval_loader, device)
        val_mse, val_psnr = evaluate(model, val_loader, device)
        test_mse, test_psnr = evaluate(model, test_loader, device)
        scores[label] = (epoch, train_mse, train_psnr, val_mse, val_psnr, test_mse, test_psnr)

    # 二值性和编码利用率检查。
    model.load_state_dict(best_weights)
    model.eval()
    all_codes = []
    with torch.no_grad():
        for images, _ in val_loader:
            all_codes.append(model.encode(images.to(device), hard=True).cpu())
    all_codes = torch.cat(all_codes)
    constant_bits = ((all_codes == all_codes[0]).all(dim=0)).sum().item()

    summary = (
        f"Train size: {args.train_size} | Validation size: {VAL_SIZE} | Test size: {len(test)}\n"
        f"Architecture channels: 64/128/256 | Parameters: {sum(p.numel() for p in model.parameters()):,}\n"
        f"Activation: {args.activation.upper()}\n"
        f"Latent: 3x4x4, strict values -1/+1 during evaluation\n"
        f"Warm-up epochs: {args.warmup_epochs} | Total epochs: {args.epochs} | Best epoch: {best_epoch}\n"
        f"Optimizer: Adam | Initial LR: {args.learning_rate} | Final LR: {optimizer.param_groups[0]['lr']}\n"
        f"Scheduler: ReduceLROnPlateau(validation MSE)\n"
        f"Constant validation bits: {constant_bits}/48\n"
    )
    for label, values in scores.items():
        epoch, train_mse, train_psnr, val_mse, val_psnr, test_mse, test_psnr = values
        summary += (
            f"{label} (epoch {epoch}): Train MSE {train_mse:.6f}, PSNR {train_psnr:.2f} dB | "
            f"Validation MSE {val_mse:.6f}, PSNR {val_psnr:.2f} dB | "
            f"Test MSE {test_mse:.6f}, PSNR {test_psnr:.2f} dB | "
            f"Train > 25: {train_psnr > 25} | Test > 25: {test_psnr > 25}\n"
        )
    print("\n" + summary, flush=True)

    run_dir = ROOT / "outputs" / "task2_optimized" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    for label, (_, weights) in selected.items():
        torch.save({name: value.cpu() for name, value in weights.items()}, run_dir / f"{label}.pt")
    torch.save({"train": train_indices, "validation": val_indices}, run_dir / "split_indices.pt")
    (run_dir / "results.txt").write_text(summary, encoding="utf-8")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "phase", "train_mse", "hard_val_mse", "hard_val_psnr", "learning_rate"])
        writer.writerows(history)

    train_images = torch.stack([source[i][0] for i in train_indices[:8].tolist()]).to(device)
    test_images = torch.stack([test[i][0] for i in range(8)]).to(device)
    for label, (_, weights) in selected.items():
        model.load_state_dict(weights)
        model.eval()
        with torch.no_grad():
            for split, images in (("train", train_images), ("test", test_images)):
                reconstructed = model(images, hard=True)
                save_image(
                    torch.cat([images, reconstructed]).cpu(),
                    run_dir / f"{label}_{split}.png",
                    nrow=8,
                )
    latent = model.encode(test_images, hard=True)
    print(f"Latent shape: {list(latent.shape)} | Values: {latent.unique().cpu().tolist()}")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
