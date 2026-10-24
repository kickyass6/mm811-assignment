"""Fine-tune the Task 4 continuous AE for perceptual reconstruction quality.

The latent remains exactly 5x8x8.  Training combines pixel MSE, LPIPS, a tiny
latent L2 penalty, and relative Gaussian latent noise so that the frozen decoder
is tolerant to imperfect samples from a learned latent prior.
"""

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from task4_models import ROOT, ContinuousAutoencoder


SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-ae", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lpips-weight", type=float, default=0.08)
    parser.add_argument("--latent-noise", type=float, default=0.10)
    parser.add_argument("--latent-weight", type=float, default=1e-5)
    parser.add_argument("--train-size", type=int, default=45000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fake-data", action="store_true")
    return parser.parse_args()


def build_datasets(args):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    ])
    eval_transform = transforms.ToTensor()
    if args.fake_data:
        total = args.train_size + args.val_size
        source_train = datasets.FakeData(
            total, image_size=(3, 32, 32), num_classes=10,
            transform=train_transform, random_offset=0,
        )
        source_eval = datasets.FakeData(
            total, image_size=(3, 32, 32), num_classes=10,
            transform=eval_transform, random_offset=0,
        )
        test = datasets.FakeData(
            args.val_size, image_size=(3, 32, 32), num_classes=10,
            transform=eval_transform, random_offset=100000,
        )
    else:
        source_train = datasets.CIFAR10(
            ROOT / "data", train=True, download=True, transform=train_transform
        )
        source_eval = datasets.CIFAR10(
            ROOT / "data", train=True, download=False, transform=eval_transform
        )
        test = datasets.CIFAR10(
            ROOT / "data", train=False, download=True, transform=eval_transform
        )
    permutation = torch.randperm(
        len(source_train), generator=torch.Generator().manual_seed(SEED)
    )
    train_indices = permutation[:args.train_size].tolist()
    val_indices = permutation[args.train_size:args.train_size + args.val_size].tolist()
    return (
        Subset(source_train, train_indices),
        Subset(source_eval, val_indices),
        test,
    )


def load_lpips(device):
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError("Install LPIPS with: python3 -m pip install --user lpips") from error
    metric = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    return metric


@torch.no_grad()
def evaluate(model, loader, perceptual, device, lpips_limit=None):
    model.eval()
    mse_sum = psnr_sum = lpips_sum = 0.0
    count = lpips_count = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        reconstructed = model(images).clamp(0, 1)
        per_image_mse = (images - reconstructed).square().flatten(1).mean(1)
        mse_sum += per_image_mse.sum().item()
        psnr_sum += (-10 * torch.log10(per_image_mse.clamp_min(1e-10))).sum().item()
        count += len(images)
        if lpips_limit is None or lpips_count < lpips_limit:
            take = len(images) if lpips_limit is None else min(len(images), lpips_limit - lpips_count)
            lpips_sum += perceptual(
                reconstructed[:take] * 2 - 1, images[:take] * 2 - 1
            ).sum().item()
            lpips_count += take
    return {
        "mse": mse_sum / count,
        "psnr": psnr_sum / count,
        "lpips": lpips_sum / max(lpips_count, 1),
    }


def main():
    args = parse_args()
    if min(args.epochs, args.batch_size, args.train_size, args.val_size) < 1:
        raise ValueError("epochs, batch size, and dataset sizes must be positive")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    train_set, val_set, test_set = build_datasets(args)
    loader_args = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(SEED), **loader_args,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    model = ContinuousAutoencoder().to(device)
    state = torch.load(args.init_ae, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    perceptual = load_lpips(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best = {"score": float("inf")}
    history = []
    print(
        f"Device: {device} | Train: {len(train_set)} | Validation: {len(val_set)} | "
        f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"mse": 0.0, "lpips": 0.0, "loss": 0.0, "count": 0}
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                latent = model.encoder(images)
                channel_scale = latent.detach().float().std(dim=(0, 2, 3), keepdim=True)
                noisy = latent + args.latent_noise * channel_scale.to(latent.dtype) * torch.randn_like(latent)
                reconstructed = model.decoder(noisy)
            # VGG LPIPS is prone to overflowing in fp16.  Keep the expensive AE
            # forward in mixed precision, but form every loss in fp32 so that a
            # finite first optimizer step is guaranteed on 8 GB Turing GPUs.
            reconstructed_float = reconstructed.float()
            mse = F.mse_loss(reconstructed_float, images.float())
            lpips_loss = perceptual(
                reconstructed_float * 2 - 1, images.float() * 2 - 1
            ).mean()
            loss = (
                mse
                + args.lpips_weight * lpips_loss
                + args.latent_weight * latent.float().square().mean()
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch = len(images)
            totals["mse"] += mse.item() * batch
            totals["lpips"] += lpips_loss.item() * batch
            totals["loss"] += loss.item() * batch
            totals["count"] += batch
        scheduler.step()
        validation = evaluate(model, val_loader, perceptual, device, lpips_limit=1024)
        score = validation["mse"] + args.lpips_weight * validation["lpips"]
        if validation["psnr"] > 25 and score < best["score"]:
            best = {
                "score": score, "epoch": epoch, "validation": validation,
                "weights": deepcopy(model.state_dict()),
            }
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / totals["count"],
            "train_mse": totals["mse"] / totals["count"],
            "train_lpips": totals["lpips"] / totals["count"],
            "val_mse": validation["mse"],
            "val_psnr": validation["psnr"],
            "val_lpips": validation["lpips"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train MSE {row['train_mse']:.6f} | "
            f"LPIPS {row['train_lpips']:.4f} | val PSNR {row['val_psnr']:.2f} dB | "
            f"val LPIPS {row['val_lpips']:.4f}", flush=True,
        )

    if "weights" not in best:
        raise RuntimeError("No checkpoint retained PSNR > 25 dB")
    model.load_state_dict(best["weights"])
    test = evaluate(model, test_loader, perceptual, device)
    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, output / "autoencoder.pt")
    reconstructed_test = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            reconstructed = model(images).clamp(0, 1)
            reconstructed_test.append((reconstructed * 255).round().byte().cpu())
    reconstructed_test = torch.cat(reconstructed_test)
    torch.save(reconstructed_test, output / "reconstructed_test_images.pt")
    sample = torch.stack([test_set[index][0] for index in range(32)]).to(device)
    with torch.no_grad():
        reconstructed = model(sample).clamp(0, 1)
    save_image(torch.cat([sample, reconstructed]).cpu(), output / "reconstruction.png", nrow=8)

    summary = {
        "architecture": "existing 5x8x8 continuous AE, perceptual fine-tuning",
        "init_ae": str(Path(args.init_ae).resolve()),
        "latent": [5, 8, 8],
        "best_epoch": best["epoch"],
        "validation": best["validation"],
        "test": test,
        "config": vars(args),
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results saved to: {output}", flush=True)


if __name__ == "__main__":
    main()
