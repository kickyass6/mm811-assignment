"""任务 4.1：训练 CIFAR-10 连续与离散自编码器。"""

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from task4_models import (
    ROOT, ContinuousAutoencoder, DiscreteAutoencoder,
    device_name, reconstruction_metrics,
)


SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--train-size", type=int, default=45000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--fake-data", action="store_true")
    return parser.parse_args()


def make_datasets(fake_data, train_size, val_size):
    evaluation_transform = transforms.ToTensor()
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    ])
    if fake_data:
        total = train_size + val_size
        train_source = datasets.FakeData(
            total, image_size=(3, 32, 32), num_classes=10,
            transform=evaluation_transform, random_offset=0,
        )
        val_source = datasets.FakeData(
            total, image_size=(3, 32, 32), num_classes=10,
            transform=evaluation_transform, random_offset=0,
        )
        test = datasets.FakeData(
            max(val_size, 64), image_size=(3, 32, 32), num_classes=10,
            transform=evaluation_transform, random_offset=100000,
        )
    else:
        train_source = datasets.CIFAR10(
            ROOT / "data", train=True, download=True, transform=training_transform
        )
        val_source = datasets.CIFAR10(
            ROOT / "data", train=True, download=False, transform=evaluation_transform
        )
        test = datasets.CIFAR10(
            ROOT / "data", train=False, download=True, transform=evaluation_transform
        )
    permutation = torch.randperm(len(train_source), generator=torch.Generator().manual_seed(SEED))
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:train_size + val_size]
    return (
        Subset(train_source, train_indices.tolist()),
        Subset(val_source, train_indices.tolist()),
        Subset(val_source, val_indices.tolist()),
        test,
        train_indices,
        val_indices,
    )


def train_one_epoch(model, loader, optimizer, device, discrete, hard, steepness):
    model.train()
    loss_sum = 0.0
    for images, _ in loader:
        images = images.to(device)
        optimizer.zero_grad()
        if discrete:
            reconstructed = model(images, hard=hard, steepness=steepness)
        else:
            reconstructed = model(images)
        loss = nn.functional.mse_loss(reconstructed, images)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * len(images)
    return loss_sum / len(loader.dataset)


def main():
    args = parse_args()
    if args.fake_data and (args.train_size, args.val_size) == (45000, 5000):
        args.train_size, args.val_size = 256, 64
    if min(args.epochs, args.batch_size, args.train_size, args.val_size) < 1:
        raise ValueError("epochs、batch size 和数据数量必须为正数")
    if not (0 <= args.warmup_epochs < args.epochs):
        raise ValueError("需要 0 <= warmup epochs < epochs")

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = device_name()
    train_set, train_eval_set, val_set, test_set, train_indices, val_indices = make_datasets(
        args.fake_data, args.train_size, args.val_size
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), drop_last=False,
    )
    train_eval_loader = DataLoader(train_eval_set, batch_size=args.batch_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    continuous = ContinuousAutoencoder().to(device)
    discrete = DiscreteAutoencoder().to(device)
    continuous_optimizer = torch.optim.AdamW(
        continuous.parameters(), lr=args.learning_rate, weight_decay=0.0001
    )
    discrete_optimizer = torch.optim.AdamW(
        discrete.parameters(), lr=args.learning_rate, weight_decay=0.0001
    )
    continuous_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        continuous_optimizer, args.epochs, eta_min=0.00001
    )
    discrete_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        discrete_optimizer, args.epochs, eta_min=0.00001
    )
    best = {
        "continuous": {"psnr": float("-inf")},
        "discrete": {"psnr": float("-inf")},
    }
    history = []
    print(
        f"Device: {device} | Train: {len(train_set)} | Validation: {len(val_set)} | "
        f"Continuous parameters: {sum(p.numel() for p in continuous.parameters()):,} | "
        f"Discrete parameters: {sum(p.numel() for p in discrete.parameters()):,}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        continuous_loss = train_one_epoch(
            continuous, train_loader, continuous_optimizer, device,
            discrete=False, hard=True, steepness=5.0,
        )
        hard = epoch > args.warmup_epochs
        steepness = 1.0 + 4.0 * min(epoch / max(args.warmup_epochs, 1), 1.0)
        discrete_loss = train_one_epoch(
            discrete, train_loader, discrete_optimizer, device,
            discrete=True, hard=hard, steepness=steepness,
        )
        continuous_scheduler.step()
        discrete_scheduler.step()
        continuous_val_mse, continuous_val_psnr = reconstruction_metrics(
            continuous, val_loader, device, discrete=False
        )
        discrete_val_mse, discrete_val_psnr = reconstruction_metrics(
            discrete, val_loader, device, discrete=True
        )
        if continuous_val_psnr > best["continuous"]["psnr"]:
            best["continuous"] = {
                "psnr": continuous_val_psnr, "epoch": epoch,
                "weights": deepcopy(continuous.state_dict()),
            }
        if discrete_val_psnr > best["discrete"]["psnr"]:
            best["discrete"] = {
                "psnr": discrete_val_psnr, "epoch": epoch,
                "weights": deepcopy(discrete.state_dict()),
            }
        history.append((
            epoch, continuous_loss, continuous_val_mse, continuous_val_psnr,
            discrete_loss, discrete_val_mse, discrete_val_psnr,
        ))
        if epoch == 1 or epoch == args.warmup_epochs or epoch % 10 == 0 or epoch == args.epochs:
            phase = "hard" if hard else "soft warm-up"
            print(
                f"Epoch {epoch:03d}/{args.epochs} | Continuous val {continuous_val_psnr:.2f} dB | "
                f"Discrete val {discrete_val_psnr:.2f} dB ({phase})",
                flush=True,
            )

    continuous.load_state_dict(best["continuous"]["weights"])
    discrete.load_state_dict(best["discrete"]["weights"])
    continuous_train = reconstruction_metrics(continuous, train_eval_loader, device, False)
    continuous_test = reconstruction_metrics(continuous, test_loader, device, False)
    discrete_train = reconstruction_metrics(discrete, train_eval_loader, device, True)
    discrete_test = reconstruction_metrics(discrete, test_loader, device, True)

    with torch.no_grad():
        sample_images = torch.stack([test_set[i][0] for i in range(min(16, len(test_set)))]).to(device)
        continuous_latent = continuous.encode(sample_images)
        discrete_latent = discrete.encode(sample_images, hard=True)
    summary = (
        f"Dataset: {'FakeData smoke test' if args.fake_data else 'CIFAR-10'}\n"
        f"Train: {len(train_set)} | Validation: {len(val_set)} | Test: {len(test_set)}\n"
        f"Continuous latent: {list(continuous_latent.shape)}, limit 8x8x5\n"
        f"Continuous best epoch: {best['continuous']['epoch']} | Train PSNR: {continuous_train[1]:.2f} dB | "
        f"Validation PSNR: {best['continuous']['psnr']:.2f} dB | Test PSNR: {continuous_test[1]:.2f} dB\n"
        f"Discrete latent: {list(discrete_latent.shape)}, values: {discrete_latent.unique().cpu().tolist()}, limit 8x8x10\n"
        f"Discrete best epoch: {best['discrete']['epoch']} | Train PSNR: {discrete_train[1]:.2f} dB | "
        f"Validation PSNR: {best['discrete']['psnr']:.2f} dB | Test PSNR: {discrete_test[1]:.2f} dB\n"
    )
    print("\n" + summary, flush=True)

    run_dir = ROOT / "outputs" / "task4_autoencoders" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save({k: v.cpu() for k, v in best["continuous"]["weights"].items()}, run_dir / "continuous.pt")
    torch.save({k: v.cpu() for k, v in best["discrete"]["weights"].items()}, run_dir / "discrete.pt")
    torch.save({"train": train_indices, "validation": val_indices}, run_dir / "split_indices.pt")
    (run_dir / "results.txt").write_text(summary, encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "epoch", "continuous_train_mse", "continuous_val_mse", "continuous_val_psnr",
            "discrete_train_mse", "discrete_val_mse", "discrete_val_psnr",
        ])
        writer.writerows(history)
    save_image(
        torch.cat([sample_images, continuous(sample_images)]).cpu(),
        run_dir / "continuous_reconstruction.png", nrow=len(sample_images),
    )
    save_image(
        torch.cat([sample_images, discrete(sample_images, hard=True)]).cpu(),
        run_dir / "discrete_reconstruction.png", nrow=len(sample_images),
    )
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
