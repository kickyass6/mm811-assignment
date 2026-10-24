"""Task 4：在原 VQ-VAE 上加入 masked reconstruction 和轻量感知损失。"""

import argparse
import math
from copy import deepcopy
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1
from torchvision.utils import save_image

from task4_vqvae_simple import ROOT, VQVAE, evaluate, get_device, make_datasets


SEED = 42


class PerceptualLoss(nn.Module):
    """用很小的预训练 SqueezeNet 比较图像特征。"""

    def __init__(self):
        super().__init__()
        self.features = squeezenet1_1(
            weights=SqueezeNet1_1_Weights.DEFAULT
        ).features[:8].eval()
        self.features.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, reconstructed, target):
        reconstructed = (reconstructed - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.l1_loss(self.features(reconstructed), self.features(target))


def mask_patches(images, ratio):
    """随机遮挡 4x4 patches，并用中性灰填充。"""
    visible = torch.rand(len(images), 1, 8, 8, device=images.device) > ratio
    visible = F.interpolate(visible.float(), size=(32, 32), mode="nearest")
    return images * visible + 0.5 * (1 - visible)


@torch.no_grad()
def reconstruct_dataset(model, loader, device):
    model.eval()
    outputs = []
    for images, _ in loader:
        reconstructed, _, _ = model(images.to(device))
        outputs.append((reconstructed.clamp(0, 1) * 255).round().byte().cpu())
    return torch.cat(outputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-size", type=int, default=45000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--mask-weight", type=float, default=0.10)
    parser.add_argument("--perceptual-weight", type=float, default=0.01)
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--keep-final", action="store_true")
    parser.add_argument("--fake-data", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.mask_ratio < 1:
        raise ValueError("mask-ratio 必须在 [0, 1) 内")
    if args.fake_data and (args.train_size, args.val_size) == (45000, 5000):
        args.train_size, args.val_size = 64, 32

    torch.manual_seed(SEED)
    device = get_device()
    train_set, train_eval_set, val_set, test_set = make_datasets(
        args.fake_data, args.train_size, args.val_size
    )
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_eval_set, args.batch_size)
    val_loader = DataLoader(val_set, args.batch_size)
    test_loader = DataLoader(test_set, args.batch_size)

    model = VQVAE(args.codebook_size).to(device)
    if args.init_checkpoint:
        model.load_state_dict(
            torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        )
    perceptual_loss = PerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1e-6
    )
    best = {"psnr": -math.inf}

    if args.init_checkpoint:
        initial = evaluate(model, val_loader, device)
        best = {
            "psnr": initial[1],
            "epoch": 0,
            "weights": deepcopy(model.state_dict()),
        }
        print(
            f"Loaded checkpoint: {args.init_checkpoint} | Initial val PSNR: "
            f"{initial[1]:.2f} dB | Codes {initial[2]}/{args.codebook_size}",
            flush=True,
        )

    print(
        f"Device: {device} | Train: {len(train_set)} | Validation: {len(val_set)} | "
        f"Codebook: {args.codebook_size} x 10 | Mask: {args.mask_ratio:.0%} | "
        f"Mask weight: {args.mask_weight} | Perceptual weight: {args.perceptual_weight}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_index, (images, _) in enumerate(train_loader):
            images = images.to(device)
            reconstructed, vq_loss, _ = model(images)
            clean_loss = F.mse_loss(reconstructed, images)

            # 只取部分图片计算感知特征，保持训练速度和代码简单。
            count = min(32, len(images))
            feature_loss = perceptual_loss(reconstructed[:count], images[:count])
            loss = clean_loss + vq_loss + args.perceptual_weight * feature_loss

            # 每四个 batch 增加一次 masked reconstruction，额外计算约 25%。
            if batch_index % 4 == 0:
                masked_images = mask_patches(images, args.mask_ratio)
                masked_reconstruction, masked_vq_loss, _ = model(masked_images)
                masked_loss = F.mse_loss(masked_reconstruction, images)
                loss = loss + args.mask_weight * (masked_loss + masked_vq_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_mse, val_psnr, used_codes, perplexity = evaluate(model, val_loader, device)
        if val_psnr > best["psnr"]:
            best = {
                "psnr": val_psnr,
                "epoch": epoch,
                "weights": deepcopy(model.state_dict()),
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | Val MSE {val_mse:.6f} | "
                f"Val PSNR {val_psnr:.2f} dB | Codes {used_codes}/{args.codebook_size} | "
                f"Perplexity {perplexity:.1f}",
                flush=True,
            )

    selected_epoch = args.epochs if args.keep_final else best["epoch"]
    if not args.keep_final:
        model.load_state_dict(best["weights"])
    train_result = evaluate(model, train_eval_loader, device)
    val_result = evaluate(model, val_loader, device)
    test_result = evaluate(model, test_loader, device)
    test_reconstructions = reconstruct_dataset(model, test_loader, device)

    sample_images = torch.stack([test_set[i][0] for i in range(16)]).to(device)
    with torch.no_grad():
        sample_reconstructions, _, sample_indices = model(sample_images)

    summary = (
        f"Dataset: {'FakeData smoke test' if args.fake_data else 'CIFAR-10'}\n"
        f"Discrete index shape: {list(sample_indices.shape)}\n"
        f"Quantized embedding shape: [16, 10, 8, 8]\n"
        f"Codebook size: {args.codebook_size}\n"
        f"Mask ratio: {args.mask_ratio} | Mask weight: {args.mask_weight}\n"
        f"Perceptual weight: {args.perceptual_weight}\n"
        f"Best epoch: {best['epoch']}\n"
        f"Selected epoch: {selected_epoch}\n"
        f"Train PSNR: {train_result[1]:.2f} dB\n"
        f"Validation PSNR: {val_result[1]:.2f} dB\n"
        f"Test PSNR: {test_result[1]:.2f} dB\n"
        f"Test codes used: {test_result[2]}/{args.codebook_size}\n"
        f"Test perplexity: {test_result[3]:.1f}\n"
        f"Test PSNR > 25: {test_result[1] > 25}\n"
    )
    print("\n" + summary)

    run_dir = ROOT / "outputs" / "task4_vqvae_masked" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    run_dir.mkdir(parents=True)
    torch.save(model.state_dict(), run_dir / "vqvae.pt")
    torch.save(test_reconstructions, run_dir / "test_reconstructions.pt")
    (run_dir / "results.txt").write_text(summary, encoding="utf-8")
    save_image(
        torch.cat([sample_images, sample_reconstructions]).cpu(),
        run_dir / "reconstruction.png",
        nrow=16,
    )
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
