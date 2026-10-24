"""Task 4: 一个尽量精简、便于阅读的 CIFAR-10 VQ-VAE。"""

import argparse
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parent
SEED = 42


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ResidualBlock(nn.Module):
    """课堂代码中的 residual layer：输出 = 输入 + 卷积结果。"""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """32x32 图片经过两次下采样，得到 10x8x8 特征。"""

    def __init__(self, latent_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 16 -> 8
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Conv2d(256, latent_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class VectorQuantizer(nn.Module):
    """把每个 10 维向量替换成 codebook 中距离最近的向量。"""

    def __init__(self, codebook_size=512, latent_dim=10, commitment=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.commitment = commitment
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        self.codebook.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

    def forward(self, z_e):
        # [B, 10, 8, 8] -> [B*8*8, 10]
        z = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z = z.view(-1, z.size(-1))

        # ||z-e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            flat_z.square().sum(1, keepdim=True)
            + self.codebook.weight.square().sum(1)
            - 2 * flat_z @ self.codebook.weight.t()
        )
        indices = distances.argmin(1)
        z_q = self.codebook(indices).view_as(z)

        # 第一项更新 codebook，第二项让 encoder 输出靠近选中的 code。
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        vq_loss = codebook_loss + self.commitment * commitment_loss

        # 前向使用 z_q，反向时梯度直接传给 encoder（straight-through）。
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        indices = indices.view(z.size(0), z.size(1), z.size(2))
        return z_q, vq_loss, indices


class Decoder(nn.Module):
    """把量化后的 10x8x8 特征恢复成 3x32x32 图片。"""

    def __init__(self, latent_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, 256, 3, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_q):
        return self.net(z_q)


class VQVAE(nn.Module):
    def __init__(self, codebook_size=512, latent_dim=10):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.quantizer = VectorQuantizer(codebook_size, latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, images):
        z_e = self.encoder(images)
        z_q, vq_loss, indices = self.quantizer(z_e)
        reconstruction = self.decoder(z_q)
        return reconstruction, vq_loss, indices

    @torch.no_grad()
    def encode(self, images):
        """返回真正的离散表示：[B, 8, 8] 整数 codebook index。"""
        _, _, indices = self.forward(images)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices):
        """把离散 index 转回 codebook 向量，再解码成图片。"""
        z_q = self.quantizer.codebook(indices)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)


def make_datasets(fake_data, train_size, val_size):
    plain = transforms.ToTensor()
    augmented = transforms.Compose([transforms.RandomHorizontalFlip(), plain])

    if fake_data:
        total = train_size + val_size
        train_source = datasets.FakeData(total, (3, 32, 32), 10, plain, random_offset=0)
        eval_source = datasets.FakeData(total, (3, 32, 32), 10, plain, random_offset=0)
        test_set = datasets.FakeData(max(val_size, 64), (3, 32, 32), 10, plain, random_offset=10000)
    else:
        train_source = datasets.CIFAR10(ROOT / "data", train=True, download=True, transform=augmented)
        eval_source = datasets.CIFAR10(ROOT / "data", train=True, download=False, transform=plain)
        test_set = datasets.CIFAR10(ROOT / "data", train=False, download=True, transform=plain)

    order = torch.randperm(len(train_source), generator=torch.Generator().manual_seed(SEED))
    train_ids = order[:train_size].tolist()
    val_ids = order[train_size:train_size + val_size].tolist()
    return Subset(train_source, train_ids), Subset(eval_source, train_ids), Subset(eval_source, val_ids), test_set


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_sum = psnr_sum = 0.0
    code_counts = torch.zeros(model.quantizer.codebook_size, device=device)

    for images, _ in loader:
        images = images.to(device)
        reconstructed, _, indices = model(images)
        image_mse = (images - reconstructed).square().mean((1, 2, 3))
        mse_sum += image_mse.sum().item()
        psnr_sum += (-10 * torch.log10(image_mse.clamp_min(1e-10))).sum().item()
        code_counts += torch.bincount(indices.flatten(), minlength=len(code_counts))

    probabilities = code_counts / code_counts.sum()
    nonzero = probabilities > 0
    perplexity = torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum()).item()
    used_codes = nonzero.sum().item()
    count = len(loader.dataset)
    return mse_sum / count, psnr_sum / count, used_codes, perplexity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-size", type=int, default=45000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--fake-data", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1e-5
    )
    best = {"psnr": -math.inf}

    print(
        f"Device: {device} | Train: {len(train_set)} | Validation: {len(val_set)} | "
        f"Codebook: {args.codebook_size} x 10",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, _ in train_loader:
            images = images.to(device)
            reconstructed, vq_loss, _ = model(images)
            reconstruction_loss = F.mse_loss(reconstructed, images)
            loss = reconstruction_loss + vq_loss

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

    model.load_state_dict(best["weights"])
    train_result = evaluate(model, train_eval_loader, device)
    val_result = evaluate(model, val_loader, device)
    test_result = evaluate(model, test_loader, device)

    sample_images = torch.stack([test_set[i][0] for i in range(16)]).to(device)
    with torch.no_grad():
        sample_reconstructions, _, sample_indices = model(sample_images)

    summary = (
        f"Dataset: {'FakeData smoke test' if args.fake_data else 'CIFAR-10'}\n"
        f"Discrete index shape: {list(sample_indices.shape)}\n"
        f"Quantized embedding shape: [16, 10, 8, 8]\n"
        f"Codebook size: {args.codebook_size}\n"
        f"Best epoch: {best['epoch']}\n"
        f"Train PSNR: {train_result[1]:.2f} dB\n"
        f"Validation PSNR: {val_result[1]:.2f} dB\n"
        f"Test PSNR: {test_result[1]:.2f} dB\n"
        f"Test codes used: {test_result[2]}/{args.codebook_size}\n"
        f"Test perplexity: {test_result[3]:.1f}\n"
        f"Test PSNR > 25: {test_result[1] > 25}\n"
    )
    print("\n" + summary)

    run_dir = ROOT / "outputs" / "task4_vqvae" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    torch.save(model.state_dict(), run_dir / "vqvae.pt")
    (run_dir / "results.txt").write_text(summary, encoding="utf-8")
    save_image(
        torch.cat([sample_images, sample_reconstructions]).cpu(),
        run_dir / "reconstruction.png",
        nrow=16,
    )
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
