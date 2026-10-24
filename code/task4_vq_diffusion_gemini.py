"""Task 4.2：Gemini 建议的标准 absorbing-mask 离散扩散。"""

import argparse
import csv
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from task4_vqvae_simple import ROOT, VQVAE, get_device


SEED = 42
NULL_CLASS = 10


def find_vqvae(user_path=None):
    """找到 VQ-VAE 权重；默认选择最新一次训练结果。"""
    if user_path:
        checkpoint = Path(user_path).expanduser().resolve()
    else:
        candidates = sorted((ROOT / "outputs" / "task4_vqvae").glob("*/vqvae.pt"))
        if not candidates:
            raise FileNotFoundError("没有找到 vqvae.pt，请先训练 task4_vqvae_simple.py")
        checkpoint = candidates[-1]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def load_vqvae(checkpoint, device):
    """从 codebook 权重自动读出 codebook size 和 latent dimension。"""
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    codebook_size, latent_dim = weights["quantizer.codebook.weight"].shape
    model = VQVAE(codebook_size, latent_dim).to(device)
    model.load_state_dict(weights)
    model.eval().requires_grad_(False)
    return model, codebook_size


@torch.no_grad()
def encode_dataset(vqvae, dataset, batch_size, device, augmentation="none"):
    """CIFAR-10 -> [N,8,8]，每个数都是一个 codebook index。"""
    loader = DataLoader(dataset, batch_size=batch_size)
    index_batches, flipped_batches, label_batches = [], [], []
    for images, labels in loader:
        images = images.to(device)
        indices = vqvae.encode(images)
        index_batches.append(indices.to(torch.int16).cpu())
        if augmentation == "pixel-flip":
            flipped_batches.append(vqvae.encode(images.flip(-1)).to(torch.int16).cpu())
        label_batches.append(labels.long())
    latents = torch.cat(index_batches)
    flipped = torch.cat(flipped_batches) if flipped_batches else latents
    return latents, flipped, torch.cat(label_batches)


def time_embedding(timesteps, width):
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class CategoricalDenoiser(nn.Module):
    """带位置编码的全局 Transformer，预测每个位置的原始 code。"""

    def __init__(self, codebook_size, width=256, layers=8, heads=8):
        super().__init__()
        self.codebook_size = codebook_size
        self.mask_id = codebook_size
        self.token_embedding = nn.Embedding(codebook_size + 1, width)
        self.position_embedding = nn.Embedding(64, width)
        self.class_embedding = nn.Embedding(11, width)  # CIFAR-10 + unconditional
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width)
        )
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.0, batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, codebook_size)

    def forward(self, noisy_indices, timesteps, labels):
        batch = noisy_indices.size(0)
        sequence = noisy_indices.flatten(1)  # [B, 64]
        positions = torch.arange(64, device=noisy_indices.device)
        hidden = self.token_embedding(sequence) + self.position_embedding(positions)[None]
        condition = self.time_mlp(time_embedding(timesteps, hidden.size(-1)))
        condition = condition + self.class_embedding(labels)
        hidden = self.transformer(hidden + condition[:, None])
        logits = self.output(self.norm(hidden))
        return logits.view(batch, 8, 8, self.codebook_size).permute(0, 3, 1, 2)


def add_noise(clean_indices, timesteps, total_timesteps, mask_id):
    """时间越大，被替换成 [MASK] 的位置越多。"""
    probability = timesteps.float()[:, None, None] / total_timesteps
    mask = torch.rand(clean_indices.shape, device=clean_indices.device) < probability

    # 每张图片至少遮住一个位置，确保 loss 始终有定义。
    empty = ~mask.flatten(1).any(1)
    if empty.any():
        random_position = torch.randint(0, 64, (empty.sum(),), device=clean_indices.device)
        rows = empty.nonzero(as_tuple=False).squeeze(1)
        mask.view(len(mask), -1)[rows, random_position] = True

    noisy = clean_indices.clone()
    noisy[mask] = mask_id
    return noisy, mask


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter, 1 - decay)


@torch.no_grad()
def sample_indices(model, labels, timesteps, guidance, temperature):
    """从全 [MASK] 开始，每一步恢复一部分 index。"""
    model.eval()
    count = len(labels)
    device = labels.device
    indices = torch.full((count, 8, 8), model.mask_id, dtype=torch.long, device=device)

    for step in range(timesteps, 0, -1):
        time = torch.full((count,), step, dtype=torch.long, device=device)
        conditional = model(indices, time, labels)

        if guidance != 1:
            null_labels = torch.full_like(labels, NULL_CLASS)
            unconditional = model(indices, time, null_labels)
            logits = unconditional + guidance * (conditional - unconditional)
        else:
            logits = conditional

        probabilities = torch.softmax(logits / temperature, dim=1)
        sampled = torch.multinomial(
            probabilities.permute(0, 2, 3, 1).reshape(-1, model.codebook_size), 1
        ).view(count, 8, 8)
        confidence = probabilities.gather(1, sampled[:, None]).squeeze(1)

        # 当前 step 后应保留多少个 [MASK]。每步揭开置信度最高的位置。
        target_mask_count = round(64 * (step - 1) / timesteps)
        for image_index in range(count):
            flat_indices = indices[image_index].view(-1)
            masked_positions = (flat_indices == model.mask_id).nonzero(as_tuple=False).squeeze(1)
            reveal_count = len(masked_positions) - target_mask_count
            if reveal_count <= 0:
                continue
            scores = confidence[image_index].flatten()[masked_positions]
            selected = masked_positions[scores.topk(reveal_count).indices]
            flat_indices[selected] = sampled[image_index].flatten()[selected]

    return indices


@torch.no_grad()
def generate_images(model, vqvae, count, timesteps, batch_size, guidance, temperature, device):
    image_batches = []
    for start in range(0, count, batch_size):
        current = min(batch_size, count - start)
        labels = torch.arange(start, start + current, device=device) % 10
        indices = sample_indices(model, labels, timesteps, guidance, temperature)
        images = vqvae.decode_indices(indices).clamp(0, 1)
        image_batches.append((images * 255).round().to(torch.uint8).cpu())
    return torch.cat(image_batches)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--generate", type=int, default=10000)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None,
                        help="EMA checkpoint used to continue training")
    parser.add_argument("--previous-epochs", type=int, default=0)
    parser.add_argument("--augmentation", choices=("none", "pixel-flip"), default="pixel-flip")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fake-data", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fake_data and args.train_size == 50000:
        args.train_size = 64
    if min(args.epochs, args.batch_size, args.timesteps, args.train_size, args.generate) < 1:
        raise ValueError("epochs、batch size、timesteps、数据量和生成量必须为正数")
    if args.temperature <= 0:
        raise ValueError("temperature 必须大于0")

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = get_device()
    checkpoint = find_vqvae(args.vqvae)
    vqvae, codebook_size = load_vqvae(checkpoint, device)

    if args.fake_data:
        dataset = datasets.FakeData(
            args.train_size, (3, 32, 32), 10,
            transforms.ToTensor(), random_offset=0,
        )
    else:
        dataset = datasets.CIFAR10(
            ROOT / "data", train=True, download=True, transform=transforms.ToTensor()
        )
        args.train_size = min(args.train_size, len(dataset))
        if args.train_size < len(dataset):
            dataset = torch.utils.data.Subset(dataset, range(args.train_size))

    run_dir = Path(args.output).expanduser() if args.output else (
        ROOT / "outputs" / "task4_vq_diffusion_gemini" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    latents, flipped_latents, labels = encode_dataset(
        vqvae, dataset, args.batch_size, device, args.augmentation
    )
    loader = DataLoader(
        TensorDataset(latents, flipped_latents, labels), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = CategoricalDenoiser(codebook_size).to(device)
    ema_model = deepcopy(model).eval().requires_grad_(False)
    if args.resume:
        resume_weights = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(resume_weights)
        ema_model.load_state_dict(resume_weights)
        print(f"Resumed from: {args.resume}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(args.epochs, 150), eta_min=1e-5
    )
    history = []

    print(
        f"Device: {device} | VQ-VAE: {checkpoint} | Latents: {list(latents.shape)} | "
        f"Values: 0..{codebook_size - 1} | Parameters: {sum(p.numel() for p in model.parameters()):,} | "
        f"Augmentation: {args.augmentation} | Seed: {args.seed}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        masked_sum = 0

        for clean_indices, flipped_indices, batch_labels in loader:
            clean_indices = clean_indices.long().to(device)
            # 选择“翻转图片后重新编码”的 latent，不能直接翻转 code 网格。
            if args.augmentation == "pixel-flip":
                flip = torch.rand(len(clean_indices), device=device) < 0.5
                clean_indices[flip] = flipped_indices.long().to(device)[flip]
            batch_labels = batch_labels.to(device)
            time = torch.randint(1, args.timesteps + 1, (len(clean_indices),), device=device)
            noisy_indices, mask = add_noise(
                clean_indices, time, args.timesteps, model.mask_id
            )

            # 10% 删除类别标签，用于采样时的 classifier-free guidance。
            dropped_labels = batch_labels.clone()
            dropped_labels[torch.rand(len(batch_labels), device=device) < 0.1] = NULL_CLASS

            logits = model(noisy_indices, time, dropped_labels)
            token_loss = F.cross_entropy(logits, clean_indices, reduction="none")
            loss = token_loss[mask].mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema_model, model)

            loss_sum += token_loss[mask].sum().item()
            masked_sum += mask.sum().item()

        scheduler.step()
        mean_loss = loss_sum / masked_sum
        history.append((epoch, mean_loss, optimizer.param_groups[0]["lr"]))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            torch.save(ema_model.state_dict(), run_dir / "diffusion_ema.pt")
            print(
                f"Epoch {epoch + args.previous_epochs:03d}/"
                f"{args.epochs + args.previous_epochs} | Masked-token CE {mean_loss:.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )

    generated = generate_images(
        ema_model, vqvae, args.generate, args.timesteps,
        min(args.batch_size, 128), args.guidance, args.temperature, device,
    )
    summary = (
        f"Dataset: {'FakeData smoke test' if args.fake_data else 'CIFAR-10'}\n"
        f"VQ-VAE checkpoint: {checkpoint}\n"
        f"Training latents: {len(latents)} x 8 x 8, integer values 0..{codebook_size - 1}\n"
        f"Diffusion: absorbing [MASK] process, {args.timesteps} timesteps\n"
        f"Augmentation: {args.augmentation} | Seed: {args.seed}\n"
        f"Epochs: {args.epochs + args.previous_epochs} total "
        f"({args.epochs} this run) | Final masked-token CE: {history[-1][1]:.6f}\n"
        f"Guidance: {args.guidance} | Temperature: {args.temperature}\n"
        f"Generated images: {len(generated)} | FID: run task4_fid_standard.py on generated_images.pt\n"
    )
    print("\n" + summary, flush=True)

    torch.save(ema_model.state_dict(), run_dir / "diffusion_ema.pt")
    torch.save(generated, run_dir / "generated_images.pt")
    (run_dir / "results.txt").write_text(summary, encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "masked_token_cross_entropy", "learning_rate"])
        writer.writerows(history)
    save_image(generated[:64].float() / 255, run_dir / "generated_grid.png", nrow=8)
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
