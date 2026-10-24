"""Task 4.2: high-capacity absorbing diffusion for discrete VQ latents."""

import argparse
import csv
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from task4_vq_diffusion_gemini import (
    NULL_CLASS,
    add_noise,
    encode_dataset,
    find_vqvae,
    load_vqvae,
    time_embedding,
    update_ema,
)
from task4_vqvae_simple import ROOT, get_device


SEED = 42


def modulate(inputs, shift, scale):
    return inputs * (1 + scale[:, None]) + shift[:, None]


class AdaptiveTransformerBlock(nn.Module):
    """Pre-LN bidirectional attention with timestep/class adaptive norms."""

    def __init__(self, width, heads):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(width, width * 4)
        )

    def forward(self, inputs, condition):
        shift1, scale1, shift2, scale2 = self.modulation(condition).chunk(4, 1)
        hidden = modulate(self.attention_norm(inputs), shift1, scale1)
        hidden = self.attention(hidden, hidden, hidden, need_weights=False)[0]
        inputs = inputs + hidden
        hidden = self.mlp(modulate(self.mlp_norm(inputs), shift2, scale2))
        return inputs + hidden


class DiscreteDiffusionTransformer(nn.Module):
    def __init__(self, codebook_size, width=320, depth=10, heads=8):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.codebook_size = codebook_size
        self.mask_id = codebook_size
        self.width = width
        self.token_embedding = nn.Embedding(codebook_size + 1, width)
        self.position_embedding = nn.Parameter(torch.randn(1, 64, width) * 0.02)
        self.class_embedding = nn.Embedding(11, width)
        self.time_network = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.SiLU(),
            nn.Linear(width * 4, width),
        )
        self.blocks = nn.ModuleList(
            AdaptiveTransformerBlock(width, heads) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(width, width * 2)
        )
        self.output = nn.Linear(width, codebook_size)

    def forward(self, noisy_indices, timesteps, labels):
        batch = len(noisy_indices)
        tokens = noisy_indices.flatten(1)
        hidden = self.token_embedding(tokens) + self.position_embedding
        condition = self.time_network(
            time_embedding(timesteps, self.width)
        ) + self.class_embedding(labels)
        for block in self.blocks:
            hidden = block(hidden, condition)
        shift, scale = self.final_modulation(condition).chunk(2, 1)
        logits = self.output(modulate(self.final_norm(hidden), shift, scale))
        return logits.view(batch, 8, 8, self.codebook_size).permute(0, 3, 1, 2)


@torch.no_grad()
def sample_indices(
    model,
    labels,
    steps,
    train_timesteps,
    guidance,
    temperature,
    choice_temperature,
):
    batch = len(labels)
    device = labels.device
    indices = torch.full(
        (batch, 8, 8), model.mask_id, dtype=torch.long, device=device
    )
    null_labels = torch.full_like(labels, NULL_CLASS)

    for iteration in range(steps):
        masked = indices.eq(model.mask_id)
        mask_ratio = masked.flatten(1).float().mean(1)
        timesteps = (
            mask_ratio * train_timesteps
        ).round().clamp(1, train_timesteps).long()
        doubled_indices = torch.cat([indices, indices])
        doubled_times = torch.cat([timesteps, timesteps])
        doubled_labels = torch.cat([null_labels, labels])
        unconditional, conditional = model(
            doubled_indices, doubled_times, doubled_labels
        ).chunk(2)
        progress = iteration / max(steps - 1, 1)
        current_guidance = 1 + (guidance - 1) * progress
        logits = unconditional + current_guidance * (
            conditional - unconditional
        )

        probabilities = F.softmax(logits / temperature, dim=1)
        sampled = torch.multinomial(
            probabilities.permute(0, 2, 3, 1).reshape(-1, model.codebook_size),
            1,
        ).view(batch, 8, 8)
        confidence = probabilities.gather(1, sampled[:, None]).squeeze(1)
        confidence = confidence.clamp_min(1e-12).log()
        indices[masked] = sampled[masked]

        next_ratio = math.cos(math.pi * (iteration + 1) / (2 * steps))
        next_count = round(64 * next_ratio)
        if iteration == steps - 1:
            next_count = 0
        noise_scale = choice_temperature * (1 - progress)
        uniform = torch.rand_like(confidence).clamp_(1e-6, 1 - 1e-6)
        gumbel = -torch.log(-torch.log(uniform))
        scores = confidence + noise_scale * gumbel
        scores[~masked] = torch.inf

        if iteration < steps - 1:
            remaining = int(masked.flatten(1).sum(1).min())
            next_count = min(next_count, max(remaining - 1, 0))
        if next_count:
            for image_index in range(batch):
                remask = scores[image_index].flatten().topk(
                    next_count, largest=False
                ).indices
                indices[image_index].view(-1)[remask] = model.mask_id
    return indices


@torch.no_grad()
def generate(model, vqvae, args, device):
    model.eval()
    images = []
    for start in range(0, args.generate, min(args.batch_size, 128)):
        current = min(min(args.batch_size, 128), args.generate - start)
        labels = torch.arange(start, start + current, device=device) % 10
        indices = sample_indices(
            model,
            labels,
            args.sample_steps,
            args.timesteps,
            args.guidance,
            args.temperature,
            args.choice_temperature,
        )
        decoded = vqvae.decode_indices(indices).clamp(0, 1)
        images.append((decoded * 255).round().byte().cpu())
    return torch.cat(images)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--class-dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--augmentation", choices=("none", "pixel-flip"), default="pixel-flip")
    parser.add_argument("--generate", type=int, default=10000)
    parser.add_argument("--sample-steps", type=int, default=12)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--choice-temperature", type=float, default=4.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fake-data", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    positive = (
        args.epochs, args.batch_size, args.learning_rate, args.timesteps,
        args.train_size, args.width, args.depth, args.heads, args.generate,
        args.sample_steps, args.temperature,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("training, model, and sampling arguments must be positive")
    if not 0 <= args.class_dropout <= 1:
        raise ValueError("class-dropout must be in [0, 1]")
    if args.fake_data and args.train_size == 50000:
        args.train_size = 64

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    device = get_device()
    checkpoint = find_vqvae(args.vqvae)
    vqvae, codebook_size = load_vqvae(checkpoint, device)
    transform = transforms.ToTensor()
    if args.fake_data:
        dataset = datasets.FakeData(
            args.train_size, (3, 32, 32), 10, transform, random_offset=0
        )
    else:
        dataset = datasets.CIFAR10(
            ROOT / "data", train=True, download=True, transform=transform
        )
        args.train_size = min(args.train_size, len(dataset))
        if args.train_size < len(dataset):
            dataset = torch.utils.data.Subset(dataset, range(args.train_size))

    run_dir = Path(args.output).expanduser() if args.output else (
        ROOT / "outputs" / "task4_vq_diffusion_final"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    latents, flipped_latents, labels = encode_dataset(
        vqvae, dataset, args.batch_size, device, args.augmentation
    )
    loader = DataLoader(
        TensorDataset(latents, flipped_latents, labels),
        args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    model = DiscreteDiffusionTransformer(
        codebook_size, args.width, args.depth, args.heads
    ).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    warmup = max(1, round(args.epochs * 0.05))

    def learning_rate_scale(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(args.epochs - warmup, 1)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, learning_rate_scale
    )
    history = []
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Device: {device} | VQ-VAE: {checkpoint} | "
        f"Latents: {list(latents.shape)} | Codes: {codebook_size} | "
        f"Parameters: {parameter_count:,} | Adaptive pre-LN: True | "
        f"Augmentation: {args.augmentation}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        masked_count = 0
        for clean, flipped, batch_labels in loader:
            clean = clean.long().to(device, non_blocking=True)
            if args.augmentation == "pixel-flip":
                use_flip = torch.rand(len(clean), device=device) < 0.5
                flipped = flipped.long().to(device, non_blocking=True)
                clean[use_flip] = flipped[use_flip]
            batch_labels = batch_labels.to(device, non_blocking=True)
            timesteps = torch.randint(
                1, args.timesteps + 1, (len(clean),), device=device
            )
            noisy, mask = add_noise(
                clean, timesteps, args.timesteps, model.mask_id
            )
            conditioned_labels = batch_labels.clone()
            drop = torch.rand(len(clean), device=device) < args.class_dropout
            conditioned_labels[drop] = NULL_CLASS
            logits = model(noisy, timesteps, conditioned_labels)
            token_loss = F.cross_entropy(logits, clean, reduction="none")
            loss = token_loss[mask].mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema, model, args.ema_decay)
            loss_sum += token_loss[mask].sum().item()
            masked_count += mask.sum().item()

        scheduler.step()
        mean_loss = loss_sum / masked_count
        history.append([epoch, mean_loss, optimizer.param_groups[0]["lr"]])
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            torch.save(ema.state_dict(), run_dir / "diffusion_ema.pt")
            print(
                f"Epoch {epoch:03d}/{args.epochs} | Masked CE {mean_loss:.5f} | "
                f"LR {optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )

    generated = generate(ema, vqvae, args, device)
    torch.save(ema.state_dict(), run_dir / "diffusion_ema.pt")
    torch.save(generated, run_dir / "generated_images.pt")
    save_image(generated[:64].float() / 255, run_dir / "generated_grid.png", nrow=8)
    summary = (
        f"Dataset: {'FakeData smoke test' if args.fake_data else 'CIFAR-10'}\n"
        f"VQ-VAE: {checkpoint}\n"
        f"Discrete latent: 8x8 integer indices, codebook={codebook_size}, embedding channels=10\n"
        f"Diffusion: absorbing MASK, timesteps={args.timesteps}\n"
        f"Denoiser: adaptive pre-LN Transformer, width={args.width}, "
        f"depth={args.depth}, heads={args.heads}, parameters={parameter_count}\n"
        f"Epochs: {args.epochs} | Final masked CE: {history[-1][1]:.6f}\n"
        f"Generated: {len(generated)} | Steps: {args.sample_steps} | "
        f"Guidance: {args.guidance} | Temperature: {args.temperature}\n"
    )
    (run_dir / "results.txt").write_text(summary, encoding="utf-8")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "masked_cross_entropy", "learning_rate"])
        writer.writerows(history)
    print("\n" + summary, flush=True)
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
