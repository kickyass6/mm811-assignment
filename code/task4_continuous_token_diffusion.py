"""Continuous latent Transformer with a conditional per-token diffusion head.

The model is autoregressive across the 64 cells of a 5x8x8 continuous latent.
Instead of fitting an explicit Gaussian mixture, each causal Transformer state
conditions a small denoising network that samples the next five-dimensional
cell with DDIM/DDPM-style updates.
"""

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from task4_models import ROOT, ContinuousAutoencoder


SEED = 42
LATENT_CHANNELS = 5
SEQUENCE_LENGTH = 64


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-width", type=int, default=512)
    parser.add_argument("--head-depth", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-repeats", type=int, default=4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--augmentation", choices=("none", "pixel-flip"), default="pixel-flip")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-eta", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--latent-clip", type=float, default=6.0)
    parser.add_argument("--generate", type=int, default=10000)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def timestep_embedding(timesteps, width=256):
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device).float() / max(half, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([angles.cos(), angles.sin()], dim=1)
    return F.pad(embedding, (0, width - embedding.shape[1]))


def cosine_alpha_cumprod(total_steps, offset=0.008):
    grid = torch.arange(total_steps + 1, dtype=torch.float64)
    signal = torch.cos(((grid / total_steps) + offset) / (1 + offset) * math.pi / 2).square()
    signal = signal / signal[0]
    betas = (1 - signal[1:] / signal[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


class AdaptiveMLPBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.normalization = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.network = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 3))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, inputs, condition):
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        hidden = self.normalization(inputs) * (1 + scale) + shift
        return inputs + gate * self.network(hidden)


class TokenDiffusionHead(nn.Module):
    def __init__(self, condition_width, width, depth):
        super().__init__()
        self.input_projection = nn.Linear(LATENT_CHANNELS, width)
        self.condition_projection = nn.Linear(condition_width, width)
        self.time_network = nn.Sequential(
            nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList([AdaptiveMLPBlock(width) for _ in range(depth)])
        self.final_normalization = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 2))
        self.output = nn.Linear(width, LATENT_CHANNELS)
        for layer in (self.final_modulation[-1], self.output):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, noisy_token, timesteps, condition):
        context = self.condition_projection(condition) + self.time_network(
            timestep_embedding(timesteps)
        )
        hidden = self.input_projection(noisy_token)
        for block in self.blocks:
            hidden = block(hidden, context)
        shift, scale = self.final_modulation(context).chunk(2, dim=-1)
        hidden = self.final_normalization(hidden) * (1 + scale) + shift
        return self.output(hidden)


class AutoregressiveTokenDiffusion(nn.Module):
    def __init__(
        self, width=512, layers=8, heads=8, dropout=0.1,
        head_width=512, head_depth=4, diffusion_steps=1000,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.config = {
            "width": width, "layers": layers, "heads": heads, "dropout": dropout,
            "head_width": head_width, "head_depth": head_depth,
            "diffusion_steps": diffusion_steps,
        }
        self.diffusion_steps = diffusion_steps
        self.input_projection = nn.Linear(LATENT_CHANNELS, width)
        self.beginning = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1, SEQUENCE_LENGTH, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.output_normalization = nn.LayerNorm(width)
        self.diffusion_head = TokenDiffusionHead(width, head_width, head_depth)
        self.register_buffer(
            "alpha_cumprod", cosine_alpha_cumprod(diffusion_steps), persistent=False
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(SEQUENCE_LENGTH, SEQUENCE_LENGTH, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def condition(self, sequence):
        batch = sequence.shape[0]
        inputs = torch.cat([
            self.beginning.expand(batch, -1, -1),
            self.input_projection(sequence[:, :-1]),
        ], dim=1)
        inputs = inputs + self.position_embedding
        hidden = self.transformer(inputs, mask=self.causal_mask, is_causal=True)
        return self.output_normalization(hidden)

    def diffusion_loss(self, sequence, repeats=4):
        batch, length, channels = sequence.shape
        condition = self.condition(sequence).reshape(batch * length, -1).repeat(repeats, 1)
        clean = sequence.reshape(batch * length, channels).repeat(repeats, 1)
        timesteps = torch.randint(
            0, self.diffusion_steps, (len(clean),), device=sequence.device
        )
        noise = torch.randn_like(clean)
        alpha = self.alpha_cumprod[timesteps, None]
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        prediction = self.diffusion_head(noisy, timesteps, condition)
        return F.mse_loss(prediction.float(), noise.float())

    @torch.no_grad()
    def sample(self, count, steps=50, eta=1.0, temperature=1.0, clip=6.0):
        device = next(self.parameters()).device
        sequence = torch.zeros(count, SEQUENCE_LENGTH, LATENT_CHANNELS, device=device)
        schedule = torch.linspace(
            self.diffusion_steps - 1, 0, steps, device=device
        ).round().long()
        one = torch.tensor(1.0, device=device)
        for position in range(SEQUENCE_LENGTH):
            condition = self.condition(sequence)[:, position]
            token = torch.randn(count, LATENT_CHANNELS, device=device) * temperature
            for index, timestep in enumerate(schedule):
                alpha = self.alpha_cumprod[timestep]
                final_step = index + 1 == len(schedule)
                previous_alpha = one if final_step else self.alpha_cumprod[schedule[index + 1]]
                prediction = self.diffusion_head(
                    token, timestep.expand(count), condition
                ).float()
                clean = (
                    (token - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
                ).clamp(-clip, clip)
                prediction = (token - alpha.sqrt() * clean) / (1 - alpha).sqrt()
                sigma = eta * ((1 - previous_alpha) / (1 - alpha)).sqrt()
                sigma = sigma * (1 - alpha / previous_alpha).clamp_min(0).sqrt()
                direction = (1 - previous_alpha - sigma.square()).clamp_min(0).sqrt() * prediction
                random_noise = (
                    torch.zeros_like(token) if final_step
                    else torch.randn_like(token) * temperature
                )
                token = previous_alpha.sqrt() * clean + direction + sigma * random_noise
            sequence[:, position] = token
        return sequence


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.decay = decay
        self.steps = 0
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for averaged, current in zip(self.model.parameters(), model.parameters()):
            averaged.mul_(decay).add_(current.detach(), alpha=1 - decay)
        for averaged, current in zip(self.model.buffers(), model.buffers()):
            averaged.copy_(current)


def make_datasets(args):
    transform = transforms.ToTensor()
    if args.fake_data:
        train = datasets.FakeData(
            args.train_size, image_size=(3, 32, 32), num_classes=10,
            transform=transform, random_offset=0,
        )
        test = datasets.FakeData(
            args.test_size, image_size=(3, 32, 32), num_classes=10,
            transform=transform, random_offset=100000,
        )
    else:
        train = datasets.CIFAR10(ROOT / "data", train=True, download=True, transform=transform)
        test = datasets.CIFAR10(ROOT / "data", train=False, download=True, transform=transform)
    if args.train_size < len(train):
        train = torch.utils.data.Subset(train, list(range(args.train_size)))
    if args.test_size < len(test):
        test = torch.utils.data.Subset(test, list(range(args.test_size)))
    return train, test


@torch.no_grad()
def encode_images(dataset, autoencoder, device, batch_size, workers, flip=False):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    encoded = []
    autoencoder.eval()
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        if flip:
            images = images.flip(-1)
        latent = autoencoder.encode(images).float().cpu()
        encoded.append(latent)
    return torch.cat(encoded)


def normalize_latents(latents, mean, std):
    normalized = (latents - mean[None, :, None, None]) / std[None, :, None, None]
    return normalized.permute(0, 2, 3, 1).reshape(len(latents), SEQUENCE_LENGTH, LATENT_CHANNELS)


@torch.no_grad()
def evaluate_loss(model, sequences, device, batch_size):
    model.eval()
    total = count = 0.0
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        TensorDataset(sequences), batch_size=batch_size, shuffle=False,
        generator=generator,
    )
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(0)
        for (batch,) in loader:
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model.diffusion_loss(batch, repeats=1)
            total += loss.item() * len(batch)
            count += len(batch)
    return total / count


@torch.no_grad()
def generate_images(model, decoder, mean, std, args, device, count):
    model.eval()
    decoder.eval()
    images = []
    for start in range(0, count, args.generation_batch_size):
        batch = min(args.generation_batch_size, count - start)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            normalized = model.sample(
                batch, steps=args.sample_steps, eta=args.sample_eta,
                temperature=args.temperature, clip=args.latent_clip,
            )
            latent = normalized.view(batch, 8, 8, LATENT_CHANNELS).permute(0, 3, 1, 2)
            latent = latent.float() * std.to(device)[None, :, None, None]
            latent = latent + mean.to(device)[None, :, None, None]
            decoded = decoder(latent).clamp(0, 1)
        images.append((decoded * 255).round().byte().cpu())
        print(f"Generated {min(start + batch, count)}/{count}", flush=True)
    return torch.cat(images)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    autoencoder = ContinuousAutoencoder().to(device)
    state = torch.load(args.autoencoder, map_location="cpu", weights_only=True)
    autoencoder.load_state_dict(state)
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    train_set, test_set = make_datasets(args)
    clean_latent = encode_images(
        train_set, autoencoder, device, args.batch_size * 2, args.num_workers
    )
    mean = clean_latent.mean(dim=(0, 2, 3))
    std = clean_latent.std(dim=(0, 2, 3)).clamp_min(1e-5)
    train_sequences = [normalize_latents(clean_latent, mean, std)]
    if args.augmentation == "pixel-flip":
        flipped_latent = encode_images(
            train_set, autoencoder, device, args.batch_size * 2,
            args.num_workers, flip=True,
        )
        train_sequences.append(normalize_latents(flipped_latent, mean, std))
    train_sequences = torch.cat(train_sequences).contiguous()
    test_latent = encode_images(
        test_set, autoencoder, device, args.batch_size * 2, args.num_workers
    )
    test_sequences = normalize_latents(test_latent, mean, std).contiguous()
    if args.eval_size < len(test_sequences):
        test_sequences = test_sequences[:args.eval_size]
    del clean_latent, test_latent
    torch.save({"mean": mean, "std": std}, output / "latent_stats.pt")

    loader = DataLoader(
        TensorDataset(train_sequences), batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = AutoregressiveTokenDiffusion(
        width=args.width, layers=args.layers, heads=args.heads,
        dropout=args.dropout, head_width=args.head_width,
        head_depth=args.head_depth, diffusion_steps=args.diffusion_steps,
    ).to(device)
    ema = ExponentialMovingAverage(model, args.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )
    total_steps = args.epochs * len(loader)

    def learning_rate_multiplier(step):
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Device: {device} | Training sequences: {len(train_sequences)} | "
        f"Validation sequences: {len(test_sequences)} | Parameters: {parameter_count:,}",
        flush=True,
    )

    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = count = 0
        for (sequence,) in loader:
            sequence = sequence.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model.diffusion_loss(sequence, repeats=args.noise_repeats)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            total_loss += loss.item() * len(sequence)
            count += len(sequence)
        validation_loss = evaluate_loss(
            ema.model, test_sequences, device, args.batch_size * 2
        )
        train_loss = total_loss / count
        history.append((epoch, train_loss, validation_loss, optimizer.param_groups[0]["lr"]))
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train noise MSE {train_loss:.6f} | "
            f"validation noise MSE {validation_loss:.6f} | "
            f"LR {optimizer.param_groups[0]['lr']:.8g}", flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(ema.model.state_dict(), output / "transformer_ema_best.pt")
        torch.save(ema.model.state_dict(), output / "transformer_ema_last.pt")
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save(
                ema.model.state_dict(),
                output / f"transformer_ema_epoch_{epoch:03d}.pt",
            )
        if args.preview_every > 0 and (epoch % args.preview_every == 0 or epoch == args.epochs):
            preview = generate_images(ema.model, autoencoder.decoder, mean, std, args, device, 64)
            save_image(preview.float() / 255, output / f"preview_epoch_{epoch:03d}.png", nrow=8)
            del preview

    ema.model.load_state_dict(
        torch.load(output / "transformer_ema_best.pt", map_location=device, weights_only=True)
    )
    generated = generate_images(
        ema.model, autoencoder.decoder, mean, std, args, device, args.generate
    )
    torch.save(generated, output / "generated_images.pt")
    save_image(generated[:64].float() / 255, output / "generated_grid.png", nrow=8)
    torch.save(ema.model.state_dict(), output / "transformer_ema.pt")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_noise_mse", "validation_noise_mse", "learning_rate"))
        writer.writerows(history)
    summary = (
        "Dataset: CIFAR-10\n"
        f"Autoencoder: {Path(args.autoencoder).resolve()} | Continuous latent: 8x8x5\n"
        "Architecture: causal pre-LN Transformer with a conditional per-token diffusion head\n"
        f"Model: width={args.width}, layers={args.layers}, heads={args.heads}, "
        f"parameters={parameter_count}\n"
        f"Epochs: {args.epochs} | Best validation noise MSE: {best_loss:.6f}\n"
        f"Sampling: steps={args.sample_steps}, eta={args.sample_eta}, "
        f"temperature={args.temperature}, clip={args.latent_clip}\n"
        f"Generated: {len(generated)}\n"
    )
    (output / "results.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)
    print(f"Results saved to: {output}", flush=True)


if __name__ == "__main__":
    main()
