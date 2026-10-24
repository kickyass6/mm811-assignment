"""任务 4 共用模型与工具。所有图像张量范围均为 [0, 1]。"""

import math
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
CONTINUOUS_CHANNELS = 5
DISCRETE_CHANNELS = 10
LATENT_SIZE = 8


def device_name():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = min(16, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs):
        return inputs + self.block(inputs)


class Encoder(nn.Module):
    def __init__(self, latent_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1),
            ResidualBlock(128),
            nn.Conv2d(128, 256, 4, 2, 1),
            ResidualBlock(256), ResidualBlock(256),
            nn.GroupNorm(16, 256), nn.SiLU(),
            nn.Conv2d(256, latent_channels, 1),
        )

    def forward(self, images):
        return self.network(images)


class Decoder(nn.Module):
    def __init__(self, latent_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 3, padding=1),
            ResidualBlock(256), ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            ResidualBlock(64),
            nn.GroupNorm(16, 64), nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, latent):
        return self.network(latent)


class ContinuousAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(CONTINUOUS_CHANNELS)
        self.decoder = Decoder(CONTINUOUS_CHANNELS)

    def encode(self, images):
        return self.encoder(images)

    def forward(self, images):
        return self.decoder(self.encode(images))


class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        return torch.where(logits >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class DiscreteAutoencoder(nn.Module):
    """8×8×10 个严格 -1/+1 值，满足作业的离散 latent 上限。"""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(DISCRETE_CHANNELS)
        self.decoder = Decoder(DISCRETE_CHANNELS)

    def encode(self, images, hard=True, steepness=5.0):
        logits = self.encoder(images)
        if hard:
            return BinarySTE.apply(logits)
        return torch.tanh(steepness * logits)

    def forward(self, images, hard=True, steepness=5.0):
        return self.decoder(self.encode(images, hard=hard, steepness=steepness))


@torch.no_grad()
def reconstruction_metrics(model, loader, device, discrete=False):
    model.eval()
    mse_sum = psnr_sum = 0.0
    for images, _ in loader:
        images = images.to(device)
        reconstructed = model(images, hard=True) if discrete else model(images)
        image_mse = (images - reconstructed).square().mean(dim=(1, 2, 3))
        mse_sum += image_mse.sum().item()
        psnr_sum += (-10 * torch.log10(image_mse.clamp_min(1e-10))).sum().item()
    count = len(loader.dataset)
    return mse_sum / count, psnr_sum / count


def sinusoidal_embedding(timesteps, width):
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    if width % 2:
        embedding = nn.functional.pad(embedding, (0, 1))
    return embedding


class ConditionalResidualBlock(nn.Module):
    def __init__(self, channels, condition_width):
        super().__init__()
        self.normalization1 = nn.GroupNorm(16, channels)
        self.convolution1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition_width, channels)
        self.normalization2 = nn.GroupNorm(16, channels)
        self.convolution2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, inputs, condition):
        hidden = self.convolution1(nn.functional.silu(self.normalization1(inputs)))
        hidden = hidden + self.condition(condition)[:, :, None, None]
        hidden = self.convolution2(nn.functional.silu(self.normalization2(hidden)))
        return inputs + hidden


class BinaryLatentDenoiser(nn.Module):
    """预测被离散扩散破坏之前的10通道二值 latent。"""

    def __init__(self, timesteps=32, width=192, blocks=6, classes=10):
        super().__init__()
        self.timesteps = timesteps
        self.width = width
        self.input = nn.Conv2d(DISCRETE_CHANNELS, width, 3, padding=1)
        self.time_network = nn.Sequential(
            nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width)
        )
        self.class_embedding = nn.Embedding(classes, width)
        self.blocks = nn.ModuleList([
            ConditionalResidualBlock(width, width) for _ in range(blocks)
        ])
        self.output = nn.Sequential(
            nn.GroupNorm(16, width), nn.SiLU(),
            nn.Conv2d(width, DISCRETE_CHANNELS, 3, padding=1),
        )

    def forward(self, noisy_latent, timesteps, labels):
        condition = self.time_network(sinusoidal_embedding(timesteps, self.width))
        condition = condition + self.class_embedding(labels)
        hidden = self.input(noisy_latent)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden)


class ContinuousLatentTransformer(nn.Module):
    """按64个空间位置自回归建模，每个位置是5维连续向量。"""

    def __init__(self, width=256, heads=8, layers=6, classes=10):
        super().__init__()
        self.width = width
        self.input_projection = nn.Linear(CONTINUOUS_CHANNELS, width)
        self.position_embedding = nn.Embedding(LATENT_SIZE * LATENT_SIZE, width)
        self.class_embedding = nn.Embedding(classes, width)
        self.start = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.0, batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.normalization = nn.LayerNorm(width)
        self.output = nn.Linear(width, CONTINUOUS_CHANNELS * 2)

    def forward(self, previous_latents, labels):
        batch, length, _ = previous_latents.shape
        if length == 0:
            encoded = self.start.expand(batch, 1, -1)
        else:
            shifted = self.input_projection(previous_latents)
            encoded = torch.cat([self.start.expand(batch, 1, -1), shifted], dim=1)
        sequence_length = encoded.size(1)
        positions = torch.arange(sequence_length, device=encoded.device)
        encoded = encoded + self.position_embedding(positions)[None]
        encoded = encoded + self.class_embedding(labels)[:, None]
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=encoded.device),
            diagonal=1,
        )
        hidden = self.transformer(encoded, mask=mask)
        parameters = self.output(self.normalization(hidden))
        mean, log_scale = parameters.chunk(2, dim=-1)
        return mean, log_scale.clamp(-5.0, 1.0)


def latest_autoencoder_run(path=None):
    if path:
        run = Path(path).expanduser().resolve()
    else:
        candidates = sorted((ROOT / "outputs" / "task4_autoencoders").glob("*/continuous.pt"))
        if not candidates:
            raise FileNotFoundError("请先运行 task4_autoencoders.py")
        run = candidates[-1].parent
    required = (run / "continuous.pt", run / "discrete.pt")
    if not all(item.is_file() for item in required):
        raise FileNotFoundError(f"自编码器权重不完整：{run}")
    return run
