"""任务 3：用任务 2 的二值 latent 训练三种自回归模型。"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from task2_optimized import OptimizedBinaryAutoencoder


ROOT = Path(__file__).resolve().parent
SEED = 42
TOKEN_VALUES = 8                 # 三个 bit 可以组成 0～7
BOS = 8
PAD = 9
INPUT_VOCAB_SIZE = 10            # 0～7，加 BOS 和 PAD
SEQUENCE_LENGTH = 16             # 4×4
SIMPLE_MODEL_IMAGES = 100
HISTOGRAM_IMAGES = 1024          # 超过作业要求的 512


def find_checkpoint(user_path):
    if user_path:
        checkpoint = Path(user_path).expanduser().resolve()
    else:
        candidates = sorted((ROOT / "outputs" / "task2_optimized").glob("*/final_epoch.pt"))
        if not candidates:
            raise FileNotFoundError("没有找到 task2_optimized 的 final_epoch.pt")
        checkpoint = candidates[-1]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def latent_to_tokens(latent):
    """[N,3,4,4] 的 -1/+1 latent 转成 [N,16]、取值 0～7 的 token。"""
    bits = (latent > 0).long()
    tokens = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    return tokens.flatten(1)


def tokens_to_latent(tokens):
    """把 [N,16] token 无损还原成 [N,3,4,4] 的 -1/+1 latent。"""
    bits = torch.stack(
        ((tokens >> 2) & 1, (tokens >> 1) & 1, tokens & 1), dim=1
    )
    return bits.reshape(-1, 3, 4, 4).float() * 2 - 1


@torch.no_grad()
def extract_tokens(autoencoder, dataset, indices, device):
    loader = DataLoader(Subset(dataset, indices), batch_size=64)
    batches = []
    autoencoder.eval()
    for images, _ in loader:
        latent = autoencoder.encode(images.to(device), hard=True)
        batches.append(latent_to_tokens(latent).cpu())
    tokens = torch.cat(batches)
    # 确认十进制 token 可以无损恢复原来的48个二值。
    assert torch.equal(tokens, latent_to_tokens(tokens_to_latent(tokens)))
    return tokens


class SequenceToToken(nn.Module):
    """输入一个前缀序列，GRU只输出下一个 token 的8分类结果。"""

    def __init__(self, embedding_size=64, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(INPUT_VOCAB_SIZE, embedding_size, padding_idx=PAD)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, TOKEN_VALUES)

    def forward(self, prefixes, lengths):
        encoded, _ = self.gru(self.embedding(prefixes))
        last = encoded[torch.arange(len(prefixes), device=prefixes.device), lengths - 1]
        return self.output(last)


class SequenceToSequence(nn.Module):
    """使用 causal mask 的 Transformer，一次输出所有位置的 token 分布。"""

    def __init__(self, width=128, heads=4, layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(INPUT_VOCAB_SIZE, width)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH, width)
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.0, batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)
        self.normalization = nn.LayerNorm(width)
        self.output = nn.Linear(width, TOKEN_VALUES)

    def forward(self, inputs):
        length = inputs.size(1)
        positions = torch.arange(length, device=inputs.device)
        encoded = self.token_embedding(inputs) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=inputs.device), diagonal=1
        )
        encoded = self.transformer(encoded, mask=causal_mask)
        return self.output(self.normalization(encoded))


class SequenceToHistogram(SequenceToToken):
    """输入前缀，输出下一个 token 的8格概率直方图。"""

    pass


def make_prefix_dataset(tokens):
    """每条长度16的序列产生16个“前缀 -> 下一 token”样本。"""
    inputs, lengths, targets = [], [], []
    for sequence in tokens.tolist():
        for position in range(SEQUENCE_LENGTH):
            prefix = [BOS] + sequence[:position]
            padded = prefix + [PAD] * (SEQUENCE_LENGTH - len(prefix))
            inputs.append(padded)
            lengths.append(len(prefix))
            targets.append(sequence[position])
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def train_sequence_to_token(model, tokens, epochs, device):
    inputs, lengths, targets = make_prefix_dataset(tokens)
    loader = DataLoader(
        TensorDataset(inputs, lengths, targets), batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for prefixes, prefix_lengths, next_tokens in loader:
            prefixes = prefixes.to(device)
            prefix_lengths = prefix_lengths.to(device)
            next_tokens = next_tokens.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(prefixes, prefix_lengths), next_tokens)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(prefixes)
        mean_loss = loss_sum / len(inputs)
        history.append(mean_loss)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"Sequence-to-Token epoch {epoch:03d}/{epochs} | loss {mean_loss:.4f}", flush=True)
    return history


def train_sequence_to_sequence(model, tokens, epochs, device):
    # 输入右移一位；第 i 个输出只能看见 BOS 和第 i 个 token 之前的内容。
    inputs = torch.cat(
        [torch.full((len(tokens), 1), BOS, dtype=torch.long), tokens[:, :-1]], dim=1
    ).to(device)
    targets = tokens.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits.reshape(-1, TOKEN_VALUES), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"Sequence-to-Sequence epoch {epoch:03d}/{epochs} | loss {loss.item():.4f}", flush=True)
    return history


def sample_from_logits(logits, temperature):
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, 1).squeeze(1)


@torch.no_grad()
def generate_sequence_to_token(model, count, temperature, device):
    model.eval()
    generated = torch.empty((count, 0), dtype=torch.long, device=device)
    for _ in range(SEQUENCE_LENGTH):
        prefixes = torch.full((count, SEQUENCE_LENGTH), PAD, dtype=torch.long, device=device)
        prefixes[:, 0] = BOS
        if generated.size(1):
            prefixes[:, 1:1 + generated.size(1)] = generated
        lengths = torch.full((count,), generated.size(1) + 1, dtype=torch.long, device=device)
        next_token = sample_from_logits(model(prefixes, lengths), temperature)
        generated = torch.cat([generated, next_token[:, None]], dim=1)
    return generated.cpu()


@torch.no_grad()
def generate_sequence_to_sequence(model, count, temperature, device):
    model.eval()
    inputs = torch.full((count, 1), BOS, dtype=torch.long, device=device)
    generated = torch.empty((count, 0), dtype=torch.long, device=device)
    for _ in range(SEQUENCE_LENGTH):
        next_token = sample_from_logits(model(inputs)[:, -1], temperature)
        generated = torch.cat([generated, next_token[:, None]], dim=1)
        if generated.size(1) < SEQUENCE_LENGTH:
            inputs = torch.cat([inputs, next_token[:, None]], dim=1)
    return generated.cpu()


def build_histogram_dataset(tokens):
    """把相同前缀合并，产生“前缀 -> 8格经验概率直方图”训练集。"""
    histograms = defaultdict(lambda: torch.zeros(TOKEN_VALUES, dtype=torch.long))
    for sequence in tokens.tolist():
        for position in range(SEQUENCE_LENGTH):
            prefix = tuple(sequence[:position])
            histograms[prefix][sequence[position]] += 1

    inputs, lengths, targets = [], [], []
    for prefix, counts in histograms.items():
        sequence = [BOS] + list(prefix)
        inputs.append(sequence + [PAD] * (SEQUENCE_LENGTH - len(sequence)))
        lengths.append(len(sequence))
        targets.append(counts.float() / counts.sum())
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.stack(targets),
    )


def train_sequence_to_histogram(model, tokens, epochs, device):
    inputs, lengths, target_histograms = build_histogram_dataset(tokens)
    loader = DataLoader(
        TensorDataset(inputs, lengths, target_histograms), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for prefixes, prefix_lengths, histograms in loader:
            prefixes = prefixes.to(device)
            prefix_lengths = prefix_lengths.to(device)
            histograms = histograms.to(device)
            optimizer.zero_grad()
            log_probabilities = nn.functional.log_softmax(
                model(prefixes, prefix_lengths), dim=1
            )
            loss = -(histograms * log_probabilities).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(prefixes)
        mean_loss = loss_sum / len(inputs)
        history.append(mean_loss)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f"Sequence-to-Histogram epoch {epoch:03d}/{epochs} | loss {mean_loss:.4f}",
                flush=True,
            )
    return history


@torch.no_grad()
def generate_sequence_to_histogram(model, count, temperature, device):
    # 网络输出8个 logits；softmax 后就是预测的下一-token直方图。
    return generate_sequence_to_token(model, count, temperature, device)


def memorization_rate(generated, training_tokens):
    training_set = {tuple(sequence) for sequence in training_tokens.tolist()}
    matches = sum(tuple(sequence) in training_set for sequence in generated.tolist())
    return matches / len(generated), matches, len(training_set)


@torch.no_grad()
def save_generated_images(autoencoder, generated, path, device):
    latent = tokens_to_latent(generated[:64]).to(device)
    images = autoencoder.decoder(latent).cpu()
    save_image(images, path, nrow=8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--token-epochs", type=int, default=400)
    parser.add_argument("--sequence-epochs", type=int, default=400)
    parser.add_argument("--histogram-epochs", type=int, default=400)
    parser.add_argument("--generate", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.token_epochs, args.sequence_epochs, args.histogram_epochs, args.generate) < 1:
        raise ValueError("epochs 和生成数量必须为正数")
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint = find_checkpoint(args.checkpoint)
    print(f"Device: {device}\nTask 2 checkpoint: {checkpoint}", flush=True)

    autoencoder = OptimizedBinaryAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    autoencoder.eval()
    autoencoder.requires_grad_(False)

    mnist = datasets.MNIST(ROOT / "data", train=True, download=True, transform=transforms.ToTensor())
    indices = torch.randperm(len(mnist), generator=torch.Generator().manual_seed(SEED))
    available = torch.cat([indices[:256], indices[512:]])
    histogram_tokens = extract_tokens(
        autoencoder, mnist, available[:HISTOGRAM_IMAGES].tolist(), device
    )
    simple_tokens = histogram_tokens[:SIMPLE_MODEL_IMAGES]
    print(
        f"Tokens: simple models {len(simple_tokens)} images, histogram {len(histogram_tokens)} images | "
        f"shape {list(simple_tokens.shape)} | values {simple_tokens.unique().tolist()}",
        flush=True,
    )

    token_model = SequenceToToken().to(device)
    token_history = train_sequence_to_token(token_model, simple_tokens, args.token_epochs, device)
    torch.manual_seed(SEED + 1)
    generated_token = generate_sequence_to_token(
        token_model, args.generate, args.temperature, device
    )

    sequence_model = SequenceToSequence().to(device)
    sequence_history = train_sequence_to_sequence(
        sequence_model, simple_tokens, args.sequence_epochs, device
    )
    torch.manual_seed(SEED + 2)
    generated_sequence = generate_sequence_to_sequence(
        sequence_model, args.generate, args.temperature, device
    )

    histogram_model = SequenceToHistogram(hidden_size=256).to(device)
    histogram_history = train_sequence_to_histogram(
        histogram_model, histogram_tokens, args.histogram_epochs, device
    )
    torch.manual_seed(SEED + 3)
    generated_histogram = generate_sequence_to_histogram(
        histogram_model, args.generate, args.temperature, device
    )

    token_rate, token_matches, token_unique = memorization_rate(generated_token, simple_tokens)
    sequence_rate, sequence_matches, sequence_unique = memorization_rate(
        generated_sequence, simple_tokens
    )
    histogram_rate, histogram_matches, histogram_unique = memorization_rate(
        generated_histogram, histogram_tokens
    )
    token_generated_unique = len(torch.unique(generated_token, dim=0))
    sequence_generated_unique = len(torch.unique(generated_sequence, dim=0))
    histogram_generated_unique = len(torch.unique(generated_histogram, dim=0))

    summary = (
        f"Task 2 checkpoint: {checkpoint}\n"
        f"Latent: 3x4x4 signed binary -> sequence length 16, vocabulary 0-7\n"
        f"Generated sequences per model: {args.generate} | Sampling temperature: {args.temperature}\n"
        f"Sequence-to-Token: training images {len(simple_tokens)}, unique training sequences {token_unique}, "
        f"unique generated {token_generated_unique}, matches {token_matches}/{args.generate}, "
        f"rate {token_rate:.2%}, target >80%: {token_rate > 0.8}\n"
        f"Sequence-to-Sequence: training images {len(simple_tokens)}, unique training sequences {sequence_unique}, "
        f"unique generated {sequence_generated_unique}, matches {sequence_matches}/{args.generate}, "
        f"rate {sequence_rate:.2%}, target >80%: {sequence_rate > 0.8}\n"
        f"Sequence-to-Histogram: training images {len(histogram_tokens)}, unique training sequences {histogram_unique}, "
        f"unique generated {histogram_generated_unique}, matches {histogram_matches}/{args.generate}, "
        f"rate {histogram_rate:.2%}, target >99%: {histogram_rate > 0.99}\n"
    )
    print("\n" + summary, flush=True)

    run_dir = ROOT / "outputs" / "task3" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save(token_model.state_dict(), run_dir / "sequence_to_token.pt")
    torch.save(sequence_model.state_dict(), run_dir / "sequence_to_sequence.pt")
    torch.save(histogram_model.state_dict(), run_dir / "sequence_to_histogram.pt")
    torch.save(
        {
            "simple_training_tokens": simple_tokens,
            "histogram_training_tokens": histogram_tokens,
            "sequence_to_token": generated_token,
            "sequence_to_sequence": generated_sequence,
            "sequence_to_histogram": generated_histogram,
        },
        run_dir / "tokens_and_generations.pt",
    )
    (run_dir / "results.txt").write_text(summary, encoding="utf-8")
    with (run_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "epoch", "sequence_to_token_loss", "sequence_to_sequence_loss",
            "sequence_to_histogram_loss",
        ])
        for epoch in range(max(len(token_history), len(sequence_history), len(histogram_history))):
            writer.writerow([
                epoch + 1,
                token_history[epoch] if epoch < len(token_history) else "",
                sequence_history[epoch] if epoch < len(sequence_history) else "",
                histogram_history[epoch] if epoch < len(histogram_history) else "",
            ])

    save_generated_images(autoencoder, generated_token, run_dir / "sequence_to_token.png", device)
    save_generated_images(autoencoder, generated_sequence, run_dir / "sequence_to_sequence.png", device)
    save_generated_images(autoencoder, generated_histogram, run_dir / "sequence_to_histogram.png", device)
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
