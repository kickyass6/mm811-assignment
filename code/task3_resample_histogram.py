"""Resample a trained Task 3 Sequence-to-Histogram model at a new temperature."""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from task2_optimized import OptimizedBinaryAutoencoder
from task3_autoregressive import (
    HISTOGRAM_IMAGES,
    SEED,
    SequenceToHistogram,
    generate_sequence_to_histogram,
    memorization_rate,
    tokens_to_latent,
)


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Task 3 run directory containing sequence_to_histogram.pt and saved tokens",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--generate", type=int, default=1000)
    return parser.parse_args()


def checkpoint_from_results(run):
    text = (run / "results.txt").read_text(encoding="utf-8")
    match = re.search(r"^Task 2 checkpoint: (.+)$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Task 2 checkpoint is missing from {run / 'results.txt'}")
    return Path(match.group(1))


def label_lookup(training_tokens):
    mnist = datasets.MNIST(
        ROOT / "data", train=True, download=True, transform=transforms.ToTensor()
    )
    permutation = torch.randperm(
        len(mnist), generator=torch.Generator().manual_seed(SEED)
    )
    available = torch.cat([permutation[:256], permutation[512:]])
    labels = [mnist[int(index)][1] for index in available[:HISTOGRAM_IMAGES]]

    candidates = defaultdict(list)
    for sequence, label in zip(training_tokens.tolist(), labels):
        candidates[tuple(sequence)].append(label)
    return {
        sequence: Counter(sequence_labels).most_common(1)[0][0]
        for sequence, sequence_labels in candidates.items()
    }


def main():
    args = parse_args()
    run = args.run.expanduser().resolve()
    if args.temperature <= 0 or args.generate < 1:
        raise ValueError("temperature and generate must be positive")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    saved = torch.load(
        run / "tokens_and_generations.pt", map_location="cpu", weights_only=True
    )
    training_tokens = saved["histogram_training_tokens"]

    model = SequenceToHistogram(hidden_size=256).to(device)
    model.load_state_dict(
        torch.load(
            run / "sequence_to_histogram.pt", map_location="cpu", weights_only=True
        )
    )
    torch.manual_seed(SEED + 3)
    generated = generate_sequence_to_histogram(
        model, args.generate, args.temperature, device
    )

    rate, matches, unique_training = memorization_rate(generated, training_tokens)
    unique_generated = len(torch.unique(generated, dim=0))
    lookup = label_lookup(training_tokens)
    generated_labels = [lookup.get(tuple(sequence), -1) for sequence in generated.tolist()]
    distribution = Counter(generated_labels)

    tag = str(args.temperature).replace(".", "p")
    output = run / f"resample_temperature_{tag}"
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = checkpoint_from_results(run)
    autoencoder = OptimizedBinaryAutoencoder().to(device)
    autoencoder.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    autoencoder.eval()
    with torch.no_grad():
        decoded = autoencoder.decoder(tokens_to_latent(generated[:64]).to(device)).cpu()
    save_image(decoded, output / "sequence_to_histogram.png", nrow=8)

    counts = [distribution.get(digit, 0) for digit in range(10)]
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    axis.bar(range(10), counts, color="#194a8a")
    axis.set_xticks(range(10))
    axis.set_xlabel("MNIST digit associated with memorized training sequence")
    axis.set_ylabel("Generated count")
    axis.set_title(f"Sequence-to-Histogram, temperature={args.temperature:g}")
    figure.tight_layout()
    figure.savefig(output / "class_distribution.png", dpi=180)
    plt.close(figure)

    summary = (
        f"Source run: {run}\n"
        f"Temperature: {args.temperature:g} | Generated: {args.generate}\n"
        f"Unique training sequences: {unique_training}\n"
        f"Unique generated sequences: {unique_generated}\n"
        f"Exact training matches: {matches}/{args.generate} ({rate:.2%})\n"
        f"Target >99%: {rate > 0.99}\n"
        f"Matched-sequence digit counts 0-9: {counts}\n"
        f"Unmatched sequences: {distribution.get(-1, 0)}\n"
    )
    print(summary, flush=True)
    torch.save(
        {"generated": generated, "labels": torch.tensor(generated_labels)},
        output / "samples.pt",
    )
    (output / "results.txt").write_text(summary, encoding="utf-8")
    print(f"Results saved to: {output}", flush=True)


if __name__ == "__main__":
    main()
