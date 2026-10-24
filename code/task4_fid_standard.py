"""用 pytorch-fid 的官方兼容 Inception 特征评估，输入为 [N,3,32,32] .pt。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

from task4_models import ROOT, device_name


@torch.no_grad()
def features(model, loader, device):
    batches = []
    for images, *_ in loader:
        images = images.to(device)
        if images.dtype == torch.uint8:
            images = images.float() / 255
        else:
            images = images.float()
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("浮点图片必须为有限的 [0,1] 像素")
        # 模型自身负责 resize 到299和映射到[-1,1]，不做ImageNet标准化。
        batches.append(model(images)[0].flatten(1).cpu().numpy())
    return np.concatenate(batches).astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("generated")
    parser.add_argument("--images", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    path = Path(args.generated).expanduser().resolve()
    images = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError("需要 [N,3,32,32] 图片张量")
    if not 2 <= args.images <= min(len(images), 10000) or args.batch_size < 1:
        parser.error("图片数必须介于2和min(已有生成图片数,10000)之间；batch-size须为正")
    device = args.device or device_name()
    torch.set_num_threads(4)
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    real = datasets.CIFAR10(ROOT / "data", train=False, download=True, transform=transforms.ToTensor())
    real = torch.utils.data.Subset(real, range(args.images))
    print(f"pytorch-fid 0.3.0 | 2048 features | Device: {device} | Images: {args.images}", flush=True)
    fake_features = features(model, DataLoader(TensorDataset(images[:args.images]), batch_size=args.batch_size), device)
    print("Generated features complete", flush=True)
    real_features = features(model, DataLoader(real, batch_size=args.batch_size), device)
    print("Real features complete; calculating FID", flush=True)
    score = calculate_frechet_distance(
        real_features.mean(0), np.cov(real_features, rowvar=False),
        fake_features.mean(0), np.cov(fake_features, rowvar=False),
    )
    result = {
        "implementation": "pytorch-fid 0.3.0, FID Inception, 2048 features",
        "reference": "CIFAR-10 test, first N images", "images": args.images,
        "generated": str(path), "fid": float(score), "fid_below_40": bool(score < 40),
    }
    (path.parent / "fid_pytorch.json").write_text(json.dumps(result, indent=2))
    text = f"pytorch-fid ({args.images} generated vs {args.images} CIFAR-10 test): {score:.4f}\nFID < 40: {score < 40}\n"
    (path.parent / "fid_pytorch.txt").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
