"""任务 1：模仿课堂结构的精简 MNIST 分类器。"""

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


TRAIN_SIZE = 1024
VAL_SIZE = 256
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.001


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.feats = nn.Sequential(
            nn.Conv2d(1, 8, 4, 2, 1), nn.ReLU(),       # 28 -> 14
            nn.Conv2d(8, 16, 4, 2, 1), nn.ReLU(),      # 14 -> 7
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),     # 7 -> 3
            nn.Conv2d(32, 64, 3),                       # 3 -> 1
        )
        self.labs = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, images):
        features = self.feats(images).flatten(1)
        return self.labs(features)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(1).cpu()
        correct += (predictions == labels).sum().item()
    return correct / len(loader.dataset)


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    root = Path(__file__).resolve().parent / "data"
    source = datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(root, train=False, download=True, transform=transforms.ToTensor())
    train, val, _ = random_split(
        source, [TRAIN_SIZE, VAL_SIZE, len(source) - TRAIN_SIZE - VAL_SIZE],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, 64)
    test_loader = DataLoader(test, 64)

    model = Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    best_val = -1

    print(f"Device: {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)

        val_acc = accuracy(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_weights = deepcopy(model.state_dict())
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss {loss_sum / len(train):.4f} | Val {val_acc:.2%}")

    model.load_state_dict(best_weights)
    print(f"Best epoch: {best_epoch} | Best validation: {best_val:.2%}")
    print(f"Test accuracy: {accuracy(model, test_loader, device):.2%}")


if __name__ == "__main__":
    main()
