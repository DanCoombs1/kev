"""A small CNN for probing whether a binary label can be read off images."""

import numpy as np
import torch
from torch import nn

from kev.metrics import auc


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = [3, 16, 32, 64, 128]
        for c_in, c_out in zip(channels, channels[1:]):
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.score = nn.Linear(channels[-1], 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.score(self.features(images).mean(dim=(2, 3))).squeeze(1)


def to_float(batch: torch.Tensor) -> torch.Tensor:
    """uint8 NHWC -> float NCHW around zero."""
    return batch.permute(0, 3, 1, 2).float() / 255 - 0.5


def scores(model: nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    images = torch.from_numpy(images)
    with torch.no_grad():
        return torch.cat([model(to_float(images[i: i + 256].to(device))).cpu() for i in range(0, len(images), 256)]).numpy()


def score_auc(model: nn.Module, images: np.ndarray, labels: np.ndarray, device: torch.device) -> float:
    s = scores(model, images, device)
    return auc(s[labels == 1].tolist(), s[labels == 0].tolist())


def train(train_x: np.ndarray, train_y: np.ndarray, device: torch.device, epochs: int = 15,
          report: tuple[np.ndarray, np.ndarray] | None = None, label: str = "") -> nn.Module:
    """Train on uint8 NHWC images; optionally print validation AUC as it goes."""
    torch.manual_seed(0)
    model = Probe().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    images, answers = torch.from_numpy(train_x), torch.from_numpy(train_y)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(images))
        total = 0.0
        for start in range(0, len(order), 128):
            batch = order[start: start + 128]
            x, y = to_float(images[batch].to(device)), answers[batch].to(device)
            flip = torch.rand(len(x), 1, 1, 1, device=device) < 0.5
            x = torch.where(flip, x.flip(3), x)
            loss = loss_fn(model(x), y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        if report is not None and epoch in (1, 5, 10, epochs):
            print(f"   {label:<9} epoch {epoch:>2}: train loss {total / len(order):.3f}   val AUC {score_auc(model, *report, device):.2f}")
    return model


def train_and_score(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
                    device: torch.device, label: str, epochs: int = 15) -> float:
    model = train(train_x, train_y, device, epochs, (val_x, val_y), label)
    return score_auc(model, val_x, val_y, device)
