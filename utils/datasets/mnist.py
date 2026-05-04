from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


@dataclass
class MnistLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    train_sampler: DistributedSampler | None
    val_sampler: DistributedSampler | None


def _build_mnist_transform(normalize: bool = False):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("torchvision is required for MNIST dataset loading.") from exc

    items = [transforms.ToTensor()]
    if normalize:
        items.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(items)


def build_mnist_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 3000,
    pin_memory: bool = True,
    download: bool = True,
    normalize: bool = False,
    drop_last_train: bool = True,
) -> MnistLoaders:
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError("torchvision is required for MNIST dataset loading.") from exc

    transform = _build_mnist_transform(normalize=normalize)
    train_set = datasets.MNIST(root=data_dir, train=True, transform=transform, download=download)
    val_set = datasets.MNIST(root=data_dir, train=False, transform=transform, download=download)

    if distributed:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=drop_last_train,
        )
        val_sampler = DistributedSampler(
            val_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return MnistLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
    )


def set_mnist_epoch(loaders: MnistLoaders, epoch: int) -> None:
    if loaders.train_sampler is not None:
        loaders.train_sampler.set_epoch(epoch)


def mnist_images_to_model_input(
    images: torch.Tensor,
    seq_len: int,
    n_input: int,
    encoding: Literal["poisson", "repeat"] = "poisson",
    gain: float = 1.0,
) -> torch.Tensor:
    """
    Convert MNIST [B,1,28,28] images to model input [B,T,n_input].
    """
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(f"Expected images with shape [B,1,H,W], got {tuple(images.shape)}")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if n_input <= 0:
        raise ValueError("n_input must be > 0")

    flat = images.flatten(start_dim=1)

    if flat.shape[1] != n_input:
        # Keep a simple deterministic projection from 784 pixels to n_input.
        x = flat.unsqueeze(1)
        x = F.interpolate(x, size=n_input, mode="linear", align_corners=False)
        flat = x.squeeze(1)

    flat = flat.clamp(0.0, 1.0)

    if encoding == "repeat":
        return flat.unsqueeze(1).expand(-1, seq_len, -1).contiguous()
    if encoding != "poisson":
        raise ValueError(f"Unsupported encoding: {encoding}")

    probs = (flat * gain).clamp(0.0, 1.0)
    spikes = torch.rand(
        flat.shape[0],
        seq_len,
        n_input,
        device=flat.device,
        dtype=flat.dtype,
    ) < probs.unsqueeze(1)
    return spikes.float()
