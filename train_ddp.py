import argparse
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import load_sparse_torch as load_sparse
from models_torch import BillehColumnTorch
from utils.datasets import build_mnist_dataloaders, mnist_images_to_model_input, set_mnist_epoch


class RateReadout(nn.Module):
    def __init__(self, n_neurons: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(n_neurons, n_classes)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        rates = spikes.float().mean(dim=1)
        return self.fc(rates)


class V1Classifier(nn.Module):
    def __init__(self, v1: nn.Module, readout: nn.Module):
        super().__init__()
        self.v1 = v1
        self.readout = readout

    def forward(self, x: torch.Tensor):
        spikes, voltages, _ = self.v1(x)
        logits = self.readout(spikes)
        return logits, spikes, voltages


@dataclass
class Runtime:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def setup_distributed() -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed:
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return Runtime(distributed, rank, local_rank, world_size, device)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def reduce_scalar(value: float, device: torch.device) -> float:
    t = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def make_toy_batch(args, device: torch.device):
    if args.toy_task == "random":
        x = (torch.rand(args.batch_size, args.seq_len, args.n_input, device=device) < args.base_p).float()
        y = torch.randint(0, args.n_classes, (args.batch_size,), device=device)
        return x, y

    if args.n_classes != 2:
        raise ValueError("left_right toy task only supports --n-classes 2")

    y = torch.randint(0, 2, (args.batch_size,), device=device)
    half = args.n_input // 2
    x = torch.zeros(args.batch_size, args.seq_len, args.n_input, device=device)

    p_left = torch.where(
        y[:, None, None] == 0,
        torch.tensor(args.high_p, device=device),
        torch.tensor(args.low_p, device=device),
    )
    p_right = torch.where(
        y[:, None, None] == 0,
        torch.tensor(args.low_p, device=device),
        torch.tensor(args.high_p, device=device),
    )

    x[:, :, :half] = (torch.rand(args.batch_size, args.seq_len, half, device=device) < p_left).float()
    x[:, :, half:] = (
        torch.rand(args.batch_size, args.seq_len, args.n_input - half, device=device) < p_right
    ).float()
    return x, y


def save_checkpoint(path: str, v1: nn.Module, readout: nn.Module, optimizer, args, epoch: int, rank: int):
    if not is_rank0(rank):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "v1": v1.state_dict(),
            "readout": readout.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def forward_step(args, v1, readout, train_module, x: torch.Tensor):
    if args.train_v1:
        logits, spikes, _ = train_module(x)
    else:
        with torch.no_grad():
            spikes, _, _ = v1(x)
        logits = readout(spikes)
    return logits, spikes


@torch.no_grad()
def evaluate_toy(args, v1, readout, train_module, runtime: Runtime):
    total_loss_sum = 0.0
    total_correct = 0.0
    total_rate_sum = 0.0
    total_samples = 0

    if args.train_v1:
        train_module.eval()
    else:
        v1.eval()
        readout.eval()

    for _ in range(args.val_steps):
        x, y = make_toy_batch(args, runtime.device)
        logits, spikes = forward_step(args, v1, readout, train_module, x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        bs = y.numel()

        total_loss_sum += loss.item() * bs
        total_correct += (pred == y).sum().item()
        total_rate_sum += spikes.float().mean().item() * 1000.0 * bs
        total_samples += bs

    if args.train_v1:
        train_module.train()
    else:
        readout.train()
        v1.eval()

    global_loss_sum = reduce_scalar(total_loss_sum, runtime.device)
    global_correct = reduce_scalar(total_correct, runtime.device)
    global_rate_sum = reduce_scalar(total_rate_sum, runtime.device)
    global_samples = reduce_scalar(total_samples, runtime.device)

    return (
        global_loss_sum / max(global_samples, 1.0),
        global_correct / max(global_samples, 1.0),
        global_rate_sum / max(global_samples, 1.0),
    )


@torch.no_grad()
def evaluate_mnist(args, v1, readout, train_module, runtime: Runtime, val_loader):
    total_loss_sum = 0.0
    total_correct = 0.0
    total_rate_sum = 0.0
    total_samples = 0

    if args.train_v1:
        train_module.eval()
    else:
        v1.eval()
        readout.eval()

    for step, (images, y) in enumerate(val_loader):
        if args.max_val_steps > 0 and step >= args.max_val_steps:
            break

        images = images.to(runtime.device, non_blocking=True)
        y = y.to(runtime.device, non_blocking=True)
        x = mnist_images_to_model_input(
            images,
            seq_len=args.seq_len,
            n_input=args.n_input,
            encoding=args.mnist_encoding,
            gain=args.mnist_gain,
        )

        logits, spikes = forward_step(args, v1, readout, train_module, x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        bs = y.numel()

        total_loss_sum += loss.item() * bs
        total_correct += (pred == y).sum().item()
        total_rate_sum += spikes.float().mean().item() * 1000.0 * bs
        total_samples += bs

    if args.train_v1:
        train_module.train()
    else:
        readout.train()
        v1.eval()

    global_loss_sum = reduce_scalar(total_loss_sum, runtime.device)
    global_correct = reduce_scalar(total_correct, runtime.device)
    global_rate_sum = reduce_scalar(total_rate_sum, runtime.device)
    global_samples = reduce_scalar(total_samples, runtime.device)

    return (
        global_loss_sum / max(global_samples, 1.0),
        global_correct / max(global_samples, 1.0),
        global_rate_sum / max(global_samples, 1.0),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic torchrun-friendly trainer for V1 model (toy or MNIST)."
    )

    # Core
    parser.add_argument("--data-dir", required=True, help="Path to GLIF_network directory.")
    parser.add_argument("--results-dir", default="./torch_results_ddp")
    parser.add_argument("--dataset", choices=["toy", "mnist"], default="mnist")
    parser.add_argument("--seed", type=int, default=3000)

    # Model
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--n-input", type=int, default=17400)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--train-v1", action="store_true")
    parser.add_argument("--full-core", action="store_true")
    parser.add_argument("--find-unused-parameters", action="store_true")

    # Optimization
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size.")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 means full dataloader for MNIST.")
    parser.add_argument("--val-steps", type=int, default=20, help="Toy validation steps.")
    parser.add_argument("--max-val-steps", type=int, default=0, help="0 means full val dataloader for MNIST.")
    parser.add_argument("--lr", type=float, default=1e-3)

    # Toy task options
    parser.add_argument("--toy-task", choices=["left_right", "random"], default="left_right")
    parser.add_argument("--base-p", type=float, default=0.01)
    parser.add_argument("--low-p", type=float, default=0.005)
    parser.add_argument("--high-p", type=float, default=0.03)

    # MNIST options
    parser.add_argument("--mnist-data-dir", default="./data/mnist")
    parser.add_argument("--mnist-encoding", choices=["poisson", "repeat"], default="poisson")
    parser.add_argument("--mnist-gain", type=float, default=1.0)
    parser.add_argument("--mnist-normalize", action="store_true")
    parser.add_argument("--download-mnist", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()
    runtime = setup_distributed()

    try:
        torch.manual_seed(args.seed + runtime.rank)
        if runtime.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + runtime.rank)
        torch.sparse.check_sparse_tensor_invariants.disable()

        if is_rank0(runtime.rank):
            print("distributed:", runtime.distributed)
            print("world_size:", runtime.world_size)
            print("per_gpu_batch_size:", args.batch_size)
            print("global_batch_size:", args.batch_size * runtime.world_size)
            print("device:", runtime.device)
            print("dataset:", args.dataset)

        loaded = load_sparse.load_billeh_torch(
            n_input=args.n_input,
            n_neurons=args.neurons,
            core_only=not args.full_core,
            data_dir=args.data_dir,
            seed=args.seed,
            connected_selection=args.full_core,
            localized_readout=False,
            neurons_per_output=4,
            device=runtime.device,
        )

        bkg_weights = loaded["bkg_weights"]
        if isinstance(bkg_weights, torch.Tensor):
            bkg_weights = bkg_weights.detach().cpu().numpy()

        v1 = BillehColumnTorch(
            loaded["network"],
            loaded["input_population"],
            bkg_weights,
            device=runtime.device,
        ).to(runtime.device)
        readout = RateReadout(args.neurons, args.n_classes).to(runtime.device)

        ddp_kwargs = {}
        if runtime.distributed and runtime.device.type == "cuda":
            ddp_kwargs = {"device_ids": [runtime.local_rank], "output_device": runtime.local_rank}

        if args.train_v1:
            if is_rank0(runtime.rank):
                print("> training V1 + readout with DDP")
            train_module = V1Classifier(v1, readout).to(runtime.device)
            if runtime.distributed:
                train_module = DDP(
                    train_module,
                    find_unused_parameters=args.find_unused_parameters,
                    **ddp_kwargs,
                )
            params = train_module.parameters()
        else:
            if is_rank0(runtime.rank):
                print("> training readout only with DDP")
            v1.eval()
            for p in v1.parameters():
                p.requires_grad_(False)
            if runtime.distributed:
                readout = DDP(readout, **ddp_kwargs)
            train_module = None
            params = readout.parameters()

        optimizer = torch.optim.AdamW(params, lr=args.lr)

        loaders = None
        if args.dataset == "mnist":
            loaders = build_mnist_dataloaders(
                data_dir=args.mnist_data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                distributed=runtime.distributed,
                rank=runtime.rank,
                world_size=runtime.world_size,
                seed=args.seed,
                pin_memory=runtime.device.type == "cuda",
                download=args.download_mnist,
                normalize=args.mnist_normalize,
            )

        for epoch in range(args.n_epochs):
            if args.train_v1:
                train_module.train()
            else:
                v1.eval()
                readout.train()

            total_loss_sum = 0.0
            total_correct = 0.0
            total_rate_sum = 0.0
            total_samples = 0

            if args.dataset == "toy":
                n_steps = args.steps_per_epoch if args.steps_per_epoch > 0 else 50
                train_iter = range(n_steps)
            else:
                set_mnist_epoch(loaders, epoch)
                train_iter = enumerate(loaders.train_loader)
                n_steps = args.steps_per_epoch if args.steps_per_epoch > 0 else len(loaders.train_loader)

            for step_data in train_iter:
                if args.dataset == "toy":
                    x, y = make_toy_batch(args, runtime.device)
                    step = step_data
                else:
                    step, (images, y) = step_data
                    if step >= n_steps:
                        break
                    images = images.to(runtime.device, non_blocking=True)
                    y = y.to(runtime.device, non_blocking=True)
                    x = mnist_images_to_model_input(
                        images,
                        seq_len=args.seq_len,
                        n_input=args.n_input,
                        encoding=args.mnist_encoding,
                        gain=args.mnist_gain,
                    )

                optimizer.zero_grad(set_to_none=True)
                logits, spikes = forward_step(args, v1, readout, train_module, x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

                pred = logits.argmax(dim=-1)
                bs = y.numel()
                correct = (pred == y).sum().item()
                rate_hz = spikes.float().mean().item() * 1000.0

                total_loss_sum += loss.item() * bs
                total_correct += correct
                total_rate_sum += rate_hz * bs
                total_samples += bs

                if is_rank0(runtime.rank):
                    print(
                        f"epoch={epoch + 1} "
                        f"step={step + 1}/{n_steps} "
                        f"loss={loss.item():.4f} "
                        f"acc={(correct / bs):.3f} "
                        f"rate_hz={rate_hz:.3f}"
                    )

            global_loss_sum = reduce_scalar(total_loss_sum, runtime.device)
            global_correct = reduce_scalar(total_correct, runtime.device)
            global_rate_sum = reduce_scalar(total_rate_sum, runtime.device)
            global_samples = reduce_scalar(total_samples, runtime.device)

            train_loss = global_loss_sum / max(global_samples, 1.0)
            train_acc = global_correct / max(global_samples, 1.0)
            train_rate = global_rate_sum / max(global_samples, 1.0)

            if args.dataset == "toy":
                val_loss, val_acc, val_rate = evaluate_toy(args, v1, readout, train_module, runtime)
            else:
                val_loss, val_acc, val_rate = evaluate_mnist(
                    args, v1, readout, train_module, runtime, loaders.val_loader
                )

            if is_rank0(runtime.rank):
                print(
                    f"> epoch {epoch + 1}: "
                    f"train_loss={train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, "
                    f"train_rate_hz={train_rate:.3f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_acc={val_acc:.3f}, "
                    f"val_rate_hz={val_rate:.3f}"
                )

            if runtime.distributed:
                dist.barrier()

            ckpt_path = os.path.join(args.results_dir, f"epoch_{epoch + 1}.pt")
            save_checkpoint(
                ckpt_path,
                v1=unwrap_ddp(v1),
                readout=unwrap_ddp(readout),
                optimizer=optimizer,
                args=args,
                epoch=epoch + 1,
                rank=runtime.rank,
            )
            if is_rank0(runtime.rank):
                print(f"> saved {ckpt_path}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
