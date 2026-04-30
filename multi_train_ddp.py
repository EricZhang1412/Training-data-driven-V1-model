import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import load_sparse_torch as load_sparse
from models_torch import BillehColumnTorch


class RateReadout(nn.Module):
    def __init__(self, n_neurons, n_classes):
        super().__init__()
        self.fc = nn.Linear(n_neurons, n_classes)

    def forward(self, spikes):
        rates = spikes.float().mean(dim=1)
        return self.fc(rates)


class V1Classifier(nn.Module):
    def __init__(self, v1, readout):
        super().__init__()
        self.v1 = v1
        self.readout = readout

    def forward(self, x):
        spikes, voltages, state = self.v1(x)
        logits = self.readout(spikes)
        return logits, spikes, voltages


def setup_distributed():
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

    return distributed, rank, local_rank, world_size, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank):
    return rank == 0


def unwrap_ddp(module):
    return module.module if isinstance(module, DDP) else module


def reduce_mean(value, device):
    tensor = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor.item()


def make_batch(args, device):
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


def save_checkpoint(path, v1, readout, optimizer, args, epoch, rank):
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


@torch.no_grad()
def evaluate(args, v1, readout, train_module, device):
    total_loss = 0.0
    total_acc = 0.0
    total_rate = 0.0

    if args.train_v1:
        train_module.eval()
    else:
        v1.eval()
        readout.eval()

    for _ in range(args.val_steps):
        x, y = make_batch(args, device)
        if args.train_v1:
            logits, spikes, voltages = train_module(x)
        else:
            spikes, voltages, state = v1(x)
            logits = readout(spikes)

        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean()
        rate_hz = spikes.float().mean() * 1000.0

        total_loss += loss.item()
        total_acc += acc.item()
        total_rate += rate_hz.item()

    if args.train_v1:
        train_module.train()
    else:
        readout.train()
        v1.eval()

    return (
        total_loss / args.val_steps,
        total_acc / args.val_steps,
        total_rate / args.val_steps,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", default="./torch_results_ddp")
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--n-input", type=int, default=17400)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size.")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--val-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--train-v1", action="store_true")
    parser.add_argument("--full-core", action="store_true")
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--toy-task", choices=["left_right", "random"], default="left_right")
    parser.add_argument("--base-p", type=float, default=0.01)
    parser.add_argument("--low-p", type=float, default=0.005)
    parser.add_argument("--high-p", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size, device = setup_distributed()

    try:
        torch.manual_seed(args.seed + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + rank)
        torch.sparse.check_sparse_tensor_invariants.disable()

        if is_rank0(rank):
            print("distributed:", distributed)
            print("world_size:", world_size)
            print("per_gpu_batch_size:", args.batch_size)
            print("global_batch_size:", args.batch_size * world_size)
            print("device:", device)

        loaded = load_sparse.load_billeh_torch(
            n_input=args.n_input,
            n_neurons=args.neurons,
            core_only=not args.full_core,
            data_dir=args.data_dir,
            seed=args.seed,
            connected_selection=args.full_core,
            localized_readout=False,
            neurons_per_output=4,
            device=device,
        )

        bkg_weights = loaded["bkg_weights"]
        if isinstance(bkg_weights, torch.Tensor):
            bkg_weights = bkg_weights.detach().cpu().numpy()

        v1 = BillehColumnTorch(
            loaded["network"],
            loaded["input_population"],
            bkg_weights,
            device=device,
        ).to(device)
        readout = RateReadout(args.neurons, args.n_classes).to(device)

        ddp_kwargs = {}
        if distributed and device.type == "cuda":
            ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank}

        if args.train_v1:
            if is_rank0(rank):
                print("> training V1 + readout with DDP")
            train_module = V1Classifier(v1, readout).to(device)
            if distributed:
                train_module = DDP(
                    train_module,
                    find_unused_parameters=args.find_unused_parameters,
                    **ddp_kwargs,
                )
            params = train_module.parameters()
        else:
            if is_rank0(rank):
                print("> training readout only with DDP")
            v1.eval()
            for p in v1.parameters():
                p.requires_grad_(False)
            if distributed:
                readout = DDP(readout, **ddp_kwargs)
            train_module = None
            params = readout.parameters()

        optimizer = torch.optim.AdamW(params, lr=args.lr)

        for epoch in range(args.n_epochs):
            if args.train_v1:
                train_module.train()
            else:
                v1.eval()
                readout.train()

            total_loss = 0.0
            total_acc = 0.0
            total_rate = 0.0

            for step in range(args.steps_per_epoch):
                x, y = make_batch(args, device)

                optimizer.zero_grad(set_to_none=True)

                if args.train_v1:
                    logits, spikes, voltages = train_module(x)
                else:
                    with torch.no_grad():
                        spikes, voltages, state = v1(x)
                    logits = readout(spikes)

                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

                pred = logits.argmax(dim=-1)
                acc = (pred == y).float().mean().item()
                rate_hz = spikes.float().mean().item() * 1000.0

                total_loss += loss.item()
                total_acc += acc
                total_rate += rate_hz

                if is_rank0(rank):
                    print(
                        f"epoch={epoch + 1} "
                        f"step={step + 1}/{args.steps_per_epoch} "
                        f"loss={loss.item():.4f} "
                        f"acc={acc:.3f} "
                        f"rate_hz={rate_hz:.3f}"
                    )

            train_loss = reduce_mean(total_loss / args.steps_per_epoch, device)
            train_acc = reduce_mean(total_acc / args.steps_per_epoch, device)
            train_rate = reduce_mean(total_rate / args.steps_per_epoch, device)

            val_loss, val_acc, val_rate = evaluate(args, v1, readout, train_module, device)
            val_loss = reduce_mean(val_loss, device)
            val_acc = reduce_mean(val_acc, device)
            val_rate = reduce_mean(val_rate, device)

            if is_rank0(rank):
                print(
                    f"> epoch {epoch + 1}: "
                    f"train_loss={train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, "
                    f"train_rate_hz={train_rate:.3f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_acc={val_acc:.3f}, "
                    f"val_rate_hz={val_rate:.3f}"
                )

            if distributed:
                dist.barrier()

            ckpt_path = os.path.join(args.results_dir, f"epoch_{epoch + 1}.pt")
            save_checkpoint(
                ckpt_path,
                v1=unwrap_ddp(v1),
                readout=unwrap_ddp(readout),
                optimizer=optimizer,
                args=args,
                epoch=epoch + 1,
                rank=rank,
            )
            if is_rank0(rank):
                print(f"> saved {ckpt_path}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
