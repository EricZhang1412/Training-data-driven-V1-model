import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import load_sparse_torch as load_sparse
from models_torch import BillehColumnTorch


class RateReadout(nn.Module):
    def __init__(self, n_neurons, n_classes):
        super().__init__()
        self.fc = nn.Linear(n_neurons, n_classes)

    def forward(self, spikes):
        rates = spikes.float().mean(dim=1)
        return self.fc(rates)


# def make_random_batch(batch_size, seq_len, n_input, n_classes, device):
#     x = (torch.rand(batch_size, seq_len, n_input, device=device) < 0.01).float()
#     y = torch.randint(0, n_classes, (batch_size,), device=device)
#     return x, y

def make_random_batch(batch_size, seq_len, n_input, n_classes, device):
    y = torch.randint(0, 2, (batch_size,), device=device)

    half = n_input // 2
    x = torch.zeros(batch_size, seq_len, n_input, device=device)

    low_p = 0.005
    high_p = 0.03

    p_left = torch.where(y[:, None, None] == 0, high_p, low_p)
    p_right = torch.where(y[:, None, None] == 0, low_p, high_p)

    x[:, :, :half] = (torch.rand(batch_size, seq_len, half, device=device) < p_left).float()
    x[:, :, half:] = (torch.rand(batch_size, seq_len, n_input - half, device=device) < p_right).float()

    return x, y




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", default="./torch_results")
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--n-input", type=int, default=17400)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--train-v1", action="store_true")
    parser.add_argument("--full-core", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.sparse.check_sparse_tensor_invariants.disable()

    os.makedirs(args.results_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    if args.train_v1:
        v1.train()
        print("> training V1 + readout")
    else:
        v1.eval()
        for p in v1.parameters():
            p.requires_grad_(False)
        print("> training readout only")

    readout = RateReadout(args.neurons, args.n_classes).to(device)

    params = list(readout.parameters())
    if args.train_v1:
        params += [p for p in v1.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=args.lr)

    for epoch in range(args.n_epochs):
        total_loss = 0.0
        total_acc = 0.0
        total_rate = 0.0

        for step in range(args.steps_per_epoch):
            x, y = make_random_batch(
                args.batch_size,
                args.seq_len,
                args.n_input,
                args.n_classes,
                device,
            )

            optimizer.zero_grad(set_to_none=True)

            if args.train_v1:
                spikes, voltages, state = v1(x)
            else:
                with torch.no_grad():
                    spikes, voltages, state = v1(x)

            logits = readout(spikes)
            loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean().item()
            spike_rate_hz = spikes.float().mean().item() * 1000

            total_loss += loss.item()
            total_acc += acc
            total_rate += spike_rate_hz

            print(
                f"epoch={epoch + 1} "
                f"step={step + 1}/{args.steps_per_epoch} "
                f"loss={loss.item():.4f} "
                f"acc={acc:.3f} "
                f"rate_hz={spike_rate_hz:.3f}"
            )

        ckpt_path = os.path.join(args.results_dir, f"epoch_{epoch + 1}.pt")
        torch.save(
            {
                "v1": v1.state_dict(),
                "readout": readout.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )

        print(
            f"> epoch {epoch + 1}: "
            f"loss={total_loss / args.steps_per_epoch:.4f}, "
            f"acc={total_acc / args.steps_per_epoch:.3f}, "
            f"rate_hz={total_rate / args.steps_per_epoch:.3f}"
        )
        print(f"> saved {ckpt_path}")


if __name__ == "__main__":
    main()
