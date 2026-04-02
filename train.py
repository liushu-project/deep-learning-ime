import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

import hyperparams as hp
from data_load import get_batch, load_vocab_json
from model import Graph


def train():
    # Load vocabulary for sizes
    pnyn2idx, _, hanzi2idx, _ = load_vocab_json()
    pnyn_vocab_size = len(pnyn2idx)
    hanzi_vocab_size = len(hanzi2idx)

    # Data loader
    dataloader, num_batch = get_batch()
    print(f"Number of training batches: {num_batch}")

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model
    model = Graph(pnyn_vocab_size, hanzi_vocab_size)
    model.to(device)
    model.train()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=hp.logdir)

    # Global step counter
    global_step = 0

    # Training loop
    for epoch in range(1, hp.num_epochs + 1):
        epoch_loss = 0.0
        epoch_acc = 0.0

        pbar = tqdm(
            enumerate(dataloader), total=num_batch, desc=f"Epoch {epoch}", unit="b"
        )
        for step, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss, acc, _ = model(x, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            global_step += 1

            pbar.set_postfix(loss=loss.item(), acc=acc.item())
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/acc", acc.item(), global_step)

        avg_loss = epoch_loss / num_batch
        avg_acc = epoch_acc / num_batch
        print(
            f"Epoch {epoch} completed: avg_loss = {avg_loss:.4f}, avg_acc = {avg_acc:.4f}"
        )

        checkpoint_path = os.path.join(
            hp.logdir, f"model_epoch_{epoch:02d}_gs_{global_step}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "loss": avg_loss,
                "acc": avg_acc,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    train()
