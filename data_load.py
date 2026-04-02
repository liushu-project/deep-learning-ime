import hyperparams as hp
import numpy as np
import re
import json
import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader


def load_vocab_json():
    """Load vocabulary mappings from JSON file."""
    if hp.isqwerty:
        vocab_file = "data/vocab.qwerty.json"
    else:
        vocab_file = "data/vocab.nine.json"

    try:
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect a tuple/list of four elements: (pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi)
        if len(data) != 4:
            raise ValueError("Vocabulary file must contain exactly four mappings.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Vocabulary file {vocab_file} not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in {vocab_file}.")


def load_train_data():
    """Load and vectorize training data."""
    pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab_json()

    print("pnyn vocabulary size is", len(pnyn2idx))
    print("hanzi vocabulary size is", len(hanzi2idx))

    xs, ys = [], []
    # Debug output file (optional, can be removed)
    with open("t", "w", encoding="utf-8") as fout:
        with open("data/zh.tsv", "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    _, pnyn_sent, hanzi_sent = line.split("\t")
                except ValueError:
                    continue  # skip malformed lines

                # Split sentences by punctuation markers (。，！？)
                # Insert separator after each punctuation, then split.
                pnyn_clauses = re.sub("(?<=([。，！？]))", r"|", pnyn_sent).split("|")
                hanzi_clauses = re.sub("(?<=([。，！？]))", r"|", hanzi_sent).split("|")
                # Remove empty strings that may result from trailing separators
                pnyn_clauses = [c for c in pnyn_clauses if c]
                hanzi_clauses = [c for c in hanzi_clauses if c]

                # Also include the full sentence as a clause
                all_pnyn = pnyn_clauses + [pnyn_sent]
                all_hanzi = hanzi_clauses + [hanzi_sent]

                fout.write(pnyn_sent + "===" + "|".join(pnyn_clauses) + "\n")

                for p_clause, h_clause in zip(all_pnyn, all_hanzi):
                    # Ensure lengths match (should, due to aligned punctuation)
                    if len(p_clause) != len(h_clause):
                        print(
                            f"Warning: length mismatch in clause: {p_clause} vs {h_clause}"
                        )
                        continue
                    if hp.minlen < len(p_clause) <= hp.maxlen:
                        x = [pnyn2idx.get(p, 1) for p in p_clause]  # 1: OOV
                        y = [hanzi2idx.get(h, 1) for h in h_clause]  # 1: OOV
                        # Store as bytes using tobytes() (formerly tostring)
                        xs.append(np.array(x, np.int32).tobytes())
                        ys.append(np.array(y, np.int32).tobytes())
    return xs, ys


def load_test_data():
    """Load test data from CSV file."""
    input_file = "eval/input.csv"
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            # Use csv.reader to handle quoted commas
            reader = csv.reader(f)
            header = next(reader)  # skip header row
            lines = list(reader)
    except FileNotFoundError:
        raise IOError(
            f"Test file {input_file} not found. Write sentences in `eval/input.csv`."
        )
    except StopIteration:
        raise IOError(f"Test file {input_file} is empty or has no header.")

    pnyn2idx, _, hanzi2idx, _ = load_vocab_json()

    nums, xs, ys = [], [], []  # ys: ground truth (list of string)
    for row in lines:
        if len(row) < 3:
            continue  # skip incomplete lines
        num, pnyn_sent, y = row[0], row[1], row[2]

        nums.append(num)
        # Vectorize pnyn sequence
        x = [pnyn2idx.get(p, 1) for p in pnyn_sent]  # 1: OOV
        # Pad or truncate to hp.maxlen
        if len(x) > hp.maxlen:
            x = x[: hp.maxlen]
        else:
            x += [0] * (hp.maxlen - len(x))
        xs.append(x)
        ys.append(y)

    X = np.array(xs, np.int32)
    return nums, X, ys


def load_test_string(pnyn2idx, test_string):
    """Vectorize a single user input string."""
    x = [pnyn2idx.get(p, 1) for p in test_string]  # 1: OOV
    # Pad or truncate to hp.maxlen
    if len(x) > hp.maxlen:
        x = x[: hp.maxlen]
    else:
        x += [0] * (hp.maxlen - len(x))
    xs = [x]
    X = np.array(xs, np.int32)
    return X


class SequenceDataset(Dataset):
    def __init__(self, X_bytes, Y_bytes):
        self.X_bytes = X_bytes
        self.Y_bytes = Y_bytes

    def __len__(self):
        return len(self.X_bytes)

    def __getitem__(self, idx):
        x = torch.from_numpy(
            np.frombuffer(self.X_bytes[idx], dtype=np.int32).copy()
        ).int()
        y = torch.from_numpy(
            np.frombuffer(self.Y_bytes[idx], dtype=np.int32).copy()
        ).int()
        return x, y


def collate_batch(batch):
    x_tensors, y_tensors = zip(*batch)
    max_len = max(t.size(0) for t in x_tensors)
    padded_x = [
        torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=0) for t in x_tensors
    ]
    padded_y = [
        torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=0) for t in y_tensors
    ]
    return torch.stack(padded_x), torch.stack(padded_y)


def get_batch():
    X_bytes, Y_bytes = load_train_data()
    dataset = SequenceDataset(X_bytes, Y_bytes)
    num_batch = len(dataset) // hp.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        drop_last=True,  # discard last incomplete batch
        num_workers=0,
    )
    return dataloader, num_batch
