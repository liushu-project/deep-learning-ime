from collections import Counter
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import csv

from config import Config
from utils.vocabulary import Vocabulary
from models.labeling.gru import BiGruTagger

def build_vocab_from_data(data: list[tuple[list[str], list[str]]], min_freq=1) -> tuple[Vocabulary, Vocabulary]:
    """从数据中构建拼音和汉字词汇表"""
    pinyin_counter = Counter()
    han_counter = Counter()
    for pinyin_seq, han_seq in data:
        pinyin_counter.update(pinyin_seq)
        han_counter.update(han_seq)

    pinyin_vocab = Vocabulary()
    han_vocab = Vocabulary()

    # 添加高频词
    for token, freq in pinyin_counter.items():
        if freq >= min_freq:
            pinyin_vocab.add_token(token)
    for token, freq in han_counter.items():
        if freq >= min_freq:
            han_vocab.add_token(token)

    return pinyin_vocab, han_vocab

def read_data(file_path: str) -> tuple[list[list[str]], list[list[str]]]:
    """返回 (pinyin_sequences, han_sequences) 列表"""
    pinyin_seqs = []
    han_seqs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            pinyin_tokens = parts[0].strip().split()
            han_tokens = parts[1].strip().split()
            # 确保长度相等
            if len(pinyin_tokens) != len(han_tokens):
                print(f"警告：长度不一致，跳过该行: {line}")
                continue
            pinyin_seqs.append(pinyin_tokens)
            han_seqs.append(han_tokens)
    return pinyin_seqs, han_seqs

class PinyinHanDataset(Dataset):
    def __init__(self, pinyin_seqs: list[list[str]], han_seqs: list[list[str]],
                 pinyin_vocab: Vocabulary, han_vocab: Vocabulary):
        self.pinyin_seqs = pinyin_seqs
        self.han_seqs = han_seqs
        self.pinyin_vocab = pinyin_vocab
        self.han_vocab = han_vocab

    def __len__(self):
        return len(self.pinyin_seqs)

    def __getitem__(self, index):
        pinyin_tokens = self.pinyin_seqs[index]
        han_tokens = self.han_seqs[index]
        pinyin_ids = self.pinyin_vocab.encode(pinyin_tokens)
        han_ids = self.han_vocab.encode(han_tokens)
        # 返回原始长度（用于后续 padding）
        return torch.tensor(pinyin_ids, dtype=torch.long), torch.tensor(han_ids, dtype=torch.long)

def collate_fn(batch):
    """batch: list of (pinyin_ids, han_ids) tensors of varying lengths"""
    pinyin_list, han_list = zip(*batch)
    # 计算当前批次最大长度
    max_len = max(p.size(0) for p in pinyin_list)
    pad_id = 0  # 我们约定 <pad> id 为 0

    padded_pinyin = []
    padded_han = []
    masks = []  # 标记有效位置（True=有效，False=padding）

    for p, h in zip(pinyin_list, han_list):
        cur_len = p.size(0)
        # pad 到 max_len
        p_pad = torch.cat([p, torch.full((max_len - cur_len,), pad_id, dtype=torch.long)])
        h_pad = torch.cat([h, torch.full((max_len - cur_len,), pad_id, dtype=torch.long)])
        mask = torch.cat([torch.ones(cur_len, dtype=torch.bool),
                          torch.zeros(max_len - cur_len, dtype=torch.bool)])
        padded_pinyin.append(p_pad)
        padded_han.append(h_pad)
        masks.append(mask)

    return torch.stack(padded_pinyin), torch.stack(padded_han), torch.stack(masks)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    correct = 0

    for pinyin_batch, han_batch, mask in dataloader:
        pinyin_batch = pinyin_batch.to(device)
        han_batch = han_batch.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        logits = model(pinyin_batch)                     # (B, L, num_classes)
        # 计算损失（忽略 pad_id=0）
        loss = criterion(logits.permute(0, 2, 1), han_batch)  # (B, num_classes, L) vs (B, L)
        # 应用 mask 只考虑有效位置
        loss = (loss * mask).sum() / mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()

        # 计算准确率
        preds = torch.argmax(logits, dim=-1)             # (B, L)
        correct += (preds == han_batch).logical_and(mask).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = correct / total_tokens
    return avg_loss, accuracy

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        for pinyin_batch, han_batch, mask in dataloader:
            pinyin_batch = pinyin_batch.to(device)
            han_batch = han_batch.to(device)
            mask = mask.to(device)

            logits = model(pinyin_batch)
            loss = criterion(logits.permute(0, 2, 1), han_batch)
            loss = (loss * mask).sum() / mask.sum()

            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == han_batch).logical_and(mask).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = correct / total_tokens
    return avg_loss, accuracy

if __name__ == '__main__':
    config = Config.from_json()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pinyin_seqs, han_seqs = read_data(config.data_path)
    print(f"Loaded {len(pinyin_seqs)} samples")

    data_pairs = list(zip(pinyin_seqs, han_seqs))
    pinyin_vocab, han_vocab = build_vocab_from_data(data_pairs)
    print(f"Pinyin vocab size: {len(pinyin_vocab)}")
    print(f"Han vocab size: {len(han_vocab)}")

    full_dataset = PinyinHanDataset(pinyin_seqs, han_seqs, pinyin_vocab, han_vocab)
    # 划分训练集和验证集（90% 训练，10% 验证）
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    model = BiGruTagger(pinyin_vocab, han_vocab, config)
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
