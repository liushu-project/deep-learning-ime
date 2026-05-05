from models.gru import BiGRU
from typing import Iterator
from utils.vocabulary import LazyVocabulary
import time
import math
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import csv

def load_data(file_path: str, batch_size: int, num_steps: int):
    source_vocab = LazyVocabulary(reserved_tokens=['<unk>', '<pad>', '<sos>', '<eos>'])
    target_vocab = LazyVocabulary(reserved_tokens=['<unk>', '<pad>', '<sos>', '<eos>'])

    raw_source_seqs = []
    raw_target_seqs = []

    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            source_tokens = row[0].split(' ')
            target_tokens = row[1].split(' ')
            raw_source_seqs.append(source_tokens)
            raw_target_seqs.append(target_tokens)

            for token in source_tokens:
                source_vocab.append(token)
            for token in target_tokens:
                target_vocab.append(token)

    total_samples = len(raw_source_seqs)
    if total_samples == 0:
        raise RuntimeError("No data loaded. Check file path and format.")
    print(f"Loaded {total_samples} samples.")

    src_data = []
    src_len = []
    tgt_data = []
    tgt_len = []

    for src_tokens, tgt_tokens in zip(raw_source_seqs, raw_target_seqs):
        src_ids, src_valid_len = build_ids(source_vocab, src_tokens, num_steps)
        tgt_ids, tgt_valid_len = build_ids(target_vocab, tgt_tokens, num_steps)
        src_data.append(src_ids)
        src_len.append(src_valid_len)
        tgt_data.append(tgt_ids)
        tgt_len.append(tgt_valid_len)

    src_tensor = torch.tensor(src_data, dtype=torch.long)
    src_len_tensor = torch.tensor(src_len, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_data, dtype=torch.long)
    tgt_len_tensor = torch.tensor(tgt_len, dtype=torch.long)

    dataset = TensorDataset(src_tensor, src_len_tensor, tgt_tensor, tgt_len_tensor)
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_iter, source_vocab, target_vocab

def build_ids(vocab: LazyVocabulary, tokens: list[str], max_len: int) -> tuple[list[int], int]:
    pad_id = vocab.get_id('<pad>')
    eos_id = vocab.get_id('<eos>')
    ids = vocab.encode(tokens) + [eos_id]
    valid_len = min(len(ids), max_len)
    return pad_or_truncate(ids, max_len, pad_id), valid_len

def pad_or_truncate(seq_ids: list[int], max_len: int, pad_id: int) -> list[int]:
    if len(seq_ids) > max_len:
        return seq_ids[:max_len]
    else:
        return seq_ids + [pad_id] * (max_len - len(seq_ids))

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train(net: BiGRU, data_iter, lr, num_epochs, tgt_vocab, device):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # 依然利用 ignore_index 屏蔽 <pad> 的损失
    pad_id = tgt_vocab.get_id('<pad>')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    net.train()
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)
        for batch in data_iter:
            optimizer.zero_grad()
            # X: 拼音序列, Y: 对应的汉字序列
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            # 直接前向传播
            logits = net(X, X_valid_len)

            # 计算 Loss: 展平 (batch * seq_len, vocab_size) vs (batch * seq_len)
            l = loss_fn(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1))

            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                num_tokens = Y_valid_len.sum()
                metric.add(l.item() * num_tokens, num_tokens)

        print(f'epoch {epoch+1}: loss {metric[0] / metric[1]:.4f}, {metric[1] / timer.stop():.1f} tokens/sec')

def predict(net, test_tokens, src_vocab, tgt_vocab, num_steps, device):
    net.eval()
    # 使用之前的 build_ids 处理输入
    test_ids, valid_len_val = build_ids(src_vocab, test_tokens, num_steps)

    X = torch.tensor([test_ids], device=device)
    L = torch.tensor([valid_len_val], device=device)

    with torch.no_grad():
        logits = net(X, L)
        # 获取概率最大的 token ID
        preds = logits.argmax(dim=2).squeeze(0)

        # 只截取有效长度部分并转为词元
        output_ids = preds[:valid_len_val].tolist()

        # 过滤掉辅助词元
        res_ids = [idx for idx in output_ids if idx not in [tgt_vocab.get_id('<pad>'), tgt_vocab.get_id('<eos>'), tgt_vocab.get_id('<sos>')]]

    return tgt_vocab.decode(res_ids)


if __name__ == '__main__':
    src_embed_size, hidden_size, num_layers = 128, 256, 2
    batch_size, num_steps = 64, 10
    data_iter, source_vocab, target_vocab = load_data('./data/pinyin-zh-small.txt', batch_size, num_steps)

    # 模型参数
    src_vocab_size = len(source_vocab)
    tgt_vocab_size = len(target_vocab)

    net = BiGRU(
        src_vocab_size,
        src_embed_size,
        hidden_size,
        num_layers,
        tgt_vocab_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs, lr = 30, 0.005
    train(net, data_iter, lr, num_epochs, target_vocab, device)

    # 简单测试
    test_pinyin = ['ni', 'hao', 'shi', 'jie']
    print(predict(net, test_pinyin, source_vocab, target_vocab, num_steps, device))
