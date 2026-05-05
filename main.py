from models.rnn import Encoder, Decoder, EncoderDecoder
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

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def train(net: EncoderDecoder, data_iter, lr, num_epochs, tgt_vocab: LazyVocabulary, device):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            sos = torch.tensor([tgt_vocab.get_id('<sos>')] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([sos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')

def predict(net, test_tokens: list[str], src_vocab: LazyVocabulary, tgt_vocab: LazyVocabulary,
            num_steps: int, device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    test_ids, enc_valid_len = build_ids(src_vocab, test_tokens, num_steps)
    enc_valid_len = torch.tensor([enc_valid_len], device=device)
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(test_ids, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X)
    dec_state = net.decoder.init_state(enc_outputs)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab.get_id('<sos>')], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab.get_id('<eos>'):
            break
        output_seq.append(pred)
    return tgt_vocab.decode(output_seq), attention_weight_seq


if __name__ == '__main__':
    embed_size, hidden_size, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    data_iter, source_vocab, target_vocab = load_data('./data/pinyin-zh-small.txt', batch_size, num_steps)

    # 模型参数
    src_vocab_size = len(source_vocab)
    tgt_vocab_size = len(target_vocab)

    encoder = Encoder(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
    decoder = Decoder(tgt_vocab_size, embed_size, hidden_size, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs, lr = 300, 0.005
    train(net, data_iter, lr, num_epochs, target_vocab, device)

    # 简单测试
    test_pinyin = ['ni', 'hao', 'shi', 'jie']
    print(predict(net, test_pinyin, source_vocab, target_vocab, num_steps, device))
