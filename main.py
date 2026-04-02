from __future__ import unicode_literals, print_function, division
from utils.time import time_since
from io import open
import unicodedata
import re
import random
import time
import math
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import IterableDataset, TensorDataset, DataLoader, RandomSampler

device_tag = "cpu"
if torch.cuda.is_available():
    device_tag = "cuda"
elif torch.backends.mps.is_available():
    device_tag = "mps"
device = torch.device(device_tag)

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
MAX_LENGTH = 50


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def build_vocab_lazily(filepath):
    print("Building vocabulary lazily...")
    input_lang = Lang("pinyin")
    output_lang = Lang("zh")

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                continue

            pinyin, zh = parts

            for word in pinyin.split(" "):
                input_lang.addWord(word)
            for char in zh.split(" "):
                output_lang.addWord(char)

    print(f"Vocab size: Pinyin {input_lang.n_words}, Zh {output_lang.n_words}")
    return input_lang, output_lang


def indexesFromSentence(lang, sentence):
    words = sentence.split(" ")
    return [lang.word2index[word] for word in words if word in lang.word2index]


class StreamingSeq2SeqDataset(IterableDataset):
    def __init__(self, filepath, input_lang, output_lang, max_length=50):
        self.filepath = filepath
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_length = max_length

    def __iter__(self):
        # 每次迭代都会打开文件流
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue

                pinyin, zh = parts

                # 过滤超长句子
                if (
                    len(pinyin.split(" ")) >= self.max_length
                    or len(zh.split(" ")) >= self.max_length
                ):
                    continue

                # 转为 IDs
                inp_ids = indexesFromSentence(self.input_lang, pinyin) + [EOS_TOKEN]
                tgt_ids = indexesFromSentence(self.output_lang, zh) + [EOS_TOKEN]

                # 补齐到 MAX_LENGTH (Padding)
                inp_ids += [PAD_TOKEN] * (self.max_length - len(inp_ids))
                tgt_ids += [PAD_TOKEN] * (self.max_length - len(tgt_ids))

                yield (
                    torch.tensor(inp_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )


def get_streaming_dataloader(filepath, batch_size, input_lang=None, output_lang=None):
    if input_lang is None or output_lang is None:
        input_lang, output_lang = build_vocab_lazily(filepath)

    dataset = StreamingSeq2SeqDataset(filepath, input_lang, output_lang, MAX_LENGTH)

    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_TOKEN)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        encoder_outputs,
        encoder_hidden,
        target_tensor=None,
        teacher_forcing_ratio=0.5,
    ):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=device
        ).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None and random.random() < teacher_forcing_ratio:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


def save_checkpoint(
    step,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    loss,
    checkpoint_dir="checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pth")
    torch.save(
        {
            "step": step,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
            "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
            "loss": loss,
        },
        filepath,
    )
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, encoder, decoder, encoder_optimizer, decoder_optimizer):
    if not os.path.exists(filepath):
        print(f"未找到 Checkpoint: {filepath}，将从零开始训练。")
        return 0

    checkpoint = torch.load(filepath, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])
    decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer_state_dict"])

    step = checkpoint["step"]
    loss = checkpoint["loss"]
    print(f"成功加载 Checkpoint: Step {step}, 上次 Loss: {loss:.4f}")
    return step

def evaluate(val_dataloader, encoder, decoder, criterion):
    # 切换为评估模式，关闭 Dropout
    encoder.eval()
    decoder.eval()

    total_loss = 0
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad(): # 验证时不计算梯度
        for data in val_dataloader:
            input_tensor, target_tensor = data
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

            encoder_outputs, encoder_hidden = encoder(input_tensor)

            # 验证时完全关闭 teacher forcing，模拟真实推理场景
            decoder_outputs, _, _ = decoder(
                encoder_outputs,
                encoder_hidden,
                target_tensor=target_tensor, # 传入 target 以便对齐长度，但下面 ratio 设为 0
                teacher_forcing_ratio=0.0
            )

            # 计算 Loss
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            total_loss += loss.item()

            # 计算准确率 (Token-level)
            # decoder_outputs shape: [batch_size, seq_len, vocab_size]
            _, predicted = decoder_outputs.max(dim=2) # 取出概率最大的词汇索引

            # 忽略 PAD_TOKEN 进行准确率计算
            mask = target_tensor != PAD_TOKEN
            correct_tokens += ((predicted == target_tensor) & mask).sum().item()
            total_tokens += mask.sum().item()

    # 恢复训练模式
    encoder.train()
    decoder.train()

    avg_loss = total_loss / len(val_dataloader)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return avg_loss, accuracy

def train(
    train_dataloader,
    val_dataloader,
    encoder,
    decoder,
    n_epochs,
    learning_rate=0.001,
    print_every=100,
    val_every_steps=1000,
    save_every_steps=5000,
    checkpoint_dir="checkpoints",
    resume=False,
    tf_ratio_start=1.0,
    tf_ratio_end=0.1,
    total_steps=500000,
):

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_TOKEN)

    start_step = 0
    if resume and os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if checkpoints:
            latest_checkpoint = max(
                checkpoints, key=lambda x: int(x.split("step")[1].split(".")[0])
            )
            full_path = os.path.join(checkpoint_dir, latest_checkpoint)
            start_step = load_checkpoint(
                full_path, encoder, decoder, encoder_optimizer, decoder_optimizer
            )

    print(f"开始训练... 起始步数: {start_step}")
    start_time = time.time()
    global_step = 0
    print_loss_total = 0

    for epoch in range(1, n_epochs + 1):
        for batch_idx, data in enumerate(train_dataloader):
            global_step += 1

            # 跳过已训练步数
            if global_step <= start_step:
                continue

            # --- 动态计算当前 Step 的 Teacher Forcing Ratio ---
            # 线性衰减公式：从 tf_ratio_start 降到 tf_ratio_end
            if global_step < total_steps:
                current_tf_ratio = tf_ratio_start - (tf_ratio_start - tf_ratio_end) * (
                    global_step / total_steps
                )
            else:
                current_tf_ratio = tf_ratio_end

            input_tensor, target_tensor = data
            input_tensor, target_tensor = (
                input_tensor.to(device),
                target_tensor.to(device),
            )

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_tensor)

            decoder_outputs, _, _ = decoder(
                encoder_outputs,
                encoder_hidden,
                target_tensor,
                teacher_forcing_ratio=current_tf_ratio,
            )

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1),
            )
            loss.backward()

            # 防范梯度爆炸
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)

            encoder_optimizer.step()
            decoder_optimizer.step()

            if global_step % val_every_steps == 0:
                val_loss, val_acc = evaluate(val_dataloader, encoder, decoder, criterion)
                print(f"[{'='*10} VALIDATION {'='*10}]")
                print(f"Step: {global_step} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")
                print("================================")

            print_loss_total += loss.item()

            if global_step % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print(
                    f"[Time: {time_since(start_time)}] Epoch: {epoch} | Step: {global_step} | "
                    f"Loss: {print_loss_avg:.4f} | TF_Ratio: {current_tf_ratio:.2f}"
                )

            # 保存 Checkpoint
            if global_step % save_every_steps == 0:
                save_checkpoint(
                    global_step,
                    encoder,
                    decoder,
                    encoder_optimizer,
                    decoder_optimizer,
                    loss.item(),
                    checkpoint_dir,
                )


if __name__ == "__main__":
    hidden_size = 256
    batch_size = 64
    full_data_path = "./data/pinyin-zh.txt"
    train_data_path = "./data/train.txt"
    val_data_path = "./data/val.txt"

    input_lang, output_lang = build_vocab_lazily(full_data_path)
    train_dataloader = get_streaming_dataloader(train_data_path, batch_size)
    val_dataloader = get_streaming_dataloader(val_data_path, batch_size)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    train(
        train_dataloader,
        val_dataloader,
        encoder,
        decoder,
        n_epochs=100,
    )
