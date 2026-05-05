import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, X, X_valid_len):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(X)

        # 使用 pack_padded_sequence 压紧序列，enforce_sorted=False 允许输入不按长度排序
        # 注意：valid_len 必须在 CPU 上
        packed_embs = pack_padded_sequence(embs, X_valid_len.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, state = self.rnn(packed_embs)

        # 解压序列，output shape: (batch_size, num_steps, hidden_size)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output, state

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 输入是：当前词嵌入 + Encoder 最后的上下文向量
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dense = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs):
        _, last_state = enc_outputs
        return last_state, last_state

    def forward(self, X, state):
        # 这里的 state 是一个元组：(当前 RNN 的隐藏状态, 固定的 Encoder 上下文)
        hidden_state, context_state = state

        X = self.embedding(X) # (batch, seq_len, embed)

        # 始终从 context_state 中取最后一层的状态作为上下文
        context = context_state[-1].unsqueeze(1).repeat(1, X.shape[1], 1)

        # 拼接词嵌入和固定上下文
        X_and_context = torch.cat((X, context), dim=2)

        # 传入当前的 hidden_state 进行计算
        output, hidden_state = self.rnn(X_and_context, hidden_state)

        output = self.dense(output)
        # 返回结果和更新后的 state（固定上下文保持不变）
        return output, (hidden_state, context_state)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, enc_valid_len, dec_X):
        enc_outputs = self.encoder(enc_X, enc_valid_len)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)
