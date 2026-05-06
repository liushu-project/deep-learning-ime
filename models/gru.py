import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import Config

class BiGRU(nn.Module):
    def __init__(self, in_vocab_size: int, out_vocab_size: int, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(in_vocab_size, config.embed_size)

        # bidirectional=True 开启双向
        self.rnn = nn.GRU(config.embed_size, config.hidden_size, config.num_layers,
                          dropout=config.dropout, batch_first=True, bidirectional=True)

        # 因为是双向，GRU 输出的维度是 hidden_size * 2
        self.classifier = nn.Linear(config.hidden_size * 2, out_vocab_size)

    def forward(self, X, X_valid_len):
        # 1. Embedding
        X = self.embedding(X) # (batch, seq_len, embed_size)

        # 2. Pack (处理变长序列，提高效率)
        packed_X = pack_padded_sequence(X, X_valid_len.cpu(), batch_first=True, enforce_sorted=False)

        # 3. GRU
        # packed_output 包含了所有时间步的特征
        packed_output, _ = self.rnn(packed_X)

        # 4. Unpack
        # output shape: (batch, seq_len, hidden_size * 2)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # 5. Classifier (对每个时间步进行分类)
        # logits shape: (batch, seq_len, out_vocab_size)
        logits = self.classifier(output)
        return logits
