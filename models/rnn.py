import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, X):
        X = self.embedding(X) # (batch, seq, embed)
        output, state = self.rnn(X) # output: (batch, seq, hidden)
        return output, state

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dense = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs):
        return enc_outputs[1] # (num_layers, batch, hidden)

    def forward(self, X, state):
        X = self.embedding(X) # (batch, seq, embed)
        # 上下文向量: 取最后一层隐状态，并扩展到每个时间步
        context = state[-1].unsqueeze(1) # (batch, 1, hidden)
        context = context.repeat(1, X.shape[1], 1) # (batch, seq, hidden)
        X_and_context = torch.cat((X, context), dim=2) # (batch, seq, embed+hidden)
        output, state = self.rnn(X_and_context, state) # output: (batch, seq, hidden)
        output = self.dense(output) # (batch, seq, vocab)
        return output, state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, init_weights: bool = True, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)):
                    for name, param in m.named_parameters():
                        if 'weight' in name:
                            nn.init.xavier_uniform_(param)
                        elif 'bias' in name:
                            nn.init.constant_(param, 0)

    def forward(self, enc_X, dec_X):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)

