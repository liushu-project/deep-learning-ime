import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, X):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        return output, state

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, num_layers, dropout=dropout)
        self.dense = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        return output, state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)

