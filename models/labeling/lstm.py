from utils.vocabulary import Vocabulary
from torch import nn

from config import Config

class BiLstmTagger(nn.Module):
    def __init__(self, in_vocab: Vocabulary, out_vocab: Vocabulary, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(len(in_vocab), config.embed_size, padding_idx=in_vocab.pad_id)
        self.rnn = nn.LSTM(
            input_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=0.2 if config.num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.classifier = nn.Linear(config.hidden_size * 2, len(out_vocab))

    def forward(self, X):
        emb = self.embedding(X)
        out, _ = self.rnn(emb)
        logits = self.classifier(out)
        return logits
