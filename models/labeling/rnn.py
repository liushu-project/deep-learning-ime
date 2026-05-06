from torch import nn

from config import Config

class SimpleRnnTagger(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(input_size, config.embed_size)
        self.rnn = nn.RNN(
            input_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=0.2 if config.num_layers > 1 else 0.0
        )
        self.classifier = nn.Linear(config.hidden_size, output_size)

    def forward(self, X):
        emb = self.embedding(X)
        out, _ = self.rnn(emb)
        logits = self.classifier(out)
        return logits
