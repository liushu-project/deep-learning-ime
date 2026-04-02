import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperparams as hp
from modules import Embedding, Conv1d, Conv1dBanks, Prenet, HighwayNet, GRU, Normalize


class Graph(nn.Module):
    def __init__(self, pnyn_vocab_size, hanzi_vocab_size):
        super().__init__()
        self.pnyn_vocab_size = pnyn_vocab_size
        self.hanzi_vocab_size = hanzi_vocab_size

        # Embedding
        self.embed = Embedding(pnyn_vocab_size, hp.embed_size, zero_pad=True)

        # Prenet
        self.prenet = Prenet(
            hp.embed_size, [hp.embed_size, hp.embed_size // 2], hp.dropout_rate
        )

        # Conv1D banks
        self.conv1d_banks = Conv1dBanks(
            in_channels=hp.embed_size // 2,
            num_units=hp.embed_size // 2,
            K=hp.encoder_num_banks,
            norm_type=hp.norm_type,
        )

        # Max pooling
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)

        # Projections
        bank_out_channels = (hp.embed_size // 2) * hp.encoder_num_banks
        self.conv_proj1 = Conv1d(
            bank_out_channels, hp.embed_size // 2, 5, padding="same"
        )
        self.norm_proj1 = Normalize(hp.embed_size // 2, hp.norm_type)
        self.conv_proj2 = Conv1d(
            hp.embed_size // 2, hp.embed_size // 2, 5, padding="same"
        )
        self.norm_proj2 = Normalize(hp.embed_size // 2, hp.norm_type)

        # Highway nets
        self.highway_nets = nn.ModuleList(
            [HighwayNet(hp.embed_size // 2) for _ in range(hp.num_highwaynet_blocks)]
        )

        # Bidirectional GRU
        self.gru = GRU(hp.embed_size // 2, hp.embed_size // 2, bidirectional=True)

        # Output layer
        self.output_dense = nn.Linear(hp.embed_size, hanzi_vocab_size, bias=False)

    def forward(self, x, y=None):
        # x: (N, T) int32
        enc = self.embed(x)  # (N, T, E)

        # Prenet
        prenet_out = self.prenet(enc, self.training)  # (N, T, E/2)

        # Conv1D banks
        enc_banks = self.conv1d_banks(prenet_out)  # (N, T, K * E/2)

        # Max pooling
        enc_banks = enc_banks.transpose(1, 2)  # (N, C, T)
        enc_banks = F.pad(enc_banks, (0, 1))  # (N, C, T+1)
        enc_banks = self.max_pool(enc_banks)  # (N, C, T)
        enc_banks = enc_banks.transpose(1, 2)  # (N, T, C)

        # Projections
        proj1 = self.conv_proj1(enc_banks)  # (N, T, E/2)
        proj1 = self.norm_proj1(proj1, activation_fn=torch.relu)
        proj2 = self.conv_proj2(proj1)  # (N, T, E/2)
        proj2 = self.norm_proj2(proj2, activation_fn=None)

        enc = proj2 + prenet_out  # residual

        # Highway nets
        for highway in self.highway_nets:
            enc = highway(enc)

        # Bidirectional GRU
        enc = self.gru(enc)  # (N, T, E)

        # Readout
        logits = self.output_dense(enc)  # (N, T, vocab_size)
        preds = torch.argmax(logits, dim=-1)  # (N, T)

        if y is not None:
            # Masking for loss and accuracy
            mask = (y != 0).float()
            loss = F.cross_entropy(
                logits.view(-1, self.hanzi_vocab_size), y.view(-1), reduction="none"
            )
            loss = loss.view_as(y)
            loss = (loss * mask).sum() / mask.sum()
            correct = (preds == y).float() * mask
            acc = correct.sum() / mask.sum()
            return loss, acc, preds
        else:
            return preds
