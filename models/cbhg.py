import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dBank(nn.Module):
    """K 个不同尺寸的卷积核并行，输出拼接"""

    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2)
                for k in range(1, K + 1)
            ]
        )
        self.bn = nn.BatchNorm1d(out_channels * K)

    def forward(self, x):
        # x: (B, T, C) → conv 需要 (B, C, T)
        x = x.transpose(1, 2)
        outs = []
        for conv in self.convs:
            out = conv(x)
            # 保证输出长度和输入一致（奇偶kernel的padding差异）
            out = out[:, :, : x.size(2)]
            outs.append(out)
        out = torch.cat(outs, dim=1)  # (B, K*C, T)
        out = F.relu(self.bn(out))
        return out.transpose(1, 2)  # (B, T, K*C)


class HighwayNet(nn.Module):
    def __init__(self, num_units):
        super().__init__()
        self.H = nn.Linear(num_units, num_units)
        self.T = nn.Linear(num_units, num_units)
        nn.init.constant_(self.T.bias, -1.0)  # 论文建议初始化为负数

    def forward(self, x):
        H = F.relu(self.H(x))
        T = torch.sigmoid(self.T(x))
        return H * T + x * (1 - T)


class CBHG(nn.Module):
    def __init__(self, in_dim, K=16, conv_channels=128, num_highway_blocks=4):
        super().__init__()

        # Conv1D Banks
        self.conv_bank = Conv1dBank(in_dim, conv_channels, K)

        # Max Pooling (stride=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        # Conv1D Projections
        self.conv_proj1 = nn.Conv1d(
            K * conv_channels, conv_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv_proj2 = nn.Conv1d(conv_channels, in_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(in_dim)

        # Highway Networks
        self.highways = nn.Sequential(
            *[HighwayNet(in_dim) for _ in range(num_highway_blocks)]
        )

        # Bidirectional GRU
        self.bigru = nn.GRU(in_dim, conv_channels, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: (B, T, in_dim)
        residual = x

        # Conv1D Bank + MaxPool
        out = self.conv_bank(x)  # (B, T, K*C)
        out = out.transpose(1, 2)  # (B, K*C, T)
        out = self.maxpool(out)[:, :, : x.size(1)]  # (B, K*C, T)

        # Conv Projections
        out = F.relu(self.bn1(self.conv_proj1(out)))  # (B, C, T)
        out = self.bn2(self.conv_proj2(out))  # (B, in_dim, T)
        out = out.transpose(1, 2)  # (B, T, in_dim)

        # Residual
        out = out + residual  # (B, T, in_dim)

        # Highway
        out = self.highways(out)  # (B, T, in_dim)

        # BiGRU
        out, _ = self.bigru(out)  # (B, T, 2*C)
        return out


class Prenet(nn.Module):
    def __init__(self, in_dim, num_units, dropout=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, num_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_units[0], num_units[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


class PinyinToChinese(nn.Module):
    def __init__(
        self,
        pinyin_vocab_size,
        hanzi_vocab_size,
        embed_size=256,
        K=16,
        num_highway_blocks=4,
        dropout=0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(pinyin_vocab_size, embed_size, padding_idx=0)

        self.prenet = Prenet(
            embed_size, num_units=[embed_size, embed_size // 2], dropout=dropout
        )

        self.cbhg = CBHG(
            in_dim=embed_size // 2,
            K=K,
            conv_channels=embed_size // 2,
            num_highway_blocks=num_highway_blocks,
        )

        # BiGRU 输出是 embed_size，直接映射到汉字词表
        self.fc_out = nn.Linear(embed_size, hanzi_vocab_size)

    def forward(self, x):
        # x: (B, T) 拼音 token ids
        emb = self.embedding(x)  # (B, T, embed_size)
        pre = self.prenet(emb)  # (B, T, embed_size//2)
        cbhg_out = self.cbhg(pre)  # (B, T, embed_size)
        logits = self.fc_out(cbhg_out)  # (B, T, hanzi_vocab_size)
        return logits
