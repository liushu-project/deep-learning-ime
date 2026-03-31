import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperparams as hp

class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zero_pad=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, num_units)
        if zero_pad:
            with torch.no_grad():
                self.embedding.weight[0].fill_(0)

    def forward(self, x):
        return self.embedding(x)


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding='same', bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        if padding == 'same':
            effective_kernel = (kernel_size - 1) * dilation + 1
            self.pad_left = (effective_kernel - 1) // 2
            self.pad_right = effective_kernel - 1 - self.pad_left
        elif padding == 'causal':
            pad_len = (kernel_size - 1) * dilation
            self.pad_left = pad_len
            self.pad_right = 0
        else:  # 'valid'
            self.pad_left = 0
            self.pad_right = 0

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=1, padding=0, dilation=dilation, bias=bias)

    def forward(self, x):
        # x: (N, T, C)
        x = x.transpose(1, 2)  # (N, C, T)
        if self.pad_left + self.pad_right > 0:
            x = F.pad(x, (self.pad_left, self.pad_right))
        x = self.conv(x)
        x = x.transpose(1, 2)  # (N, T, out_channels)
        return x


class Normalize(nn.Module):
    def __init__(self, num_features, norm_type):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "bn":
            self.norm = nn.BatchNorm1d(num_features)
        elif norm_type == "ln":
            self.norm = nn.LayerNorm(num_features)
        elif norm_type == "ins":
            self.norm = nn.InstanceNorm1d(num_features, affine=True)
        else:
            self.norm = nn.Identity()

    def forward(self, x, activation_fn=None):
        # x: (N, T, C)
        if self.norm_type in ("bn", "ins"):
            x = x.transpose(1, 2)   # (N, C, T)
            x = self.norm(x)
            x = x.transpose(1, 2)   # (N, T, C)
        else:  # ln or identity
            x = self.norm(x)
        if activation_fn is not None:
            x = activation_fn(x)
        return x


class Conv1dBanks(nn.Module):
    def __init__(self, in_channels, num_units, K, norm_type):
        super().__init__()
        self.K = K
        self.num_units = num_units
        self.convs = nn.ModuleList()
        for k in range(1, K+1):
            self.convs.append(Conv1d(in_channels, num_units, k, padding='same'))
        self.norm = Normalize(num_units * K, norm_type)

    def forward(self, x):
        outputs = self.convs[0](x)
        for conv in self.convs[1:]:
            outputs = torch.cat([outputs, conv(x)], dim=-1)
        outputs = self.norm(outputs, activation_fn=torch.relu)
        return outputs


class Prenet(nn.Module):
    def __init__(self, in_units, num_units, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(in_units, num_units[0])
        self.fc2 = nn.Linear(num_units[0], num_units[1])

    def forward(self, inputs, is_training=True):
        x = torch.relu(self.fc1(inputs))
        x = F.dropout(x, p=self.dropout_rate, training=is_training)
        x = torch.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout_rate, training=is_training)
        return x


class HighwayNet(nn.Module):
    def __init__(self, num_units):
        super().__init__()
        self.linear_H = nn.Linear(num_units, num_units)
        self.linear_T = nn.Linear(num_units, num_units)
        nn.init.constant_(self.linear_T.bias, -1.0)

    def forward(self, inputs):
        H = torch.relu(self.linear_H(inputs))
        T = torch.sigmoid(self.linear_T(inputs))
        C = 1. - T
        return H * T + inputs * C


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)

    def forward(self, inputs, seqlen=None):
        outputs, _ = self.gru(inputs)
        return outputs
