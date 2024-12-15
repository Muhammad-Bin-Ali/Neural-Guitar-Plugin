import torch
import torch.nn as nn


class ResidualLayer(nn.Module):
    def __init__(self, dilations, num_channels, kernel_size):
        super().__init__()
        self.conv_tanh = nn.Conv1d(dilation=dilations, in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size)
        self.conv_sigm = nn.Conv1d(dilation=dilations, in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size)
        self.residual = nn.Conv1d(dilation=dilations, in_channels=num_channels, out_channels=num_channels, kernel_size=1)
        self.skip = nn.Conv1d(dilation=dilations, in_channel=num_channels, out_channels=num_channels, kernel_size=1)

    def forward(self, input):
        out1 = self.conv_tanh(input)
        out2 = self.conv_sigm(input)
        out = torch.tanh(out1) * torch.sigmoid(out2)
        skip = self.skip(out)
        out = self.residual(out)
        out = out + input[:, :, -out.size(2) :]  # aligns the two tensors
        return out, skip


class WaveNetModified(nn.Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2):
        super().__init__()
        dilations = [2**depth for depth in range(dilation_depth)] * num_repeat

        self.causalConv = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=kernel_size)
        self.res_layers = nn.ModuleList()
        for depth in dilations:
            res_layer = ResidualLayer(dilations=depth, num_channels=num_channels, kernel_size=kernel_size)
            self.res_layers.append(res_layer)

        # num_channels * num_repeat * dilation_depth gives out total number of outputs from our Residual Stack
        self.linear_mix = nn.Conv1d(in_channels=num_channels * num_repeat * dilation_depth, out_channels=1, kernel_size=1)

    def forward(self, input):
        skips = []

        out = self.causalConv(input)

        for res_layer in self.res_layers:
            out, skip = res_layer(out)
            skips.append(skip)

        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)

        return out
