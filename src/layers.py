import torch
import torch.nn as nn


class TCJA(nn.Module):
    def __init__(self, kernel_size_t: int = 2, kernel_size_c: int = 1, T: int = 8, channel: int = 128):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=T, out_channels=T,
                              kernel_size=kernel_size_t, padding='same', bias=False)
        self.conv_c = nn.Conv1d(in_channels=channel, out_channels=channel,
                                kernel_size=kernel_size_c, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        x = torch.mean(x_seq.permute(1, 0, 2, 3, 4), dim=[3, 4])
        x_c = x.permute(0, 2, 1)
        conv_t_out = self.conv(x).permute(1, 0, 2)
        conv_c_out = self.conv_c(x_c).permute(2, 0, 1)
        out = self.sigmoid(conv_c_out * conv_t_out)
        y_seq = x_seq * out[:, :, :, None, None]
        return y_seq


class TLA(nn.Module):
    # TODO: 删除无用参数
    def __init__(self, kernel_size_t: int = 2, kernel_size_c: int = 1, T: int = 8, channel: int = 128):
        super().__init__()

        # Excitation
        self.conv_c = nn.Conv1d(in_channels=channel, out_channels=channel,
                                kernel_size=kernel_size_c, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        x = torch.mean(x_seq.permute(1, 0, 2, 3, 4), dim=[3, 4])
        x_c = x.permute(0, 2, 1)
        conv_c_out = self.conv_c(x_c).permute(2, 0, 1)
        out = self.sigmoid(conv_c_out)
        y_seq = x_seq * out[:, :, :, None, None]
        return y_seq


class CLA(nn.Module):
    def __init__(self, kernel_size_t: int = 2, kernel_size_c: int = 1, T: int = 8, channel: int = 128):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=T, out_channels=T,
                              kernel_size=kernel_size_t, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        x = torch.mean(x_seq.permute(1, 0, 2, 3, 4), dim=[3, 4])
        conv_t_out = self.conv(x).permute(1, 0, 2)
        out = self.sigmoid(conv_t_out)
        # max_out = self.con(torch.amax(x_seq, dim =[3,4]))

        y_seq = x_seq * out[:, :, :, None, None]
        return y_seq


class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)