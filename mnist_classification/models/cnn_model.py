import torch
import torch.nn as nn


class ConvolutionalBlock(nn.Module):

    def __init__(self,in_channel,out_channel):
        self.in_channel=in_channel
        self.out_channel=out_channel

        # Block
        ## Conv-ReLU-BatchNorm-Conv(stride=2)-ReLU-BatchNorm
        self.layer = nn.Sequential([
            nn.Conv2d(in_channel,out_channel,(3,3),padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel,out_channel,(3,3),stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel)
        ])

    def forward(self,x):
        # |x| = (batch_size,in_channel,h,w)
        y = self.layer(x)
        # |y| = (batch_size,out_channel,h,w)
        return y