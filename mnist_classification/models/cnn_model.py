import torch
import torch.nn as nn


class ConvolutionalBlock(nn.Module):

    def __init__(self,in_channel,out_channel):
        self.in_channel=in_channel
        self.out_channel=out_channel

        super().__init__()

        # Block
        ## Conv-ReLU-BatchNorm-Conv(stride=2)-ReLU-BatchNorm
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,(3,3),padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel,out_channel,(3,3),stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self,x):
        # |x| = (batch_size,in_channel,h,w)
        y = self.layer(x)
        # |y| = (batch_size,out_channel,h,w)
        return y

    
class ConvolutionalClassifier(nn.Module):

    def __init__(self,output_size):
        self.output_size=output_size

        super().__init__()

        self.blocks=nn.Sequential( # |x| = (n,1,28,28)
            ConvolutionalBlock(1,32),   # (n,32,14,14)
            ConvolutionalBlock(32,64),  # (n,64,7,7)
            ConvolutionalBlock(64,128), # (n,128,4,4)
            ConvolutionalBlock(128,256),# (n,256,2,2)
            ConvolutionalBlock(256,512) # (n,512,1,1)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(512,50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50,output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self,x):
        assert x.dim() > 2

        if x.dim() == 3: # added channel dim
            x = x.reshape(x.shape[0],1,x.shape[-2],x.shape[-1])

        # |x| = (batch_size,1,28,28)
        z = self.blocks(x)
        # |z| = (batch_size,512,1,1)
        # |z.squeeze()| = (batch_size,512)
        y = self.fc_layer(z.squeeze())
        # |y| = (batch_size,output_size)

        return y