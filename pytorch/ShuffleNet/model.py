import torch
import torch.nn as nn
from torch import Tensor

from typing import List,Callable

def channel_shuffle(x:Tensor,groups:int):
    batch_size,num_channels,height,width=x.size()

    channels_per_group=num_channels//groups
    x=x.view(batch_size,groups,channels_per_group,height,width)
    x=torch.transpose(x,1,2).contiguous()
    x=x.view(batch_size,-1,height,width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self,input_c:int,output_c:int,stride:int):
        super(InvertedResidual, self).__init__()

        if stride not in [1,2]:
            raise ValueError("illegal stride value")
        self.stride=stride

        assert output_c%2==0
        branch_features=output_c//2

        assert self.stride!=1 or input_c==branch_features<<1

        if self.stride==2:
            self.branch1=nn.Sequential(
                self.depthwise_conv(input_c,input_c,kernel_s=3,stride=self.stride,padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c,branch_features,kernel_size=1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(True)
            )
        else:
            self.branch1=nn.Sequential()

        self.branch2=nn.Sequential(
            nn.Conv2d(input_c if self.stride>1 else branch_features,branch_features,kernel_size=1,
                      stride=1,padding=0,bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(True),
            self.depthwise_conv(branch_features,branch_features,kernel_s=3,stride=self.stride,padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features,branch_features,kernel_size=1,stride=1,bias=False,padding=0),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(True)
        )

    def depthwise_conv(self,
                       input_c:int,
                       output_c:int,
                       kernel_s:int,
                       stride:int=1,
                       padding:int=0,
                       bias:bool=False):
        return nn.Conv2d(in_channels=input_c,out_channels=output_c,kernel_size=kernel_s,stride=stride,padding=padding,
                         bias=bias,groups=input_c)

    def forward(self,x:Tensor)->Tensor:
        if self.stride==1:
            x1,x2=torch.chunk(x,2,dim=1)
            out=torch.cat((x1,self.branch2(x2)),dim=1)
        else:
            out=torch.cat((self.branch1(x),self.branch2(x)),dim=1)
        out=channel_shuffle(out,2)

        return out

class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repets:List[int],
                 stages_out_channels:List[int],
                 num_classes:int=1000,
                 inverted_residual:Callable[...,nn.Module]=InvertedResidual):
        super(ShuffleNetV2, self).__init__()
        if len(stages_repets)!=3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels)!=5:
            raise  ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels=stages_out_channels

        input_channels=3
        output_channels=self._stage_out_channels[0]
        self.conv1=nn.Sequential(
            nn.Conv2d(input_channels,output_channels,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True)
        )
        input_channels=output_channels
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.stage2:nn.Sequential
        self.stage3:nn.Sequential
        self.stage4:nn.Sequential

        stage_names=["stage{}".format(i) for i in [2,3,4]]
        for name,repeats,output_channels in zip(stage_names,stages_repets,self._stage_out_channels[1:]):
            seq=[inverted_residual(input_channels,output_channels,2)]
            for i in range(repeats-1):
                seq.append(inverted_residual(output_channels,output_channels,1))
            setattr(self,name,nn.Sequential(*seq))
            input_channels=output_channels

        output_channels=self._stage_out_channels[-1]
        self.conv5=nn.Sequential(
            nn.Conv2d(input_channels,output_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True)
        )
        self.fc=nn.Linear(output_channels,num_classes)

    def _forward(self,x:Tensor):
        x=self.conv1(x)
        x=self.maxpool(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.stage4(x)
        x=self.conv5(x)
        x=x.mean([2,3])
        x=self.fc(x)
        return x
    def forward(self,x):
        return self._forward(x)

def shufflenet_v2_x1_0(num_classes=1000):
    model=ShuffleNetV2(stages_repets=[4,8,4],
                       stages_out_channels=[24, 116, 232, 464, 1024],
                       num_classes=num_classes)
    return model

def shuffle_v2_x0_5(num_classes=1000):
    model=ShuffleNetV2(stages_repets=[4,8,4],
                       stages_out_channels=[24, 48, 96, 192, 1024],
                       num_classes=num_classes)
    return model