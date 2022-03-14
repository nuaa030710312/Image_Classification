import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2D(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(BasicConv2D, self).__init__()
        self.conv2d=nn.Conv2d(in_channels,out_channels,**kwargs)
        self.relu=nn.ReLU(True)

    def forward(self,x):
        x=self.relu(self.conv2d(x))
        return x

class Inception(nn.Module):
    def __init__(self,in_channels,ch1x1,ch3x3red,ch3x3,ch5x5red,ch5x5,pool_proj):
        super(Inception, self).__init__()
        self.branch1=BasicConv2D(in_channels,ch1x1,kernel_size=1)
        self.branch2=nn.Sequential(
            BasicConv2D(in_channels,ch3x3red,kernel_size=1),
            BasicConv2D(ch3x3red,ch3x3,kernel_size=3,padding=1)
        )
        self.branch3=nn.Sequential(
            BasicConv2D(in_channels,ch5x5red,kernel_size=1),
            BasicConv2D(ch5x5red,ch5x5,kernel_size=5,padding=2)
        )
        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            BasicConv2D(in_channels,pool_proj,kernel_size=1)
        )

    def forward(self,x):
        b1=self.branch1(x)
        b2=self.branch2(x)
        b3=self.branch3(x)
        b4=self.branch4(x)

        output=[b1,b2,b3,b4]
        return torch.cat(output,dim=1)

class InceptionAux(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool=nn.AvgPool2d(kernel_size=5,stride=3)
        self.conv=BasicConv2D(in_channels,128,kernel_size=1)
        self.f1=nn.Linear(2048,1024)
        self.f2=nn.Linear(1024,num_classes)

    def forward(self,x):
        x=self.averagePool(x)
        x=self.conv(x)
        x=torch.flatten(x,1)
        x=F.dropout(x,0.5,training=self.training)
        x=F.relu(self.f1(x),inplace=True)
        x=F.dropout(x,0.5,training=self.training)
        x=self.f2(x)
        return x

class GoogleNet(nn.Module):
    def __init__(self,num_classes,aux_logits,init_weights):
        super(GoogleNet, self).__init__()
        self.aux_logits=aux_logits
        self.conv1=BasicConv2D(3,64,kernel_size=7,stride=2,padding=3)
        self.pool1=nn.MaxPool2d(3,stride=2,ceil_mode=True)
        self.conv2=BasicConv2D(64,64,kernel_size=1)
        self.conv3=BasicConv2D(64,192,kernel_size=3,padding=1)
        self.pool2=nn.MaxPool2d(3,stride=2,ceil_mode=True)

        self.inception3a=Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b=Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3=nn.MaxPool2d(3,stride=2,ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1=InceptionAux(512,num_classes)
            self.aux2=InceptionAux(528,num_classes)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.drop=nn.Dropout(0.4)
        self.fc=nn.Linear(1024,num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.pool2(x)

        x=self.inception3a(x)
        x=self.inception3b(x)
        x=self.pool3(x)
        x=self.inception4a(x)
        if self.training and self.aux_logits:
            aux1=self.aux1(x)

        x=self.inception4b(x)
        x=self.inception4c(x)
        x=self.inception4d(x)
        if self.training and self.aux_logits:
            aux2=self.aux2(x)

        x=self.inception4e(x)
        x=self.pool4(x)
        x=self.inception5a(x)
        x=self.inception5b(x)

        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.drop(x)
        x=self.fc(x)
        if self.training and self.aux_logits:
            return x,aux2,aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=0.01)
                nn.init.constant_(m.bias,0)

# net=GoogleNet(num_classes=5,aux_logits=True,init_weights=True)
#
# input=torch.rand(1,3,224,224)
# o1,o2,o3=net(input)