import torch.nn as nn
import torch

cfgs={
    "vgg11":[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_features(cfg:list):
    layers=[]
    in_channels=3

    for v in cfg:
        if v=='M':
            layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d=nn.Conv2d(in_channels,v,kernel_size=3,stride=1,padding=1)
            layers+=[conv2d,nn.ReLU(True)]
            in_channels=v

    return  nn.Sequential(*layers)

class Vgg(nn.Module):
    def __init__(self,features,num_classes=1000,init_weights=False):
        super(Vgg,self).__init__()
        self.features=features
        self.classifier=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*7*7,2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048,2048),
            nn.ReLU(True),
            nn.Linear(2048,num_classes)
        )

        if init_weights:
            self._init_weight()

    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1)
        x=self.classifier(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

def vgg(model_name='vgg16',**kwargs):
    assert model_name in cfgs,"warning: model {} is not in cfgs dict".format(model_name)
    cfg=cfgs[model_name]
    model=Vgg(make_features(cfg),**kwargs)
    return model