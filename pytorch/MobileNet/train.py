import json
import os.path
import sys

import torch
import torch.nn as nn
from torchvision import transforms,datasets
from tqdm import tqdm
import torch.optim as optim

from model_v2 import MobileNetV2

device=("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device".format(device))

transforms={
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val":transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_root=os.path.abspath(os.path.join(os.getcwd(),".."))
image_path=os.path.join(data_root,"AlexNet","data_set","flower_data")
assert os.path.exists(image_path),"path {} does not exist.".format(image_path)
train_dataset=datasets.ImageFolder(image_path+"/train",transform=transforms['train'])

train_num=len(train_dataset)

flower_list=train_dataset.class_to_idx
cla_dict=dict((val,key) for key,val in flower_list.items())
json_str=json.dumps(cla_dict,indent=4)
with open("class_indices.json",'w') as json_file:
    json_file.write(json_str)

batch_size=16
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

val_dataset=datasets.ImageFolder(image_path+"/val",transform=transforms["val"])
val_num=len(val_dataset)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

print("using {} images for training,{} images for validation".format(train_num,val_num))

net=MobileNetV2(num_classes=5)
model_weight_path="./mobilenet_v2_pre.pth"
assert os.path.exists(model_weight_path),"path {} does not exist.".format(model_weight_path)
pre_weight=torch.load(model_weight_path,map_location="cpu")
#输出层的预训练参数权值删掉
pre_dict={k:v for k,v in pre_weight.items() if net.state_dict()[k].numel()==v.numel()}
missing_keys,unexpected_keys=net.load_state_dict(pre_dict,strict=False)
#冻结features层的权值
for param in net.features.parameters():
    param.requires_grad=False

net.to(device)

loss_function=nn.CrossEntropyLoss()
params=[p for p in net.parameters() if p.requires_grad]
optimizer=optim.Adam(params,lr=0.0001)

best_acc=0.0
epochs=1
save_path="./MobileNet_v2.pth"
train_steps=len(train_loader)

for epoch in range(epochs):
    net.train()
    running_loss=0.0
    train_bar=tqdm(train_loader,file=sys.stdout)
    for step,data in enumerate(train_bar):
        images,labels=data
        optimizer.zero_grad()
        output=net(images.to(device))
        loss=loss_function(output,labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss+=loss
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    net.eval()
    acc=0.0
    with torch.no_grad():
        val_bar=tqdm(val_loader,file=sys.stdout)
        for data in val_bar:
            images,labels=data
            output=net(images.to(device))
            #loss=loss_function(output,labels)
            predict_y=torch.max(output,1)[1]
            acc+=torch.eq(predict_y,labels.to(device)).sum().item()
            val_bar.desc="valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
    val_acc=acc/val_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, val_acc))

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(net.state_dict(), save_path)

print("Finished training.")
