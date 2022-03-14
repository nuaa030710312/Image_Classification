import json
import os.path
import sys

import torch
import torch.nn as nn
from torchvision import datasets,transforms
import torch.optim as optim
from tqdm import tqdm
from model import GoogleNet

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device".format(device))

transforms={
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "val":transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

data_root=os.path.abspath(os.path.join(os.getcwd(),".."))
image_path=os.path.join(data_root,"AlexNet","data_set","flower_data")
assert os.path.exists(image_path),"{} path does not exist.".format(image_path)
train_dataset=datasets.ImageFolder(root=os.path.join(image_path,"train"),transform=transforms["train"])

train_num=len(train_dataset)

flower_list=train_dataset.class_to_idx
cla_dict=dict((val,key) for key,val in flower_list.items())
json_str=json.dumps(cla_dict,indent=4)
with open("class_indices.josn","w") as json_file:
    json_file.write(json_str)

batch_size=32
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
val_dataset=datasets.ImageFolder(root=os.path.join(image_path,"val"),transform=transforms["val"])
val_num=len(val_dataset)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
print("using {} images for trianing,{} images for validation".format(train_num,val_num))

net=GoogleNet(num_classes=5,aux_logits=True,init_weights=True)
net.to(device)
loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=0.003)

epochs=2
best_acc=0.0
save_path="./Googlenet.pth"
train_steps=len(train_loader)

for epoch in range(epochs):
    net.train()
    train_bar=tqdm(train_loader,file=sys.stdout)
    running_loss=0.0
    for step,data in enumerate(train_bar):
        images,labels=data
        optimizer.zero_grad()
        logits,aux2_logits,aux1_logits=net(images.to(device))
        loss0=loss_function(logits,labels.to(device))
        loss1=loss_function(aux1_logits,labels.to(device))
        loss2=loss_function(aux2_logits,labels.to(device))
        loss=loss0+loss1*0.3+loss2*0.3
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

        train_bar.desc="[train epoch {}/{}] loss:{:.3f} ".format(epoch+1,epochs,loss)

    net.eval()
    acc=0.0
    with torch.no_grad():
        val_bar=tqdm(val_loader,file=sys.stdout)
        for val_data in val_bar:
            images,labels=val_data
            outputs=net(images.to(device))
            predict_y=torch.max(outputs,dim=1)[1]
            acc+=torch.eq(predict_y,labels.to(device)).sum().item()
    val_acc=acc/val_num
    print("[epoch {}] val_accuracy:{:.3f} train_loss:{:.3f} ".format(epoch+1,val_acc,running_loss/train_steps))

    if best_acc<val_acc:
        best_acc=val_acc
        torch.save(net.state_dict(),save_path)

print("Finished training")
