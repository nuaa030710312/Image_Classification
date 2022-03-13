import json
import os.path
import sys

import torch
from torchvision import transforms,datasets
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model import vgg

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


data_transforms={
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]),
    "val":transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
}

data_root=os.path.abspath(os.getcwd())
image_path=os.path.join(data_root,"data_set","flower_data")
assert os.path.exists(image_path),"{} is not exist.".format(image_path)
train_dataset=datasets.ImageFolder(root=image_path+"/train",transform=data_transforms["train"])
train_num=len(train_dataset)

flower_list=train_dataset.class_to_idx
cla_dict=dict((val,key) for key,val in flower_list.items())
json_str=json.dumps(cla_dict,indent=4)
with open("class_indices.json",'w') as json_file:
    json_file.write(json_str)

batch_size=32

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

valid_dataset=datasets.ImageFolder(root=image_path+"/val",transform=data_transforms["val"])
val_num=len(valid_dataset)
valid_loader=torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

print("Using {} images for training,{} images for validation".format(train_num,val_num))

model_name="vgg16"
model=vgg(model_name,num_classes=5,init_weights=True)
model.to(device)
loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.0001)

epochs=1
best_acc=0.0
save_path="./{}Net.pth".format(model_name)
train_step=len(train_loader)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    train_bar=tqdm(train_loader,file=sys.stdout)
    for step,data in enumerate(train_bar):
        images,labels=data
        outputs=model(images.to(device))
        loss=loss_function(outputs,labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss+=loss

        train_bar.desc="train epoch[{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss)

    model.eval()
    acc=0.0
    with torch.no_grad():
        val_bar=tqdm(valid_loader,file=sys.stdout)
        for val_data in val_bar:
            images,labels=val_data
            outputs=model(images.to(device))
            y_predict=torch.max(outputs,dim=1)[1]
            acc+=torch.eq(y_predict,labels.to(device)).sum().item()

        val_acc=acc/val_num
        print("[epoch {}] train_loss:{:.3f} val_accuracy:{:.3f}".format(epoch+1,running_loss/train_step,val_acc))

    if best_acc<val_acc:
        best_acc=val_acc
        torch.save(model.state_dict(),save_path)

print("Finished Training.")

