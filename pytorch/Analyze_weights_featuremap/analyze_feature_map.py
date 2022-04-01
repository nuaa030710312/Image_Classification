import torch
import matplotlib.pyplot as plt
from alexnet_model import AlexNet
import numpy as np
from PIL import Image
from torchvision import transforms

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model=AlexNet(num_classes=5)
model_weight_path="../AlexNet/AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
print(model)

img=Image.open("../AlexNet/flower.jpeg")

img=data_transform(img)
img=torch.unsqueeze(img,dim=0)

output=model(img)

for feature_map in output:
    im=np.squeeze(feature_map.detach().numpy())
    im=np.transpose(im,[1,2,0])

    plt.figure()
    for i in range(12):
        ax=plt.subplot(3,4,i+1)
        plt.imshow(im[:,:,i])
    plt.show()

