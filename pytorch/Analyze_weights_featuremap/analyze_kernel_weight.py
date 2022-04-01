import torch
from resnet_model import resnet34
from alexnet_model import AlexNet
import matplotlib.pyplot as plt
import numpy as np


model = AlexNet(num_classes=5)
# model = resnet34(num_classes=5)
# load model weights
model_weight_path = "../AlexNet/AlexNet.pth"  # "resNet34.pth"
model.load_state_dict(torch.load(model_weight_path))
print(model)

weight_keys=model.state_dict().keys()
for key in weight_keys:
    if "num_batches_tracked" in key:
        continue
    weight_t=model.state_dict()[key].numpy()
    weight_mean = weight_t.mean()
    weight_std = weight_t.std(ddof=1)
    weight_min = weight_t.min()
    weight_max = weight_t.max()
    print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
                                                                weight_std,
                                                                weight_max,
                                                                weight_min))

    # plot hist image
    plt.close()
    weight_vec = np.reshape(weight_t, [-1])
    plt.hist(weight_vec, bins=50)
    plt.title(key)
    plt.show()