import math
import os
import argparse

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from model import resnet34
from my_dataset import MyDataSet
from data_utils import read_split_data,plot_class_preds
from train_eval_utils import train_one_epoch,evaluate

def main(args):
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(args)
    print('start tensorboard with "tensorboard --logdir=runs",view at https://localhost:6006/')

    tb_writer=SummaryWriter(logdir='runs/flower_experiment')
    if os.path.exists("./weights") is False:
        os.makedirs('./weights')

    train_images_path,train_images_label,val_images_path,val_images_label=read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set=MyDataSet(train_images_path,train_images_label,data_transform['train'])
    val_data_set=MyDataSet(val_images_path,val_images_label,data_transform['val'])

    batch_size=args.batch_size

    train_loader=torch.utils.data.DataLoader(train_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=False,
                                             num_workers=0,
                                             collate_fn=train_data_set.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_data_set.collate_fn)

    model=resnet34(num_classes=args.num_classes).to(device)

    init_img=torch.zeros((1,3,224,224),device=device)
    tb_writer.add_graph(model,init_img)

    if os.path.exists(args.weights):
        weights_dict=torch.laod(args.weights,map_location=device)
        load_weights_dict={k:v for k,v in weights_dict.items()
                           if model.state_dict()[k].numel()==v.numel()}
        model.load_state_dict(load_weights_dict,strict=False)
    else:
        print("not use pre-train weights.")

    if args.freeze_layers:
        print("freeze layers except fc layer.")
        for name,para in model.named_parameters():
            if 'fc' in name:
                para.requires_grad_(False)

    pg=[p for p in model.parameters() if p.requires_grad]
    optimizer=optim.SGD(pg,lr=args.lr,momentum=0.9,weight_decay=0.005)
    lf=lambda x:((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler=lr_scheduler.LambdaLR(optimizer,lr_lambda=lf)

    for epoch in range(args.epochs):
        mean_loss=train_one_epoch(model=model,
                                  optimizer=optimizer,
                                  data_loader=train_loader,
                                  device=device,
                                  epoch=epoch)
        scheduler.step()
        acc=evaluate(model=model,
                     data_loader=val_loader,
                     device=device)

        print("[Epoch {}] accuracy:{}".format(epoch,acc))
        tags=['train_loss','accuracy','learning_rate']
        tb_writer.add_scalar(tags[0],mean_loss,epoch)
        tb_writer.add_scalar(tags[1],acc,epoch)
        tb_writer.add_scalar(tags[2],optimizer.param_groups[0]['lr'],epoch)

        fig=plot_class_preds(net=model,
                             images_dir='./plot_img',
                             transform=data_transform['val'],
                             num_plot=5,
                             device=device)
        if fig is not None:
            tb_writer.add_figure("predictions vs. actuals",
                                 figure=fig,
                                 global_step=epoch)

        tb_writer.add_histogram(tag="conv1",
                                values=model.conv1.weight,
                                global_step=epoch)
        tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=model.layer1[0].conv1.weight,
                                global_step=epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--num_classes',type=int,default=5)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--lr',type=int,default=0.001)
    parser.add_argument('--lrf',type=int,default=0.01)

    img_root="../AlexNet/data_set/flower_data/flower_photos"
    parser.add_argument('--data-path',type=str,default=img_root)

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    opt=parser.parse_args()

    main(opt)