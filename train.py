import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

import argparse
from func import *

def main():
    train_arg = train_args()
    
    # 数据集
    train_dir = train_arg.data_dir+'/train'
    valid_dir = train_arg.data_dir+'/val'
    test_dir = train_arg.data_dir+'/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485,0.456,0.406),
                                                                (0.229,0.224,0.225))])

    test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485,0.456,0.406],
                                                                     [0.229,0.224,0.225])])
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = test_valid_transforms)                               
    test_data = datasets.ImageFolder(test_dir, transform = test_valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data,batch_size = 24,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data,batch_size = 16)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size = 16)

    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    
    model = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}
    # model = {'vgg': vgg16}
    fmodel = model[train_arg.arch]
    for param in fmodel.parameters():#冻结梯度
        param.require_grad = False

    classifier = nn.Sequential(OrderedDict([
                                ('fcl',nn.Linear(25088, 4096)),
                                ('relu1',nn.ReLU()),
                                ('fc2',nn.Linear(4096, 1024)),
                                ('relu2',nn.ReLU()),
                                ('fc3',nn.Linear(1024, 54)),
                                ('output',nn.LogSoftmax(dim = 1))
    ]))
    fmodel.classifier = classifier
    #是否要加载已有模型
    if train_arg.load == 1:
        model_load(fmodel, train_arg.save_dir)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(fmodel.classifier.parameters(), lr=0.001, weight_decay = 1e-4)

    train(fmodel, trainloader, validloader, train_arg.epochs, criterion, 20, optimizer, train_arg.device)

    accuracy_test(fmodel, testloader)

    model_save(fmodel, train_arg.save_dir)

if __name__ == "__main__":
    main()




