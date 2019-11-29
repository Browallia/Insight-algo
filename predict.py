import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets,transforms,models
import argparse
import json
from collections import OrderedDict

from func import *

def main():

    pre_arg = predict_args()
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)

    model = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}
    # model = {'vgg' : vgg16}
    fmodel = model[pre_arg.arch]
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
    model_load(fmodel, pre_arg.save_dir)

    result = predict(pre_arg.dirpic,fmodel,pre_arg.topk)
    probs = result[0]
    classes = result[1]
    with open(pre_arg.category_names, 'r',encoding='UTF-8') as f:
        label_id_name = json.load(f)
    
    names = []
    for i in classes:
    
        names.append(label_id_name[i])
    
    print('the top possible category is :')
    for i in names:
        print(i)
    print('with the possiblity of :')
    for i in probs:
        print(i)

if __name__ == "__main__":
    main()
    
