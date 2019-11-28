'''
Insight_algorithm
functions
Version : 1.0.0
'''
import torch
import numpy as np
from torch import nn
from torch import optim

from torchvision import datasets, transforms, models
import argparse

#定义训练参数
def train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load',help = 'load the checkpoint or new training',default = '1')
    parser.add_argument('--data_dir',help = 'path to the image folder')
    parser.add_argument('--save_dir',help = 'path to the training checkpoint')
    parser.add_argument('--arch',help = 'the architechture of the network',default = 'vgg')
    parser.add_argument('--lr',help = 'the learning rate',default = 0.003)
    parser.add_argument('--hidden units',help = 'the hidden units',default = 512)
    parser.add_argument('--epochs',help = 'setting the epochs',type = int, default = 20)
    parser.add_argument('--device',help = 'CPU OR CUDA',default = 'cuda')
    return parser.parse_args()

#定义预测参数
def predict_args():
    parser  =argparse.ArgumentParser()
    
    parser.add_argument('--topk',help = 'print the top N class',default = 3)
    parser.add_argument('--category_names',help = 'the index of the labels to classes',default = './label_id_name.json')
    parser.add_argument('--device',help = 'CPU OR CUDA',default = 'cpu')
    parser.add_argument('--arch',help = 'the architechture of the network',default = 'vgg')
    parser.add_argument('--save_dir',help = 'path to the training checkpoint')
    parser.add_argument('--dirpic',help = 'path to the picture to test')
    return parser.parse_args()

def accuracy_test(model, dataloader):
    correct = 0
    total = 0
    #model.cuda()   #如果用GPU计算则取消注释
    with torch.no_grad():   #验证集关闭梯度计算
        for data in dataloader:
            images, labels = data
            #images, labels = images.to('cuda'),labels.to('cuda')

            outputs = model(images)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predict == labels).sum().item()
    
    print('the accuracy is {:.4f}'.format(correct/total))

def train(model, trainloader, validloader, epochs, criterion, step_to_print, optimizer, device):
    epochs = epochs#迭代次数
    step_to_print = step_to_print
    steps = 0
    # model.to(device)  如果有GPU则输入 device = 'cuda'
    for epoch in range(epochs):
        running_loss = 0
        for step, (inputs, labels) in enumerate(trainloader):
            steps += 1
            # inputs, labels = inputs.to('device'), labels.to('device')
            optimizer.zero_grad()

            #前馈及反馈
            outputs = model(inputs) #正向传播
            loss = criterion(outputs, labels)#计算误差
            loss.backward()#反向传播
            optimizer.step()#更新参数
            
            running_loss += loss.item()
            if steps % step_to_print == 0:
                print('EPOCHS: {}/{}'.format(epoch+1, epochs),
                '|','LOSS: {:.4f}'.format(running_loss/step_to_print))
            accuracy_test(model, validloader)

#加载模型
def model_load(model, ckp_path):
    state_dict = torch.load(ckp_path)
    model.load_state_dict(state_dict)
    print('The model is loaded')

#保存模型
def model_save(model, ckp_path):
    torch.save(model.state_dict(), ckp_path)
    print('The model is saved')

from PIL import Image # 使用image模块导入图片

# 图片处理
def process_image(image):   
    #调整图片大小
    pic = Image.open(image)

    if pic.size[0] < pic.size[1]:
        ratio = float(256) / float(pic.size[0])
    else:
        ratio = float(256) / float(pic.size[1])
    
    new_size = (int(pic.size[0]*ratio),int(pic.size[1]*ratio)) 
    
    pic.thumbnail(new_size) # 缩放为等长等宽
    
    #从图片中心抠出224 *224 的图像
    pic = pic.crop([pic.size[0]/2-112,pic.size[1]/2-112,pic.size[0]/2+112,pic.size[1]/2+112])
    
    #将图片转化为numpy数组
    mean = [0.485,0.456,0.406]
    std = [0.229,0.224,0.225]
    np_image = np.array(pic)
    np_image = np_image/255

    for i in range(2):          # 使用和训练集同样的参数对图片进行数值标准化
        np_image[:,:,i] -= mean[i]
        np_image[:,:,i] /= std[i]

    np_image = np_image.transpose((2,0,1))  #PyTorch 要求颜色通道为第一个维度，但是在 PIL 图像和 Numpy 数组中是第三个维度，所以调整
    np_image = torch.from_numpy(np_image) # 转化为张量
    np_image = np_image.float()
    print(np_image.type)
    return np_image

#预测
def predict(image_path, model, topk=3):
    img = process_image(image_path)
    img = img.unsqueeze(0)#将图片增加一维
    # img = img.cuda()
    result = model(img).topk(topk)
    probs= []
    classes = []
    a = result[0]
    b = result[1].tolist()
    
    for i in a[0]:
        probs.append(torch.exp(i).tolist())
    for n in b[0]:
        classes.append(str(n+1))
    
    return(probs,classes)