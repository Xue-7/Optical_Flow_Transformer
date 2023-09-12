import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import time
import csv
f = open('CLP_newsize_ResNet18.csv','a',encoding='utf-8',newline="")
csv_write = csv.writer(f)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
path = 'D:\\CLP_newsize\\'

# 利用ResNet18进行图像特征提取
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.net = models.resnet18(pretrained=True)

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        return output



transform = transforms.Compose([
    transforms.ToTensor()]
)


bound = 15
for root, dirs, files in os.walk("D:\\CLP_newsize"):
    for file in files:
        # if file[0:9] == '20190308':
        #     break
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        imgPath = path + file
        # fileName = imgPath.split('/')[-1]
        # feature_path = 'E:\\SWR\\CLP_Feature\\'+file[0:13]+'.txt'
        img = Image.open(imgPath)
        img = img.convert('RGB')
        img = transform(img)
        # print('图片维度：',img.shape)
        startTime = time.perf_counter()
        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
        # print(x.shape)
        print('CLP:'+file)
        with torch.no_grad():
            model = net()
            y = model(x).cpu()
            y = torch.squeeze(y)
            y = y.data.numpy()
            print(y.shape)
        endTime = time.perf_counter()
        # print("cpu time:" + str(endTime - startTime))
        model2 = model.to(device)
        endTime = time.perf_counter()
        with torch.no_grad():
            x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
            x = x.to(device)
            y = model2(x).cpu()
            y = torch.squeeze(y)
            y = y.data.numpy()
            # print(y)
            # print(y.shape)
        endTime2 = time.perf_counter()
        # print("GPU time:" + str(endTime2 - endTime))
        csv_write.writerow(y)
        # np.savetxt(feature_path, y, delimiter=',')
        # with open(feature_path, 'w') as f:
        #     f.write(y)

