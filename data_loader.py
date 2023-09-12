# from ClearSkyModel import createPreFZ
# from ClearSkyModel import createTrueFZ
import csv
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 加载训练测试数据集

def normalize_data(data):
    # 归一化
    mean, std = np.mean(data), np.std(data)
    data = (data - mean) / std
    return data

#随机数种子确定
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1234)

class ModelRadarDataset(Dataset):
    def __init__(self, xtw=10, ytw=8, data_path='all_30min_final_adaptive.csv',
                  mode='train',station=0):
        #
        '''
        :param tw: [num_sample,P+Q,N]
        :param TE: [num_sample,P+Q]
        :param transform:
        :param mode:
        '''
        #
        self.xtw = xtw
        self.ytw = ytw
        data= self.Readdata(data_path, station)
        feature, label = self.create_sequences(data, self.xtw, self.ytw)

        x_train = []
        x_test = []
        y_train = []
        y_test = []

        df=pd.read_csv('time_seq.csv')
        for i in range(df.shape[0]):
            date=df['0'][i]
            yyyy,mm,dd=date.split('/')
            if int(dd)%5==3:
                x_test.append(feature[i:i+1, :])
                y_test.append(label[i:i + 1,:])
            else:
                x_train.append(feature[i:i+1, :])
                y_train.append(label[i:i + 1,:])
        x_test = torch.cat(x_test, dim=0)
        y_test = torch.cat(y_test, dim=0)
        x_train = torch.cat(x_train, dim=0)
        y_train = torch.cat(y_train, dim=0)



        if mode == 'train':
            self.X = x_train
            self.Y = y_train
        if mode == 'test':
            self.X = x_test
            self.Y = y_test




    def Readdata(self, path, station):

        df = pd.read_csv(path)
        # 所有值
        data = np.array(df.iloc[:,3:])
        data0 = np.array(df.iloc[:, 3])
        data1 = np.array(df.iloc[:, 4])
        data2 = np.array(df.iloc[:, 5])
        data3 = np.array(df.iloc[:,6:518])
        data4 = np.array(df.iloc[:,518:1030])
        data5 = np.array(df.iloc[:,1030:1542])
        data6 = np.array(df.iloc[:, 1542:2054])

        # 功率实测值
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data0 = scaler.fit_transform(data0.reshape(-1, 1))
        data0 = torch.FloatTensor(data0).view(-1, 1)

        # 辐照度真实值和理想值的差值
        scaler1 = MinMaxScaler(feature_range=(-1, 1))
        data1 = scaler1.fit_transform(data1.reshape(-1, 1))
        data1 = torch.FloatTensor(data1).view(-1, 1)

        # 理想辐照度
        scaler2 = MinMaxScaler(feature_range=(-1, 1))
        data2 = scaler2.fit_transform(data2.reshape(-1, 1))
        data2 = torch.FloatTensor(data2).view(-1, 1)

        # ARP的特征
        scaler3 = MinMaxScaler(feature_range=(-1, 1))
        data3 = scaler3.fit_transform(data3.reshape(-1, 512))
        data3 = torch.FloatTensor(data3).view(-1, 512)

        #CLP
        scaler4 = MinMaxScaler(feature_range=(-1, 1))
        data4 = scaler4.fit_transform(data4.reshape(-1, 512))
        data4 = torch.FloatTensor(data4).view(-1, 512)

        # Cloud_lizi
        scaler5 = MinMaxScaler(feature_range=(-1, 1))
        data5 = scaler5.fit_transform(data5.reshape(-1, 512))
        data5 = torch.FloatTensor(data5).view(-1, 512)

        # 云图特征
        scaler6 = MinMaxScaler(feature_range=(-1, 1))
        data6 = scaler6.fit_transform(data6.reshape(-1, 512))
        data6 = torch.FloatTensor(data6).view(-1, 512)

        data = torch.cat((data0,data1,data2,data3,data4,data5,data6),dim=1)
        return data

    def create_sequences(self, input_data, xtw, ytw):
        x = []
        y = []
        L = len(input_data)
        time_seq = []
        for j in range(249):
            for i in range(19-xtw-ytw+1):
                train_seq = input_data[j*19+i:j*19+i + xtw]
                train_label = input_data[j*19+i + xtw:j*19+i + xtw + ytw,0:1]
                x.append(train_seq.unsqueeze(0))
                y.append(train_label.unsqueeze(0))
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        return x, y


    def __len__(self):
        return len(self.X)

    # yield:
    def __getitem__(self, idx):

        X = self.X[idx, :]
        Y = self.Y[idx,:]
        sample = {
            'X': X,
            'Y': Y
        }
        return sample


class ToTensor(object):
    def __call__(self, sample):
        sample['X'] = torch.from_numpy(sample['X']).type(torch.FloatTensor)
        sample['Y'] = torch.from_numpy(sample['Y']).type(torch.FloatTensor)
        return sample







