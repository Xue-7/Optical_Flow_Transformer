import numpy as np
from train import *
from sklearn.metrics import mean_absolute_error,mean_squared_error
import pandas as pd


def rmse(y_true, y_pred):
    res = []
    for i in range(len(y_pred)):
        a = (y_pred[i] - y_true[i])**2
        res.append(a)
    return np.sqrt(np.mean(res))

def mape(y_true,y_pred):
    res = []
    for i in range(len(y_pred)):
        if y_true[i] < 1e-4:
            continue
        a = np.abs((y_pred[i] - y_true[i]) / (y_true[i] + 1e-9))
        res.append(a)
    return np.mean(res)

def mse(y_true, y_pred):
    res = []
    for i in range(len(y_pred)):
        f = y_pred[i] - y_true[i]
        a = (y_pred[i] - y_true[i])**2
        res.append(a)
    return np.mean(res)

def mae(y_true, y_pred):
    res = []
    for i in range(len(y_pred)):
        a = abs(y_pred[i] - y_true[i])
        res.append(a)
    return np.mean(res)


name = 'epoch20MAE0.97647744station0.csv'
df = pd.read_csv('result/'+name)
#df = df/100
evaluate = []
a = []
b = []
c = []
d = []
for time_step in range(8):

    y_hat = np.array(df.iloc[:,time_step+1]).reshape((-1,1))
    y = np.array(df.iloc[:,time_step+9]).reshape((-1,1))


    MSE = mse(y, y_hat)
    RMSE = rmse(y, y_hat)**0.5
    MAE = mae(y, y_hat)
    MAPE = mape(y, y_hat)
    a.append(MSE)
    b.append(RMSE)
    c.append(MAE)
    d.append(MAPE)
    print('MSE:',MSE)
    print('RMSE:',RMSE)
    print('MAE:',MAE)
    print('MAPE:',MAPE)

    list = []
    list.append(MSE)
    list.append(RMSE)
    list.append(MAE)
    list.append(MAPE)
    evaluate.append(list)
list_tabel = pd.DataFrame(data=zip(*evaluate),index=['MSE','RMSE','MAE','MAPE'])
list_tabel.to_csv(f'evaluate/time_step{time_step}{name}')
print(np.mean(a))
print(np.mean(b))
print(np.mean(c))
print(np.mean(d))
