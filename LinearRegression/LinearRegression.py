#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

# 数据预处理，只读取pm2.5的行
def dataProcess(df):
    '''
    #只读取PM2.5的行
    df=df[df['test'].isin(['PM2.5'])]
    #删除PM2.5的列
    df=df.drop(['test'], axis=1)
    '''
    #相比上面代码，下面一步到位
    pm=df[df['test']=='PM2.5'].iloc[:,1:]
    # x_list, y_list = [], []
    # # astype() 转换array中元素数据类型
    # array = np.array(df).astype(float)
    # 将数据集拆分为多个数据帧
    x_data=[]
    y_data=[]
    array=np.array(pm).astype(float)
    for i in range(15):
        #读取i到i+9列
        tempx=array[:,i:i+9]
        tempy=array[:, i+9]
        x_data.append(tempx)
        y_data.append(tempy)
    # x=np.array(x_data, dtype=float)
    x=np.array(x_data)
    y=np.array(y_data)
    print(x)
    return x, y, array

# 更新参数，训练模型
def train(x_train, y_train, epoch):
    bias = 0 # 偏置值初始化
    weights = np.ones(9) # 权重初始化
    learning_rate = 1 # 初始学习率
    reg_rate = 0.001 # 正则项系数
    bg2_sum = 0 # 用于存放偏置值的梯度平方和
    wg2_sum = np.zeros(9) # 用于存放权重的梯度平方和

    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(9)
        # 在所有数据上计算Loss_label的梯度
        for j in range(3200):
            b_g += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-1)
            for k in range(9):
                w_g[k] += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-x_train[j, 9, k]) 
        # 求平均    
        b_g /= 3200
        w_g /= 3200
        #  加上Loss_regularization在w上的梯度
        for m in range(9):
            w_g[m] += reg_rate * weights[m]
        
        # adagrad
        bg2_sum += b_g**2
        wg2_sum += w_g**2
        # 更新权重和偏置
        bias -= learning_rate/bg2_sum**0.5 * b_g
        weights -= learning_rate/wg2_sum**0.5 * w_g

        # 每训练200轮，输出一次在训练集上的损失
        if i%200 == 0:
            loss = 0
            for j in range(3200):
                loss += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias)**2
            print('after {} epochs, the loss on train data is:'.format(i), loss/3200)

    return weights, bias

# 验证模型效果
def validate(x_val, y_val, weights, bias):
    loss = 0
    for i in range(400):
        loss += (y_val[i] - weights.dot(x_val[i, 9, :]) - bias)**2
    return loss / 400

def main():
    # 从csv中读取有用的信息
    # 由于大家获取数据集的渠道不同，所以数据集的编码格式可能不同
    # 若读取失败，可在参数栏中加入encoding = 'gb18030'
    path=os.getcwd()
    path+='/LinearRegression/'
    #参数usecols读取指定列
    df = pd.read_csv(path+'train.csv', usecols=range(2,27), encoding='gb18030')
    x, y, _ = dataProcess(df)
    #划分训练集与验证集
    x_train, y_train = x[0:3200], y[0:3200]
    x_val, y_val = x[3200:3600], y[3200:3600]
    epoch = 2000 # 训练轮数
    # 开始训练
    w, b = train(x_train, y_train, epoch)
    # 在验证集上看效果
    loss = validate(x_val, y_val, w, b)
    print('The loss on val data is:', loss)
def test():
    path=os.getcwd()
    path+='/LinearRegression/'
    #参数usecols读取指定列
    df = pd.read_csv(path+'train.csv', usecols=range(2,27), encoding='gb18030')
    #只读取PM2.5的行
    df=df[df['test'].isin(['PM2.5'])]
    #删除PM2.5的列
    # df=df.drop(['test'], axis=1)
    # print(df)
    dataProcess(df)
if __name__ == '__main__':
    test()
    
    # main()