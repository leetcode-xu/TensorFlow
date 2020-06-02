#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_datas(filename='/home/jimao/PycharmProjects/tensorflow/data/boston_house_prices.csv'):
    datas = pd.read_csv(filename)[['RM', 'LSTAT', 'PTRATIO', 'MEDV']]
    num = datas.shape[0]
    ceshi_x = datas[datas.columns[:-1]][:int(num*0.8)]
    real_x = datas[datas.columns[:-1]][int(num*0.8):]
    ceshi_y = datas[datas.columns[-1]][:int(num*0.8)]
    real_y = datas[datas.columns[-1]][int(num*0.8):]
    return ceshi_x.values, ceshi_y.values.reshape(int(num*0.8), 1), real_x.values, real_y.values.reshape(num-int(num*0.8), 1)


def guiyi(ndarray):
    for i in range(ndarray.shape[1]):
        ndarray[:, i] = (ndarray[:, i] - ndarray[:, i].mean()) / ndarray[:, i].std()
    return ndarray


def get_loss(w, ceshi_x, real_y):
    pre_y = ceshi_x.dot(w)
    cha = pre_y - real_y
    return cha.T.dot(cha) / real_y.shape[0]


def grad_des():
    study = 0.000001
    steps = 10000
    ceshi_x, ceshi_y, real_x, real_y = get_datas()
    ceshi_x = guiyi(ceshi_x)
    ceshi_y = guiyi(ceshi_y)
    # real_y = guiyi(real_y)
    real_x = guiyi(real_x)
    inse = np.ones((ceshi_x.shape[0], 1))
    ceshi_x = np.hstack((ceshi_x, inse))
    inse = np.ones((real_x.shape[0], 1))
    real_x = np.hstack((real_x, inse))
    # print(ceshi_x.shape)
    # print(ceshi_y.shape)
    w = np.zeros((ceshi_x.shape[1], 1))
    for i in range(steps):
        w -= study * ceshi_x.T.dot(ceshi_x.dot(w)- ceshi_y)
    print(get_loss(w, ceshi_x, ceshi_y))
    pre_y = real_x.dot(w)
    pre_y = pre_y * real_y.std() + real_y.mean()
    x = np.arange(pre_y.shape[0])
    # plt.scatter(x, pre_y[:, 0], s=pre_y[:, 0], c='r', marker='o')
    # plt.scatter(x, real_y[:, 0], s=real_y[:, 0], c='b', marker='.')
    plt.plot(x, pre_y[:, 0], '-r')
    plt.plot(x, real_y[:, 0], '-.b')
    plt.show()


def ceshi():
    data = pd.read_csv('data/data.csv', names=['x', 'y'])
    data = data.apply(lambda cloum: (cloum - cloum.mean())/cloum.std())
    num = data.shape[0]
    one = pd.DataFrame({'one': np.ones(num)})
    data = pd.concat([one, data], axis=1)
    xy = data.values
    x = xy[:, :-1]
    y = xy[:, -1].reshape(num, 1)
    w = np.zeros((2, 1))
    step = 5700
    canshu = 0.00001
    for i in range(step):
        w -= canshu * x.T.dot(x.dot(w) - y)
        if i % 50 ==0:
            print((x.dot(w) - y).T.dot(x.dot(w) - y).mean())
    yuce = w.T.dot(x.T)
    plt.scatter(x[:, 1], y[:, 0])
    plt.plot(x[:, 1], yuce[0], '.-r')
    plt.show()


def draw():
    x = np.linspace(-10,10, 1000)
    y = 1 / (1 + np.exp(-x))
    plt.plot(x, y)
    plt.axvline(0, c='k')
    plt.show()


if __name__ == '__main__':
    # grad_des()
    # ceshi()
    draw()