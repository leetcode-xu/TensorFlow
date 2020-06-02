#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from tensorflow import keras
from tensorflow import data, GradientTape, reduce_mean, losses
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Network(keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = keras.layers.Dense(64, activation='relu')
        self.fc2 = keras.layers.Dense(64, activation='relu')
        self.fc3 = keras.layers.Dense(1)


    def call(self, input, training=None, mask=None):
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def norm(datas):
    datas_stats = datas.describe()
    return (datas - datas_stats.loc['mean']) / datas_stats.loc['std']


def dispose_mpg_data(filename='/home/jimao/PycharmProjects/tensorflow/data/auto-mpg.data-original'):
    datas = pd.read_csv(filename, sep='\s+', names=['MPG','Cylinders','Displacement','Horsepower',
                                                    'Weight', 'Acceleration', 'Model Year', 'Origin', 'name'])
    datas = datas.drop(columns=['name'])
    # datas = datas[~datas['MPG'].isna()&~datas['Horsepower'].isna()]
    datas = datas.dropna()
    origin = datas.pop('Origin')
    datas['usa'] = (origin==1) * 1.
    datas['europe'] = (origin==2) * 1.
    datas['japan'] = (origin==3) * 1.
    train_datas = datas.sample(frac=.8, random_state=0)
    test_datas = datas.drop(index=train_datas.index)
    train_lables = train_datas.pop('MPG')
    test_lables = test_datas.pop('MPG')
    test_datas = norm(test_datas)
    train_datas = norm(train_datas)
    # train_db = tf.data.Dataset.from_tensor_slices((train_datas.values, train_lables.values))
    # train_db = train_db.shuffle(100).batch(32)
    return train_datas, train_lables, test_datas, test_lables


def mpg_test(filename='/home/jimao/PycharmProjects/tensorflow/data/auto-mpg.data-original'):
    train_datas, train_lables, test_datas, test_lables = dispose_mpg_data(filename)
    model = Network()
    # 通过 build 函数完成内部张量的创建，其中 4 为任意设置的 batch 数量，9 为输入特征长度
    model.build(input_shape=(4,9))
    # 打印网络信息
    model.summary()
    train_db = data.Dataset.from_tensor_slices((train_datas.values, train_lables.values))
    train_db = train_db.shuffle(100).batch(32)
    optimzer = keras.optimizers.RMSprop(.001)
    for epoch in range(200):
        for step, (x, y) in enumerate(train_db):
            with GradientTape() as tape:
                out = model(x)
                loss = reduce_mean(losses.MSE(y, out))
            # if step % 10 == 0:
            #     print(epoch, step, float(loss))
            grads = tape.gradient(loss, model.trainable_variables)
            optimzer.apply_gradients(zip(grads, model.trainable_variables))
    print(model.variables)
    print(model.trainable_variables)





if __name__ == '__main__':
    # dispose_mpg_data()
    mpg_test()