# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/1/4 3:26 下午
# @File: ctr_FNN.py
'''
FNN
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import yaml

from tools import *


class FM(Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4):
        super(FM, self).__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.b = self.add_weight(name='b', shape=(1,), initializer=tf.zeros_initializer(), trainable=True)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1), initializer=tf.random_normal_initializer(), trainable=True, regularizer=tf.keras.regularizers.l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k), initializer=tf.random_normal_initializer(), trainable=True, regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs):
        # 线性部分
        linear_part = tf.matmul(inputs, self.w) + self.b  # (batchsize, 1)
        # 内积项
        inter_cross1 = tf.square(inputs @ self.v)  # (batchsize, k)
        inter_cross2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))  # (batchsize, k)
        cross_part = 0.5 * tf.reduce_sum(inter_cross1 - inter_cross2, axis=1, keepdims=True)  # (batchsize, 1)

        return tf.sigmoid(linear_part + cross_part)

class DNN(Model):
    def __init__(self, hidden_units=[256, 128, 64], output_dim=1, activation='relu', droup_out=0.):
        super(DNN, self).__init__()
        self.dnn = tf.keras.Sequential()
        for hidden in hidden_units:
            self.dnn.add(layers.Dense(hidden))
            self.dnn.add(layers.BatchNormalization())
            self.dnn.add(layers.Activation(activation))
            self.dnn.add(layers.Dropout(droup_out))
        self.dnn.add(layers.Dense(output_dim, activation=None))

    def call(self, inputs, training=None, mask=None):
        output = self.dnn(inputs)

        return tf.nn.sigmoid(output)

def train1(train_X, test_X, val_X, train_y, test_y, val_y):
    model = FM(k=4, w_reg=1e-4, v_reg=1e-4)
    # optimizer = optimizers.SGD(0.01)
    adam = optimizers.Adam(lr=config['train']['adam_lr'], beta_1=0.95, beta_2=0.96)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    train_dataset = train_dataset.batch(2000).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=3)

    # 评估
    fm_pre = model(test_X)
    fm_pre = [1 if x > 0.5 else 0 for x in fm_pre]

    # 获取FM训练得到的隐向量
    v = model.variables[2]  # [None, onehot_dim, k]
    print('FM隐向量提取完成')

    X_train = tf.cast(tf.expand_dims(train_X, -1), tf.float32)  # [None, onehot_dim, 1]
    X_train = tf.reshape(tf.multiply(X_train, v), shape=(-1, v.shape[0] * v.shape[1]))  # [None, onehot_dim*k]

    hidden_units = [256, 128, 64]
    model = DNN(hidden_units, 1, 'relu')
    optimizer = optimizers.SGD(0.0001)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, train_y))
    train_dataset = train_dataset.batch(2000).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=3)

    # 评估
    X_test = tf.cast(tf.expand_dims(test_X, -1), tf.float32)
    X_test = tf.reshape(tf.multiply(X_test, v), shape=(-1, v.shape[0] * v.shape[1]))
    fnn_pre = model(X_test)
    fnn_pre = [1 if x > 0.5 else 0 for x in fnn_pre]

    print("FM Accuracy: ", accuracy_score(test_y, fm_pre))
    print("FNN Accuracy: ", accuracy_score(test_y, fnn_pre))

def train2(train_X, test_X, val_X, train_y, test_y, val_y):
    '''   预训练FM   '''
    fm_modle = FM(k=8, w_reg=1e-4, v_reg=1e-4)

    adam = optimizers.Adam(lr=config['train']['adam_lr'], beta_1=0.95, beta_2=0.96,)
    fm_modle.compile(
        optimizer=adam,
        loss='binary_crossentropy',
        metrics=[metrics.AUC(), metrics.Precision(), metrics.Recall()]
    )
    fm_modle.fit(
        train_X, train_y,
        # validation_data=(val_X.values, val_y),
        batch_size=config['train']['batch_size'],
        # epochs=config['train']['epochs'],
        epochs=3,
        verbose=1,
    )

    fm_pre = fm_modle(test_X)
    fm_pre = [1 if x > 0.5 else 0 for x in fm_pre]

    '''   获取FM训练得到的隐向量   '''
    v = fm_modle.variables[2]  # [None, onehot_dim, k]
    print('FM隐向量提取完成')
    print(v.shape)

    '''   隐向量代替随机初始化的权重   '''
    train_X = tf.cast(tf.expand_dims(train_X, -1), tf.float32)  # 隐向量是3维的，新增一维，[None, onehot_dim, 1]
    print(train_X.shape)
    print(tf.multiply(train_X, v).shape)
    train_X = tf.reshape(tf.multiply(train_X, v), shape=(-1, v.shape[0] * v.shape[1]))  # [None, onehot_dim*k]
    print(train_X.shape)
    exit()
    val_X = tf.cast(tf.expand_dims(val_X, -1), tf.float32)
    val_X = tf.reshape(tf.multiply(val_X, v), shape=(-1, v.shape[0] * v.shape[1]))
    test_X = tf.cast(tf.expand_dims(test_X, -1), tf.float32)
    test_X = tf.reshape(tf.multiply(test_X, v), shape=(-1, v.shape[0] * v.shape[1]))

    dnn_model = DNN(hidden_units=[256, 128, 64],
                    output_dim=1,
                    activation='relu',
                    droup_out=0.)
    dnn_model.compile(
        optimizer=adam,
        loss='binary_crossentropy',
        metrics=[metrics.AUC(), metrics.Precision(), metrics.Recall()]
    )
    dnn_model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        batch_size=config['train']['batch_size'],
        epochs=config['train']['epochs'],
        verbose=1,
    )

    fnn_pre = dnn_model(test_X)
    fnn_pre = [1 if x > 0.5 else 0 for x in fnn_pre]

    print("FM Accuracy: ", accuracy_score(test_y, fm_pre))
    print("FNN Accuracy: ", accuracy_score(test_y, fnn_pre))


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.Loader(f).get_data()

    data = pd.read_csv('../../data/criteo_sampled_data_OK.csv')
    data_X = data.iloc[:, 1:]
    data_y = data['label'].values

    # I1-I13：总共 13 列数值型特征
    # C1-C26：共有 26 列类别型特征
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]

    tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size=0.2, random_state=42, stratify=data_y)
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.1, random_state=42, stratify=tmp_y)

    # train1(train_X.values, test_X.values, val_X.values, train_y, test_y, val_y)
    train2(train_X.values, test_X.values, val_X.values, train_y, test_y, val_y)

