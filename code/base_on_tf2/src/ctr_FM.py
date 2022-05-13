# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2021/12/30 3:44 下午
# @File: ctr_FM.py
'''
FM
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import yaml

from tools import *

# embedding 代替 w、v
class FM(Model):
    '''   FM   '''
    def __init__(self, feature_fields, embed_dim):
        super(FM, self).__init__()
        self.input_dims = sum(feature_fields) + 1
        self.input_lens = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype=np.long)
        self.linear = layers.Embedding(self.input_dims, 1, input_length=self.input_lens)
        self.embedding = layers.Embedding(self.input_dims, embed_dim, input_length=self.input_lens)

    '''   自定义线性部分的bias偏置项   '''
    def build(self, input_shape):
        self.bias = self.add_weight(shape=(1,),
                                    initializer='random_normal',
                                    trainable=True,
                                    name='linear_bias',
                                    dtype=tf.float32)

    def call(self, x):
        x = x + self.offsets
        # 线性部分
        linear_part = tf.reduce_sum(self.linear(x), axis=1) + self.bias
        # 内积项，这里用embedding代替交叉特征隐向量
        x = self.embedding(x)
        # 当 keepidms=True,保持其二维或者三维的特性,(结果保持其原来维数)
        square_of_sum = tf.square(tf.reduce_sum(x, axis=1, keepdims=True))  # 先加和再平方
        sum_of_square = tf.reduce_sum(x * x, axis=1, keepdims=True)  # 先平方再加和
        cross_part = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=2, keepdims=False)

        x = linear_part + cross_part
        x = tf.sigmoid(x)
        return x

# 有w、v
class FM2(Model):

    def __init__(self, k, w_reg, v_reg):
        super(FM2, self).__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1), trainable=True, initializer=tf.random_normal_initializer(), regularizer=regularizers.l2(self.w_reg))
        self.bias = self.add_weight(name='b', shape=(1,), initializer='random_normal', trainable=True, dtype=tf.float32)
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k), trainable=True, initializer='random_normal', regularizer=regularizers.l2(self.v_reg))

    def call(self, inputs):

        # 线性部分
        linear_part = tf.matmul(inputs, self.w) + self.bias   # (batchsize, 1)
        # 内积项
        inter_cross1 = tf.square(inputs @ self.v)  # (batchsize, k)
        inter_cross2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))  # (batchsize, k)
        cross_part = 0.5 * tf.reduce_sum(inter_cross1 - inter_cross2, axis=1, keepdims=True)  # (batchsize, 1)

        x = linear_part + cross_part
        x = tf.sigmoid(x)
        return x


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.Loader(f).get_data()

    data = pd.read_csv('../../data/criteo_sampled_data_OK.csv')
    data_X = data.iloc[:, 1:]
    data_y = data['label'].values

    # 模型输入的feature_fields，即每列最大数+1，用于embedding
    fields = data_X.max().values + 1
    fields = fields.astype(np.int)

    tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size=0.2, random_state=42, stratify=data_y)
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.1, random_state=42, stratify=tmp_y)

    # model = FM(feature_fields=fields, embed_dim=config['FM']['embed_dim'])
    model = FM2(k=config['FM']['k'], w_reg=config['FM']['w_reg'], v_reg=config['FM']['v_reg'])

    adam = optimizers.Adam(lr=config['train']['adam_lr'], beta_1=0.95, beta_2=0.96,
                           decay=config['train']['adam_lr'] / config['train']['epochs'])
    model.compile(
        optimizer=adam,
        loss='binary_crossentropy',
        metrics=[metrics.AUC(), metrics.Precision(), metrics.Recall()]
    )
    model.fit(
        train_X.values, train_y,
        validation_data=(val_X.values, val_y),
        batch_size=config['train']['batch_size'],
        epochs=config['train']['epochs'],
        verbose=1,
    )

    model.summary()
