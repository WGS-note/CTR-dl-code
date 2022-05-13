# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/1/4 5:14 下午
# @File: ctr_AFM.py
'''
AFM
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import yaml

from tools import *

''' Pair-wise-Interaction Layer '''
class Interaction_layer(layers.Layer):
    '''
     二阶特征交叉层，类别特征embedding等价于没有偏置的全连接。
     input shape:  [batchsize, field, embed_dim]
     output shape: [batchsize, field*(field-1)/2, embed_dim]   field = 26*(26-1)/2 = 325
     :except field=4，二阶交叉后=6
    '''
    def __init__(self):
        super(Interaction_layer, self).__init__()

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError('输入的维度为{}，但要求为3'.format(K.ndim(inputs)))

        element_wise_product_list = []
        for i in range(inputs.shape[1]):
            for j in range(i+1, inputs.shape[1]):
                element_wise_product_list.append(tf.multiply(inputs[:, i], inputs[:, j]))  #[batchsize, embed_dim]
        element_wise_product = tf.convert_to_tensor(element_wise_product_list)  # (325, batchsize, embed_dim)
        element_wise_product = tf.transpose(element_wise_product, [1, 0, 2])    # [batchsize, 325, embed_dim]
        return element_wise_product

''' Attention-based Pooling'''
class Attention_layer(layers.Layer):
    '''
     注意力网络就是一个全连接层经过relu激活，通过Softmax映射成注意力权重
     input shape：[batchsize，field，embed_dim]
     output shape：[batchsize，embed_dim]
    '''
    def __init__(self):
        super(Attention_layer, self).__init__()

    def build(self, input_shape):  # [batchsize，field，embed_dim]
        self.attention_w = layers.Dense(input_shape[1], activation='relu')
        self.attention_h = layers.Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError('输入的维度为{}，但要求为3'.format(K.ndim(inputs)))

        x = self.attention_w(inputs)  # [batchsize，field，field]
        x = self.attention_h(x)       # [batchsize，field，1]
        a_score = tf.nn.softmax(x)
        a_score = tf.transpose(a_score, [0, 2, 1])  # # [batchsize，1，field]
        output = tf.reshape(tf.matmul(a_score, inputs), shape=(-1, inputs.shape[2]))  # [batchsize，embed_dim]
        return output

class AFM(Model):
    def __init__(self, spare_feature_columns, dense_feature_columns):
        super(AFM, self).__init__()
        self.spare_feature_columns = spare_feature_columns
        self.dense_feature_columns = dense_feature_columns
        self.embed_layer = {'emb_{}'.format(i): layers.Embedding(feat['vocabulary_size'], feat['embed_dim'])
                            for i, feat in enumerate(self.spare_feature_columns)}
        self.interaction_layer = Interaction_layer()
        self.attention_layer = Attention_layer()
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        if K.ndim(inputs) != 2:
            raise ValueError('输入维度为{}，但要求为2'.format(K.ndim(inputs)))

        # dense_inputs: 数值特征，13维
        # sparse_inputs： 类别特征，26维
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        embed = [self.embed_layer['emb_{}'.format(i)](sparse_inputs[:, i])
                 for i in range(sparse_inputs.shape[1])]  # list
        embed = tf.convert_to_tensor(embed)   # (26, batchsize, embed_dim)
        embed = tf.transpose(embed, [1, 0, 2])      # [batchsize，26，embed_dim]

        # Pair-wise Interaction
        embed = self.interaction_layer(embed)   # [batchsize, field, embed_dim]  field：两两交叉后的个数

        # Attention-based Pooling
        x = self.attention_layer(embed)   # (batchsize, embed_dim)

        x = tf.reshape(tf.reduce_sum(x, axis=1), shape=(-1, 1))  # [batchsize，1]
        output = tf.nn.sigmoid(self.output_layer(x))
        return output


if __name__ == '__main__':
    # with open(read_path_criteo, 'r') as f:
    with open('/ad_ctr/base_on_tf2/src/config.yaml', 'r') as f:
        config = yaml.Loader(f).get_data()

    data = pd.read_csv(config['read_path_criteo'])
    data_X = data.iloc[:, 1:]
    data_y = data['label'].values

    # I1-I13：总共 13 列数值型特征
    # C1-C26：共有 26 列类别型特征
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]

    dense_feature_columns = [denseFeature(feat) for feat in dense_features]
    spare_feature_columns = [sparseFeature(feat, data_X[feat].nunique(), 4) for feat in
                             sparse_features]

    tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size=0.2, random_state=42, stratify=data_y)
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.1, random_state=42, stratify=tmp_y)

    model = AFM(spare_feature_columns=spare_feature_columns, dense_feature_columns=dense_feature_columns,)

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
        batch_size=2000,
        epochs=config['train']['epochs'],
        verbose=2,
    )

    model.summary()
