# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/1/4 11:12 上午
# @File: ctr_NFM.py
'''
NFM
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from sklearn.model_selection import train_test_split
import yaml

from tools import *

class NFM(Model):
    def __init__(self, spare_feature_columns, dense_feature_columns, hidden_units, output_dim, activation, droup_out, w_reg):
        super(NFM, self).__init__()
        self.spare_feature_columns = spare_feature_columns
        self.dense_feature_columns = dense_feature_columns
        self.spare_shape = len(spare_feature_columns)
        self.w_reg = w_reg

        # embedding
        self.embedding_layer = {'embed_layer{}'.format(i): layers.Embedding(feat['vocabulary_size'], feat['embed_dim'])
                                for i, feat in enumerate(self.spare_feature_columns)}

        # dnn
        self.DNN = tf.keras.Sequential()
        for hidden in hidden_units:
            self.DNN.add(layers.Dense(hidden))
            self.DNN.add(layers.BatchNormalization())
            self.DNN.add(layers.Activation(activation))
            self.DNN.add(layers.Dropout(droup_out))
        self.DNN.add(layers.Dense(output_dim, activation=None))  # output_dim=1

    def build(self, input_shape):
        self.b = self.add_weight(name='b', shape=(1,), initializer=tf.zeros_initializer(), trainable=True, )
        self.w = self.add_weight(name='w', shape=(self.spare_shape, 1), initializer=tf.random_normal_initializer(), trainable=True, regularizer=tf.keras.regularizers.l2(self.w_reg))

    def call(self, inputs, training=None, mask=None):
        # dense_inputs: 数值特征，13维
        # sparse_inputs： 类别特征，26维
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        # # LR part
        # linear_part = tf.matmul(sparse_inputs, self.w) + self.b  # (batchsize, 1)

        # embedding
        sparse_embeds = [self.embedding_layer['embed_layer{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]  # list
        sparse_embed = tf.convert_to_tensor(sparse_embeds)   # (26, batchsize, embed_dim)
        sparse_embed = tf.transpose(sparse_embed, [1, 0, 2])  # (batchsize, 26, embed_dim)

        # Bi-Interaction Layer
        bi_layer = 0.5 * (tf.pow(tf.reduce_sum(sparse_embed, axis=1), 2) - tf.reduce_sum(tf.pow(sparse_embed, 2), axis=1))  # (batchsize, embed_dim)

        # bi + dense
        x = tf.concat([dense_inputs, bi_layer], axis=1)  # (batchsize, embed_dim + 13)

        # hidden layer
        hidden_part = self.DNN(x)   # (batchsize, 1)

        # output = tf.nn.sigmoid(linear_part + hidden_part)
        output = tf.nn.sigmoid(hidden_part)

        return output


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

    dense_feature_columns = [denseFeature(feat) for feat in dense_features]
    spare_feature_columns = [sparseFeature(feat, data_X[feat].nunique(), config['NFM']['embed_dim']) for feat in
                             sparse_features]

    tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size=0.2, random_state=42, stratify=data_y)
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.1, random_state=42, stratify=tmp_y)

    model = NFM(spare_feature_columns=spare_feature_columns,
                dense_feature_columns=dense_feature_columns,
                hidden_units=config['NFM']['hidden_units'],
                output_dim=config['NFM']['output_dim'],
                activation=config['NFM']['activation'],
                droup_out=config['NFM']['droup_out'],
                w_reg=config['NFM']['w_reg'],)

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

