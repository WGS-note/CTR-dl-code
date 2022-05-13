# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2021/12/30 5:45 下午
# @File: ctr_DeepFM.py
'''
DeepFM
'''
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from sklearn.model_selection import train_test_split
import yaml

from tools import *

class DeepFM(Model):
    def __init__(self, spare_feature_columns, dense_feature_columns, k, w_reg, v_reg, hidden_units, output_dim, activation, droup_out):
        super(DeepFM, self).__init__()
        self.spare_feature_columns = spare_feature_columns
        self.dense_feature_columns = dense_feature_columns
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.k = k

        # embedding
        self.embedding_layer = {'embed_layer{}'.format(i): layers.Embedding(feat['vocabulary_size'], feat['embed_dim'])
                                for i, feat in enumerate(self.spare_feature_columns)}

        # 做完embedding后的维度
        self.onedim = len(dense_feature_columns)
        for i, feat in enumerate(self.spare_feature_columns):
            self.onedim += feat['embed_dim']

        # dnn
        self.DNN = tf.keras.Sequential()
        for hidden in hidden_units:
            self.DNN.add(layers.Dense(hidden))
            self.DNN.add(layers.BatchNormalization())
            self.DNN.add(layers.Activation(activation))
            self.DNN.add(layers.Dropout(droup_out))
        # self.DNN.add(layers.Dense(output_dim, activation=None))
        self.DNN.add(layers.Dense(2, activation=None))

    def build(self, input_shape):
        self.b = self.add_weight(name='b', shape=(1,), initializer=tf.zeros_initializer(), trainable=True, )
        self.w = self.add_weight(name='w', shape=(self.onedim, 1), initializer=tf.random_normal_initializer(), trainable=True, regularizer=tf.keras.regularizers.l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(self.onedim, self.k), initializer=tf.random_normal_initializer(), trainable=True, regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, training=None, mask=None):
        # dense_inputs: 数值特征，13维
        # sparse_inputs： 类别特征，26维
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        # embedding
        sparse_embed = tf.concat([self.embedding_layer['embed_layer{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])], axis=1)  # (batchsize, 26*embed_dim)
        # FM、Deep 共享embedding
        x = tf.concat([dense_inputs, sparse_embed], axis=1)  # (batchsize, 26*embed_dim + 13)

        # FM part
        linear_part = tf.matmul(x, self.w) + self.b  # (batchsize, 1)
        inter_cross1 = tf.square(x @ self.v)  # (batchsize, k)
        inter_cross2 = tf.matmul(tf.pow(x, 2), tf.pow(self.v, 2))  # (batchsize, k)
        cross_part = 0.5 * tf.reduce_sum(inter_cross1 - inter_cross2, axis=1, keepdims=True)  # (batchsize, 1)
        fm_output = linear_part + cross_part

        # Deep part
        dnn_out = self.DNN(x)  # (batchsize, 1)

        # output = tf.nn.sigmoid(fm_output + dnn_out)
        # output = tf.nn.sigmoid(0.5 * (fm_output + dnn_out))
        output = tf.nn.softmax(fm_output + dnn_out)
        return output


if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.Loader(f).get_data()
    data = pd.read_csv('../../data/criteo_sampled_data_OK.csv')

    # with open('/ad_ctr/base_on_tf2/src/config.yaml', 'r') as f:
    #     config = yaml.Loader(f).get_data()
    # data = pd.read_csv(config['read_path_criteo'])

    data = shuffle(data, random_state=42)

    data_X = data.iloc[:, 1:]
    data_y = data['label'].values

    # I1-I13：总共 13 列数值型特征
    # C1-C26：共有 26 列类别型特征
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]

    dense_feature_columns = [denseFeature(feat) for feat in dense_features]
    spare_feature_columns = [sparseFeature(feat, data_X[feat].nunique(), config['DeepFM']['embed_dim']) for feat in sparse_features]

    tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size=0.05, random_state=42, stratify=data_y)
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.05, random_state=42, stratify=tmp_y)

    model = DeepFM(spare_feature_columns=spare_feature_columns,
                   dense_feature_columns=dense_feature_columns,
                   k=config['DeepFM']['k'],
                   w_reg=config['DeepFM']['w_reg'],
                   v_reg=config['DeepFM']['v_reg'],
                   # hidden_units=config['DeepFM']['hidden_units'],
                   hidden_units=[64, 128, 128],
                   output_dim=config['DeepFM']['output_dim'],
                   activation=config['DeepFM']['activation'],
                   droup_out=config['DeepFM']['droup_out'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-8,
            patience=3,
            verbose=1)
    ]

    # adam = optimizers.Adam(lr=config['train']['adam_lr'], beta_1=0.95, beta_2=0.96, decay=config['train']['adam_lr'] / config['train']['epochs'])
    adam = optimizers.Adam(lr=1e-4, beta_1=0.95, beta_2=0.96)

    model.compile(
        optimizer=adam,
        loss='binary_crossentropy',
        metrics=[metrics.AUC(), metrics.Precision(), metrics.Recall(), 'accuracy']
        # metrics=[metrics.AUC(), 'accuracy']
    )
    model.fit(
        # train_X.values, train_y,
        # validation_data=(val_X.values, val_y),
        train_X.values, tf.keras.utils.to_categorical(train_y, num_classes=2),
        validation_data=(val_X.values, tf.keras.utils.to_categorical(val_y, num_classes=2)),
        # batch_size=config['train']['batch_size'],
        batch_size=2000,
        # epochs=config['train']['epochs'],
        epochs=30,
        verbose=2,
        shuffle=True,
        callbacks=callbacks,
    )

    # model.summary()
    

    # scores = model.evaluate(test_X.values, test_y, verbose=2)
    scores = model.evaluate(test_X.values, tf.keras.utils.to_categorical(test_y, num_classes=2), verbose=2)
    print(' %s: %.4f' % (model.metrics_names[0], scores[0]))
    print(' %s: %.4f' % (model.metrics_names[1], scores[1]))
    print(' %s: %.4f' % (model.metrics_names[2], scores[2]))
    print(' %s: %.4f' % (model.metrics_names[3], scores[3]))
    print(' %s: %.4f' % ('F1', (2 * scores[2] * scores[3]) / (scores[2] + scores[3])))
    print(' %s: %.4f' % (model.metrics_names[4], scores[4]))
    y_pre_sc = model.predict(test_X.values, batch_size=256)
    print(y_pre_sc)
    y_pre = []
    # for i in y_pre_sc:
    #     if i > 0.5:
    #         y_pre.append(1)
    #     else:
    #         y_pre.append(0)
    for i in y_pre_sc:
        if i[0] >= i[1]:
            y_pre.append(0)
        else:
            y_pre.append(1)
    print('---', f1_score(test_y, y_pre))
    # print(' %s: %.4f' % ('F1: ', (2 * P * R) / (P + R)))
    print(' %s: %.4f' % ('ACC: ', accuracy_score(test_y, y_pre)))



'''
docker run -d --gpus '"device=1"' \
    --rm -it --name ctr_tf_DeepFM \
    -v /data/wangguisen/ctr_note/base_on_tf2:/ad_ctr/base_on_tf2 \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:3.0 \
    sh -c 'python3 -u /ad_ctr/base_on_tf2/src/ctr_DeepFM.py 1>>/ad_ctr/base_on_tf2/log/ctr_DeepFM.log 2>>/ad_ctr/base_on_tf2/log/ctr_DeepFM.err'
'''

