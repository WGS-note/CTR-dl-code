# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2021/12/30 4:31 下午
# @File: ctr_DCN.py
'''
DCN
'''
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
import yaml

from tools import *

class DeepCrossNet(Model):
    def __init__(self, spare_feature_columns, dense_feature_columns, hidden_units, output_dim, activation, droup_out, layer_num, reg_w, reg_b):
        super(DeepCrossNet, self).__init__()
        self.spare_feature_columns = spare_feature_columns
        self.dense_feature_columns = dense_feature_columns
        self.reg_w = reg_w
        self.reg_b = reg_b
        self.layer_num = layer_num

        # embedding
        self.embedding_layer = {'embed_layer{}'.format(i): layers.Embedding(feat['vocabulary_size'], feat['embed_dim'])
                                for i, feat in enumerate(self.spare_feature_columns)}

        # 做完embedding后的维度，计算Cross部分要用到
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
        self.output_layer = layers.Dense(output_dim, activation=None)

    def build(self, input_shape):
        self.cross_w = [self.add_weight(name='w{}'.format(i), shape=(self.onedim, 1), initializer=tf.random_normal_initializer(), regularizer=regularizers.l2(self.reg_w), trainable=True)
                        for i in range(self.layer_num)]
        # 从公式图来看b为列向量
        self.cross_b = [self.add_weight(name='b{}'.format(i), shape=(self.onedim, 1), initializer=tf.zeros_initializer(), regularizer=regularizers.l2(self.reg_b), trainable=True)
                        for i in range(self.layer_num)]

    def call(self, inputs, training=None, mask=None):
        # dense_inputs: 数值特征，13维
        # sparse_inputs： 类别特征，26维
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        # embedding
        sparse_embed = tf.concat([self.embedding_layer['embed_layer{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])], axis=1)  # (batchsize, 26*embed_dim)
        x = tf.concat([dense_inputs, sparse_embed], axis=1)  # (batchsize, 26*embed_dim + 13)

        # Cross
        x0 = tf.expand_dims(x, axis=2)  # (batchsize, dim, 1)  dim=39
        xl = x0
        for i in range(self.layer_num):
            # 先乘后两项（忽略第一维，(dim, 1)表示第一个样本的特征 ）
            xl_w = tf.transpose(xl, perm=[0, 2, 1]) @ self.cross_w[i]  # (batchsize, 1, 1)
            xl = x0 @ xl_w + self.cross_b[i] + xl  # (batchsize, 39, 1)

        cross_out = tf.squeeze(xl, axis=2)  # (batchsize, 39)

        # DNN
        dnn_out = self.DNN(x)  # # (batchsize, 1)

        x = tf.concat([cross_out, dnn_out], axis=1)
        output = tf.nn.sigmoid(self.output_layer(x))

        return output



if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
    # with open('/ad_ctr/base_on_tf2/src/config.yaml', 'r') as f:
        config = yaml.Loader(f).get_data()

    data = pd.read_csv(config['read_path_criteo'])
    data = shuffle(data, random_state=42)

    data_X = data.iloc[:, 1:]
    data_y = data['label'].values

    # I1-I13：总共 13 列数值型特征
    # C1-C26：共有 26 列类别型特征
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]

    dense_feature_columns = [denseFeature(feat) for feat in dense_features]
    spare_feature_columns = [sparseFeature(feat, data_X[feat].nunique(), config['DCN']['embed_dim']) for feat in sparse_features]

    tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size=0.05, random_state=42, stratify=data_y)
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.05, random_state=42, stratify=tmp_y)

    # model = DeepCrossNet(
    #     spare_feature_columns=spare_feature_columns,
    #     dense_feature_columns=dense_feature_columns,
    #     hidden_units=config['DCN']['hidden_units'],
    #     output_dim=config['DCN']['output_dim'],
    #     activation=config['DCN']['activation'],
    #     # droup_out=config['DCN']['droup_out'],
    #     droup_out=0.5,
    #     layer_num=config['DCN']['layer_num'],
    #     reg_w=config['DCN']['reg_w'],
    #     reg_b=config['DCN']['reg_b']
    # )
    #
    # # adam = optimizers.Adam(lr=config['train']['adam_lr'], beta_1=0.95, beta_2=0.96, decay=config['train']['adam_lr'] / config['train']['epochs'])
    # adam = optimizers.Adam(lr=1e-4, beta_1=0.95, beta_2=0.96, decay=config['train']['adam_lr'] / config['train']['epochs'])
    # model.compile(
    #     optimizer=adam,
    #     loss='binary_crossentropy',
    #     metrics=[metrics.AUC(), metrics.Precision(), metrics.Recall()]
    # )
    # model.fit(
    #     train_X.values, train_y,
    #     validation_data=(val_X.values, val_y),
    #     batch_size=config['train']['batch_size'],
    #     epochs=config['train']['epochs'],
    #     verbose=2,
    # )
    #
    # model.summary()
    #
    # scores = model.evaluate(test_X.values, test_y, verbose=2)
    # print(' %s: %.4f' % (model.metrics_names[0], scores[0]))
    # print(' %s: %.4f' % (model.metrics_names[1], scores[1]))
    # print(' %s: %.4f' % (model.metrics_names[2], scores[2]))
    # print(' %s: %.4f' % (model.metrics_names[3], scores[3]))
    # print(' %s: %.4f' % ('F1', (2 * scores[2] * scores[3]) / (scores[2] + scores[3])))
    # y_pre_sc = model.predict(test_X.values, batch_size=256)
    # y_pre = []
    # for i in y_pre_sc:
    #     if i > 0.5:
    #         y_pre.append(1)
    #     else:
    #         y_pre.append(0)
    # print(' %s: %.4f' % ('ACC: ', accuracy_score(test_y, y_pre)))

    print('模型训练开始 ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    start_time_start = time.time()

    model = DeepCrossNet(
        spare_feature_columns=spare_feature_columns,
        dense_feature_columns=dense_feature_columns,
        hidden_units=[64, 128, 128],
        output_dim=1,
        # activation='relu',
        activation='selu',
        # droup_out=config['DCN']['droup_out'],
        droup_out=0.15,
        layer_num=3,
        reg_w=0.01,
        reg_b=0.01
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # 当‘val_loss’不再下降时候停止训练
            min_delta=1e-8,  # “不再下降”被定义为“减少不超过1e-2”
            patience=3,  # “不再改善”进一步定义为“至少几个epoch”
            verbose=1)
    ]

    # adam = optimizers.Adam(lr=config['train']['adam_lr'], beta_1=0.95, beta_2=0.96, decay=config['train']['adam_lr'] / config['train']['epochs'])
    # adam = optimizers.Adam(lr=1e-4, beta_1=0.95, beta_2=0.96, decay=1e-4/100)
    adam = optimizers.Adam(lr=1e-4, beta_1=0.95, beta_2=0.96)

    model.compile(
        optimizer=adam,
        loss='binary_crossentropy',
        # loss='categorical_crossentropy',
        # metrics=[metrics.AUC(), metrics.Precision(), metrics.Recall()]
        metrics=['accuracy', metrics.AUC()]
        # metrics=['accuracy', metrics.AUC(), F1score]
    )
    model.fit(
        train_X.values, train_y,
        validation_data=(val_X.values, val_y),
        # train_X.values, tf.keras.utils.to_categorical(train_y, num_classes=2),
        # validation_data=(val_X.values, tf.keras.utils.to_categorical(val_y, num_classes=2)),
        batch_size=2000,
        # epochs=30,
        epochs=1000,
        verbose=2,
        shuffle=True,
        callbacks=callbacks,
    )

    scores = model.evaluate(test_X.values, test_y, verbose=2)
    # scores = model.evaluate(test_X.values, tf.keras.utils.to_categorical(test_y, num_classes=2), verbose=2)
    print(' %s: %.4f' % (model.metrics_names[0], scores[0]))
    print(' %s: %.4f' % (model.metrics_names[1], scores[1]))
    print(' %s: %.4f' % (model.metrics_names[2], scores[2]))
    # print(' %s: %.4f' % (model.metrics_names[3], scores[3]))
    # print(' %s: %.4f' % ('F1', (2 * scores[2] * scores[3]) / (scores[2] + scores[3])))
    y_pre_sc = model.predict(test_X.values, batch_size=256)
    y_pre = []
    for i in y_pre_sc:
        if i > 0.5:
            y_pre.append(1)
        else:
            y_pre.append(0)
    P = precision_score(test_y, y_pre)
    R = recall_score(test_y, y_pre)
    print(' %s: %.4f' % ('F1: ', (2 * P * R)/(P + R)))
    print(' %s: %.4f' % ('ACC: ', accuracy_score(test_y, y_pre)))
    print(' 测试集评估完成', time.strftime("%H:%M:%S", time.localtime(time.time())))

    end_time_end = time.time()
    print(('模型训练运行时间： {:.0f}分 {:.0f}秒'.format((end_time_end - start_time_start) // 60,
                                              (end_time_end - start_time_start) % 60)))
    print(('{:.0f}小时'.format((end_time_end - start_time_start) // 60 / 60)))
    start_time_start = time.time()



'''
docker run -d --gpus '"device=1"' \
    --rm -it --name ctr_tf_DCN \
    -v /data/wangguisen/ctr_note/base_on_tf2:/ad_ctr/base_on_tf2 \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:3.0 \
    sh -c 'python3 -u /ad_ctr/base_on_tf2/src/ctr_DCN.py 1>>/ad_ctr/base_on_tf2/log/ctr_DCN.log 2>>/ad_ctr/base_on_tf2/log/ctr_DCN.err'
'''