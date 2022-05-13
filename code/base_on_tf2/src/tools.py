# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2021/12/30 10:04 上午
# @File: tools.py
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf

'''   处理 criteo_sampled_data.csv   '''
# data, _, _ = deal_with_criteo('../../data/criteo_sampled_data.csv', True, '../../data/criteo_sampled_data_OK.csv')
def deal_with_criteo(path, saveflag=False, savename=''):
    data = pd.read_csv(path)
    # print(data.columns)
    # I1-I13：总共 13 列数值型特征
    # C1-C26：共有 26 列类别型特征
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    # 简单填充缺失
    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features] = data[dense_features].fillna(0)
    # 编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    if saveflag:
        data.to_csv(savename, header=True, index=False)

    return data, dense_features, sparse_features

def negBpow(data, name, label, seqflag=False):
    dc = len(data)
    if seqflag:
        poscount = len(data[data == 1])
        negcount = len(data[data == 0])
    else:
        poscount = len(data[data[label] == 1])
        negcount = len(data[data[label] == 0])
    pos_rate, neg_rate = poscount / dc, negcount / dc
    print(' {}: 正样本数：{}，负样本数：{}，正负样本比: {} : {}'.format(name, poscount, negcount, 1, np.around(negcount / poscount, decimals=4)))
    # print('   正样本占比：{}，负样本占比：{}'.format(pos_rate, neg_rate))
    return poscount, negcount

'''   用于 spare field embedding   '''
def sparseFeature(feat, vocabulary_size, embed_dim):
    return {'spare': feat, 'vocabulary_size': vocabulary_size, 'embed_dim': embed_dim}

'''   用于 dense field embedding   '''
def denseFeature(feat):
    return {'dense': feat}

def F1score(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    # TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    F1score = 2 * precision * recall / (precision + recall)
    return F1score


