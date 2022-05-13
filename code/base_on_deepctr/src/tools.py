# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2021/12/27 9:30 上午
# @File: get_data.py
'''
获取数据
'''
import pandas as pd, numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

'''   criteo_sampled_data   '''
def deal_with_criteo(path, name):
    data = pd.read_csv(path)
    print(data.columns)
    # I1-I13：总共 13 列数值型特征
    # C1-C26：共有 26 列类别型特征
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    # 采样1:1
    data = getRata(data, num=1, flag=False)
    negBpow(data, name)
    # 简单填充缺失
    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features] = data[dense_features].fillna(0)
    # 编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    return data, dense_features, sparse_features

'''   DIN、DIEN处理数据   '''
def deal_with_sequential(path):
    # 用户历史点击商品序列、用户历史点击品类序列、推荐广告商品序列、推荐广告商品品类、点击标记
    hist_goods_sequence = []
    hist_class_sequence = []
    ad_goods = []
    ad_class = []
    label = []
    with open(path, mode='r') as f:
        for line in f:
            line = line.strip().split(';')
            hist_goods_sequence.append([int(i) for i in line[0].split()])
            hist_class_sequence.append([int(i) for i in line[1].split()])
            ad_goods.append([int(i) for i in line[2].split()])
            ad_class.append([int(i) for i in line[3].split()])
            label.append([int(i) for i in line[4].split()])

    # padding
    hist_goods_sequence, maxlen = padding_array(hist_goods_sequence)
    hist_class_sequence, maxlen = padding_array(hist_class_sequence)

    feature_dict = {
        'item_id': np.array(ad_goods).flatten(),
        'cate_id': np.array(ad_class).flatten(),
        'hist_item_id': np.array(hist_goods_sequence),
        'hist_cate_id': np.array(hist_class_sequence),
    }
    label = np.array(label).flatten()

    return feature_dict, label, maxlen

'''   padding   '''
def padding_array(ars):
    maxlen = 0
    for i in range(len(ars)):
        if len(ars[i]) > maxlen:
            maxlen = len(ars[i])
    arrs = []
    for i in range(len(ars)):
        if len(ars[i]) != maxlen:
            padds = maxlen - len(ars[i])
            for j in range(padds):
                ars[i].append(0)
        arrs.append(ars[i])
    return arrs, maxlen

'''   只有序列特征是二维，其它基本特征是一维   '''
def dimTo(ars):
    return ars.flatten()

'''   行为序列去重   '''
def duplicate_seq(ars):
    newars = np.array(list(set([tuple(t) for t in ars])))
    return newars, len(newars), len(ars)

'''   正负样本比   '''
def negBpow(data, name):
    dc = len(data)
    poscount = len(data[data['label'] == 1])
    negcount = len(data[data['label'] == 0])
    pos_rate, neg_rate = poscount / dc, negcount / dc
    print(' {}: 正样本数：{}，负样本数：{}，正负样本比: {} : {}'.format(name, poscount, negcount, 1, np.around(negcount / poscount, decimals=4)))
    print('   正样本占比：{}，负样本占比：{}'.format(pos_rate, neg_rate))
    return poscount, negcount

'''   1:1采样，num=1   '''
def getRata(data, num=1, flag=True):
    data_pos = data[data['label'] == 1]
    if flag:
        data_neg = data[data['label'] == 0]
        data_neg = data_neg.dropna()
    else:
        data_neg = data[data['label'] == 0]
    poscount = len(data_pos)
    negcount = len(data_neg)
    data_neg = data_neg.sample(frac=num * poscount / negcount + 0.001, replace=False, axis=0)  # axis=0 行抽取，replace=False无放回
    data = pd.concat([data_pos, data_neg], axis=0, ignore_index=True)
    return data

'''  98:1:1 训练集、验证集、测试集划分'''
def split_val_test(data, n=98):
    data_pos = data[data['label'] == 1].reset_index(drop=True)
    data_neg = data[data['label'] == 0].reset_index(drop=True)
    poscount = len(data_pos)
    negcount = len(data_neg)

    n_tv = (100 - n) / 2
    plens = poscount // 100
    tr, te, vl = int(plens * n), int(plens * n + plens * n_tv), int(plens * n_tv)
    pos_train = data_pos.iloc[0:tr, :]
    pos_test = data_pos.iloc[tr:te, :]
    pos_val = data_pos.iloc[te:, :]

    nlens = negcount // 100
    neg_train = data_neg.iloc[0:tr, :]
    neg_test = data_neg.iloc[tr:te, :]
    neg_val = data_neg.iloc[te:, :]

    traindata = pd.concat([pos_train, neg_train], axis=0, ignore_index=True)
    testdata = pd.concat([pos_test, neg_test], axis=0, ignore_index=True)
    valdata = pd.concat([pos_val, neg_val], axis=0, ignore_index=True)

    return traindata, testdata, valdata

'''   可视化   '''
def visualization(history, dataLen, saveflag=True, showflag=True, path='', eager_flag=False):
    if not eager_flag:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        auc = history.history['auc']
        val_auc = history.history['val_auc']
        pre = history.history['precision']
        val_pre = history.history['val_precision']
        recall = history.history['recall']
        val_recall = history.history['val_recall']
        epochs = range(1, dataLen + 1)
    else:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        auc = history.history['AUC']
        val_auc = history.history['val_AUC']
        pre = history.history['Precision']
        val_pre = history.history['val_Precision']
        recall = history.history['Recall']
        val_recall = history.history['val_Recall']
        epochs = range(1, dataLen + 1)

    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss, 'bo--', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, auc, 'bo--', label='Training auc')
    plt.plot(epochs, val_auc, 'ro-', label='Validation auc')
    plt.title('Training and Validation auc')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, pre, 'bo--', label='Training precision')
    plt.plot(epochs, val_pre, 'ro-', label='Validation precision')
    plt.title('Training and Validation precision')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, recall, 'bo--', label='Training recall')
    plt.plot(epochs, val_recall, 'ro-', label='Validation recall')
    plt.title('Training and Validation recall')
    plt.legend()

    if saveflag:
        plt.savefig(path)
    if showflag:
        plt.show()

'''   labelEncoder [1, n]   '''
def labEncoder_map(arrs, save_flag=False, save_path=""):
    lab = LabelEncoder()
    # 编码是从0开始，因为序列数据padding的也是0，所以这里都加1
    arrs = lab.fit_transform(arrs)
    arrs = arrs + np.ones(len(arrs), dtype=int)

    if save_flag:
        with open(save_path, 'wb') as f:
            pickle.dump(lab, f)

    return lab, arrs

'''   用于序列数据的编码映射   '''
def labEncoder_map_seq(lab, arrseq, save_flag=False, save_path=""):
    # 维护一个字典，便于将序列数据映射字典
    labdict = {}
    for key in lab.classes_:
        labdict[key] = lab.transform([key])[0] + 1

    jlen = len(arrseq[0])
    for i in range(len(arrseq)):
        for j in range(jlen):
            if arrseq[i][j] == 0:
                continue
            if arrseq[i][j] in labdict.keys():
                arrseq[i][j] = labdict[arrseq[i][j]]
            else:
                newenc = len(labdict.keys()) + 1
                labdict[arrseq[i][j]] = newenc
                arrseq[i][j] = newenc

    if save_flag:
        with open(save_path, 'wb') as f:
            pickle.dump(labdict, f)

    return labdict, arrseq

'''   预测时-用于序列数据的编码映射   '''
def encDict_map(arrseq, ldict):
    jlen = len(arrseq[0])
    for i in range(len(arrseq)):
        for j in range(jlen):
            if arrseq[i][j] == 0:
                continue
            try:
                arrseq[i][j] = ldict[arrseq[i][j]]
            except Exception as e:
                print('!!! 有没有在编码规则里的新数据 ！！！')
                print(str(e))
                exit()
    return arrseq

'''   用户历史行为序列长度   '''
def get_behavior_length(arrsseq):
    res = []
    jlen = len(arrsseq[0])
    for i in range(len(arrsseq)):
        tmp = 0
        for j in range(jlen):
            if arrsseq[i][j] != 0:
                tmp += 1
        res.append(tmp)
    return np.array(res)

def get_feature_dict(data_X, data_y, arrs, is_flag=False):
    feature_dict = {}
    # feature_dict[arrs[0]] = data_X[arrs[0]].values
    # feature_dict[arrs[1]] = data_X[arrs[1]].values
    # feature_dict['hist_cate_id'] = data_X[arrs[2]].values
    # feature_dict['cate_id'] = data_X[arrs[3]].values
    if is_flag:
        feature_dict['hist_cate_id'] = data_X[arrs[0]].values
        feature_dict['cate_id'] = data_X[arrs[1]].values
    else:
        feature_dict['hist_cate_id'] = data_X[arrs[0]].values
        feature_dict['cate_id'] = data_X[arrs[1]].values
        feature_dict['seq_length'] = data_X[arrs[2]].values
    label = data_y.values
    return feature_dict, label


if __name__ == '__main__':
    pass
    # deal_with_criteo()
    # feature_dict, label, maxlen = deal_with_sequential('../../data/sample_data.txt')
    # print(feature_dict)
    # print(label)

    # tmpdict = {'hello': '你好', 'hi': '嗨', 'hungry': '饿'}
    # print(tmpdict)
    # with open('../../base_on_deepctr/enc/labdict.pkl','wb') as f:
    #         pickle.dump(tmpdict, f)
    #
    # with open('../../base_on_deepctr/enc/labdict.pkl', 'rb') as f:
    #     lab = pickle.load(f)
    # print(lab)
    # print(type(lab))
