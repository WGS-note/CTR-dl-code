# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2021/12/27 9:52 上午
# @File: ctr_MLR.py
'''
MLR
'''
import time

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from deepctr.models import MLR
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.model_selection import train_test_split

from tools import *
from settings import *

'''   train MLR   '''
def train_MLR(parameter, criteo_sampled_data_path, criteo_name, mlr_visual):
    print('MLR 模型训练开始 ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    start_time_start = time.time()

    data, dense_features, sparse_features = deal_with_criteo(criteo_sampled_data_path, criteo_name)
    data = shuffle(data)

    # train_data, test_data, val_data = split_val_test(data, n=90)
    tmp_X, test_X, tmp_y, test_y = train_test_split(data.iloc[:, 1:], data.iloc[:, 0], test_size=0.05, random_state=42, stratify=data.iloc[:, 0])
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.05, random_state=42, stratify=tmp_y)

    print(len(train_y))
    print(len(val_y))
    print(len(test_y))

    # embedding
    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=int(data[feat].max()) + 1, embedding_dim=parameter['embedding_dim'])
                               for i, feat in enumerate(sparse_features)]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

    print(' sparse_features count: ', len(sparse_features))
    print(' dense_features count: ', len(dense_features))

    # 找不到合适的数据，此处为全部特征
    region_feature_columns = sparse_feature_columns + dense_feature_columns
    feature_names = get_feature_names(region_feature_columns)
    print(' feature_names: ', feature_names)

    # feed input
    train_x = {name: train_X[name].values for name in feature_names}
    test_x = {name: test_X[name].values for name in feature_names}
    val_x = {name: val_X[name].values for name in feature_names}
    train_y = train_y.values
    test_y = test_y.values
    val_y = val_y.values
    print(' 数据处理完成', time.strftime("%H:%M:%S", time.localtime(time.time())))

    '''
    region_feature_columns: 用于聚类的用户特征
    base_feature_columns：基模型特征，其实可以是全部特征，也可以是用于训练的广告特征
    l2_reg_linear：LR的正则强度(L2正则)
    bias_feature_columns: 偏好特征，不同的人群具有聚类特性，同一类人群具有类似的广告点击偏好。
    '''
    model = MLR(
        region_feature_columns=region_feature_columns,
        region_num=parameter['region_num'],
        l2_reg_linear=parameter['l2_reg_linear'],
        task='binary',
    )

    mNadam = Adam(lr=parameter['lr'], beta_1=0.95, beta_2=0.96, decay=parameter['lr']/parameter['epochs'])
    model.compile(optimizer=mNadam, loss='binary_crossentropy',
                 metrics=['AUC', 'Precision', 'Recall'])

    print(' 组网完成', time.strftime("%H:%M:%S", time.localtime(time.time())))
    print(' 训练开始 ', time.strftime("%H:%M:%S", time.localtime(time.time())))
    start_time = time.time()
    history = model.fit(
        train_x, train_y, validation_data=(val_x, val_y),
        batch_size=parameter['batch_size'],
        epochs=parameter['epochs'],
        verbose=2,
        shuffle=True,
    )

    end_time = time.time()
    print(' 训练完成', time.strftime("%H:%M:%S", time.localtime(time.time())))
    print((' 训练运行时间： {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60)))

    # # 模型保存成yaml文件
    # save_model(model, save_path_MLR)
    # print(' 模型保存完成', time.strftime("%H:%M:%S", time.localtime(time.time())))

    # 训练可视化
    visualization(history, parameter['epochs'], saveflag=True, showflag=False, path=mlr_visual)

    # 测试集评估
    scores = model.evaluate(test_x, test_y, verbose=2)
    print(' %s: %.4f' % (model.metrics_names[0], scores[0]))
    print(' %s: %.4f' % (model.metrics_names[1], scores[1]))
    print(' %s: %.4f' % (model.metrics_names[2], scores[2]))
    print(' %s: %.4f' % (model.metrics_names[3], scores[3]))
    print(' %s: %.4f' % ('F1', (2 * scores[2] * scores[3]) / (scores[2] + scores[3])))
    y_pre_sc = model.predict(test_x, batch_size=256)
    y_pre = []
    for i in y_pre_sc:
        if i > 0.5:
            y_pre.append(1)
        else:
            y_pre.append(0)
    print(' %s: %.4f' % ('ACC: ', accuracy_score(test_y, y_pre)))
    print(' 测试集评估完成', time.strftime("%H:%M:%S", time.localtime(time.time())))

    end_time_end = time.time()
    print(('MLR 模型训练运行时间： {:.0f}分 {:.0f}秒'.format((end_time_end - start_time_start) // 60, (end_time_end - start_time_start) % 60)))
    print(('{:.0f}小时'.format((end_time_end - start_time_start) // 60 / 60)))


if __name__ == '__main__':
    pass
    parameter = {
        'embedding_dim': 4,
        'region_num': 6,
        'l2_reg_linear': 0.01,
        'dnn_dropout': 0.25,
        'lr': 1e-3,
        'batch_size': 2000,
        'epochs': 1000,
    }

    train_MLR(parameter, criteo_sampled_data_path, criteo_name, visual.format('MLR'))

