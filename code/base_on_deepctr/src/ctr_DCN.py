# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2021/12/27 2:52 下午
# @File: ctr_DCN.py
'''
Deep&Cross
'''
import time
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from deepctr.models import DCN
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tools import *
from settings import *

'''   train DCN   '''
def train_DCN(parameter, criteo_sampled_data_path, criteo_name, wdl_visual):
    print('DCN 模型训练开始 ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    start_time_start = time.time()

    data, dense_features, sparse_features = deal_with_criteo(criteo_sampled_data_path, criteo_name)
    data = shuffle(data, random_state=42)

    # train_data, test_data, val_data = split_val_test(data, n=90)
    tmp_X, test_X, tmp_y, test_y = train_test_split(data.iloc[:, 1:], data.iloc[:, 0], test_size=0.05, random_state=42, stratify=data.iloc[:, 0])
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.05, random_state=42, stratify=tmp_y)

    print(len(train_y))
    print(len(val_y))
    print(len(test_y))

    # embedding
    sparse_feature_columns = [
        SparseFeat(feat, vocabulary_size=int(data[feat].max()) + 1, embedding_dim=parameter['embedding_dim'])
        for i, feat in enumerate(sparse_features)]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

    print(' sparse_features count: ', len(sparse_features))
    print(' dense_features count: ', len(dense_features))

    linear_feature_columns = dense_feature_columns
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns
    feature_names = get_feature_names(sparse_feature_columns + dense_feature_columns)
    print(' feature_names: ', feature_names)

    # feed input
    train_x = {name: train_X[name].values for name in feature_names}
    test_x = {name: test_X[name].values for name in feature_names}
    val_x = {name: val_X[name].values for name in feature_names}
    train_y = train_y.values
    test_y = test_y.values
    val_y = val_y.values
    print(' 数据处理完成', time.strftime("%H:%M:%S", time.localtime(time.time())))

    # train
    model = DCN(linear_feature_columns, dnn_feature_columns,
                cross_num=parameter['cross_num'], cross_parameterization=parameter['cross_parameterization'],
                dnn_hidden_units=parameter['dnn_hidden_units'],
                l2_reg_linear=parameter['l2_reg_linear'], l2_reg_embedding=parameter['l2_reg_embedding'],
                l2_reg_cross=parameter['l2_reg_cross'], l2_reg_dnn=parameter['l2_reg_dnn'],
                dnn_dropout=parameter['dnn_dropout'],
                dnn_activation='relu',
                dnn_use_bn=True,
                task='binary')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-10,
            patience=3,
            verbose=1)
    ]

    mNadam = Adam(learning_rate=parameter['lr'], beta_1=0.95, beta_2=0.96, decay=parameter['lr'] / parameter['epochs'])
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
        callbacks=callbacks
    )

    end_time = time.time()
    print(' 训练完成', time.strftime("%H:%M:%S", time.localtime(time.time())))
    print((' 训练运行时间： {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60)))

    # # 模型保存成yaml文件
    # save_model(model, save_path_DCN)
    # print(' 模型保存完成', time.strftime("%H:%M:%S", time.localtime(time.time())))

    # 训练可视化
    # visualization(history, parameter['epochs'], saveflag=True, showflag=False, path=wdl_visual)

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
    print(('DCN 模型训练运行时间： {:.0f}分 {:.0f}秒'.format((end_time_end - start_time_start) // 60,
                                                  (end_time_end - start_time_start) % 60)))
    print(('{:.0f}小时'.format((end_time_end - start_time_start) // 60 / 60)))


if __name__ == '__main__':
    pass
    parameter = {
        'embedding_dim': 8,
        'cross_num': 3,
        'cross_parameterization': 'vector',
        'l2_reg_cross': 0.1,
        'l2_reg_dnn': 0.01,
        'dnn_hidden_units': (64, 128, 256),
        'l2_reg_linear': 0.01,
        'l2_reg_embedding': 0.01,
        'dnn_dropout': 0.5,
        'lr': 1e-4,
        # 'batch_size': 1000,
        # 'batch_size': 512,
        'batch_size': 256,
        'epochs': 1000,
    }

    train_DCN(parameter, criteo_sampled_data_path, criteo_name, visual.format('DCN'))


