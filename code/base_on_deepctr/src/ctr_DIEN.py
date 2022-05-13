# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2021/12/27 2:53 下午
# @File: ctr_DIEN.py
'''
DIEN
'''
import time

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from deepctr.models import DIEN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from sklearn.model_selection import train_test_split
import pickle

from tools import *
from settings import *

'''   基于 sample_data.txt 实现的，目的是快速搭建   '''
def train_DIEN(parameter, sequential_data_path, encoder_paths, visual):
    print('DIEN 模型训练开始 ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    start_time_start = time.time()

    feature_dict, label, maxlen = deal_with_sequential(sequential_data_path)

    # encoder
    lab_item_id, feature_dict['item_id'] = labEncoder_map(feature_dict['item_id'], save_flag=False, save_path=encoder_paths[0])
    lab_cate_id, feature_dict['cate_id'] = labEncoder_map(feature_dict['cate_id'], save_flag=False, save_path=encoder_paths[1])
    _, feature_dict['hist_item_id'] = labEncoder_map_seq(lab_item_id, feature_dict['hist_item_id'], save_flag=False, save_path=encoder_paths[2])
    _, feature_dict['hist_cate_id'] = labEncoder_map_seq(lab_cate_id, feature_dict['hist_cate_id'], save_flag=False, save_path=encoder_paths[3])

    # 对基础特征进行 embedding
    bise_feature = [SparseFeat('item_id', vocabulary_size=len(np.unique(feature_dict['item_id'])), embedding_dim=parameter['bise_embedding_dim'][0]),
                    SparseFeat('cate_id', vocabulary_size=len(np.unique(feature_dict['cate_id'])), embedding_dim=parameter['bise_embedding_dim'][1]), ]

    # 用户历史行为序列长度
    behavior_length = get_behavior_length(feature_dict['hist_item_id'])
    feature_dict["seq_length"] = behavior_length

    # 历史行为序列embedding
    behavior_feature = [VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=feature_dict['hist_item_id'].max() + 1, embedding_dim=parameter['hist_embedding_dim'][0], embedding_name='item_id'), maxlen=maxlen, length_name="seq_length"),
                        VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=feature_dict['hist_cate_id'].max() + 1, embedding_dim=parameter['hist_embedding_dim'][1], embedding_name='cate_id'), maxlen=maxlen, length_name="seq_length")]

    # 指定历史行为序列对应的特征
    behavior_feature_list = ['item_id', 'cate_id']

    # # 负采样：neg_hist_item_id 包含了每一个用户的负采样序列，
    # # 实际环境中对于CTR数据肯定负样本要远大于正样本，这里负采样序列的意思是：对于每个用户分组，负采样其中的行为序列。
    # # 这里方便演示，用行为序列的数据。
    feature_dict['neg_hist_item_id'] = feature_dict['hist_item_id']
    feature_dict['neg_hist_cate_id'] = feature_dict['hist_cate_id']
    neg_behavior_feature = [VarLenSparseFeat(SparseFeat('neg_hist_item_id', vocabulary_size=feature_dict['neg_hist_item_id'].max() + 1, embedding_dim=parameter['hist_embedding_dim'][0], embedding_name='item_id'), maxlen=maxlen, length_name="seq_length"),
        VarLenSparseFeat(SparseFeat('neg_hist_cate_id', feature_dict['neg_hist_cate_id'].max() + 1, embedding_dim=parameter['hist_embedding_dim'][1], embedding_name='cate_id'), maxlen=maxlen, length_name="seq_length")]

    feature_columns = bise_feature + behavior_feature + neg_behavior_feature
    feature_names = get_feature_names(feature_columns)
    print(' feature_names: ', feature_names)

    # feed input
    x = {name: feature_dict[name] for name in feature_names}
    y = label

    print(x)
    print(y)

    print(' 数据处理完成', time.strftime("%H:%M:%S", time.localtime(time.time())))

    # train
    model = DIEN(dnn_feature_columns=feature_columns, history_feature_list=behavior_feature_list,
                 gru_type=parameter['gru_type'],
                 use_negsampling=parameter['use_negsampling'],
                 alpha=parameter['alpha'],
                 dnn_hidden_units=parameter['dnn_hidden_units'],
                 l2_reg_embedding=parameter['l2_reg_embedding'],
                 l2_reg_dnn=parameter['l2_reg_dnn'],
                 dnn_dropout=parameter['dnn_dropout'],
                 dnn_activation=parameter['dnn_activation'],
                 att_activation=parameter['att_activation'],
                 task='binary')

    # model.summary()
    print(' 训练开始 ', time.strftime("%H:%M:%S", time.localtime(time.time())))
    start_time = time.time()

    mNadam = Adam(learning_rate=parameter['lr'], beta_1=0.9, beta_2=0.999, decay=parameter['lr'] / parameter['epochs'])
    model.compile(optimizer=mNadam,
                  loss='binary_crossentropy',
                  metrics=['AUC', 'Precision', 'Recall'])

    history = model.fit(
        x, y, validation_data=(x, y),
        batch_size=parameter['batch_size'],
        epochs=parameter['epochs'],
        verbose=2,
    )

    end_time = time.time()
    print(' 训练完成', time.strftime("%H:%M:%S", time.localtime(time.time())))
    print((' 训练运行时间： {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60)))

    # # 保存成yaml文件,用于tf serving在线服务
    # deep.save(settings.save_path.format('DIEN', 'DIENmodel-11-13_serving'), save_format="tf")
    # # 保存成h5文件，用于离线评估
    # save_model(deep, settings.save_path.format('DIEN', 'DIENmodel-11-13.h5'))

    # predict_DIEN(model, sequential_data_path, encoder_paths, feature_names)

    # 训练可视化
    visualization(history, parameter['epochs'], saveflag=True, showflag=False, path=visual, eager_flag=True)

    # 测试集评估
    scores = model.evaluate(x, y, verbose=2)
    print(' %s: %.4f' % (model.metrics_names[0], scores[0]))
    print(' %s: %.4f' % (model.metrics_names[1], scores[1]))
    print(' %s: %.4f' % (model.metrics_names[2], scores[2]))
    print(' %s: %.4f' % (model.metrics_names[3], scores[3]))
    if scores[2] != 0 and scores[3] != 0:
        print(' %s: %.4f' % ('F1', (2 * scores[2] * scores[3]) / (scores[2] + scores[3])))
    print(' 测试集评估完成', time.strftime("%H:%M:%S", time.localtime(time.time())))

    end_time_end = time.time()
    print(('DIEN 模型训练运行时间： {:.0f}分 {:.0f}秒'.format((end_time_end - start_time_start) // 60,
                                                  (end_time_end - start_time_start) % 60)))
    print(('{:.0f}小时'.format((end_time_end - start_time_start) // 60 / 60)))

'''   和tf2的DIN用的数据集一样，amazon-books-100k-preprocessed.csv'''
def train_DIEN2(parameter, sequential_data_path, encoder_paths, visual):
    print('DIEN 模型训练开始 ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    start_time_start = time.time()

    data = pd.read_csv(sequential_data_path)

    hist = []
    for i in range(40):
        hist.append('hist_cate_{}'.format(i))

    # 序列最大长度
    maxlen = 40
    # 序列emb词典最大值
    maxseqval = 0
    for i in hist:
        if data[i].max() > maxseqval:
            maxseqval = data[i].max()

    # 对基础特征进行 embedding
    bise_feature = [SparseFeat('cate_id', vocabulary_size=maxseqval + 1, embedding_dim=parameter['bise_embedding_dim'][1]), ]

    # 用户历史行为序列长度
    behavior_length = []
    for index, row in data.iterrows():
        lens = 0
        for i in row.values[:-2]:
            if i != 0:
                lens += 1
        behavior_length.append(lens)
    data["seq_length"] = behavior_length
    data = data[hist + ['seq_length', 'cateID', 'label']]

    # 历史行为序列embedding
    behavior_feature = [VarLenSparseFeat(
        SparseFeat('hist_cate_id', vocabulary_size=maxseqval + 1,
                   embedding_dim=parameter['hist_embedding_dim'][0], embedding_name='cate_id'), maxlen=maxlen, length_name="seq_length")]

    # 指定历史行为序列对应的特征
    behavior_feature_list = ['cate_id']

    tmp_X, test_X, tmp_y, test_y = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.05,
                                                    random_state=42, stratify=data.iloc[:, -1])
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.05, random_state=42, stratify=tmp_y)

    feature_columns = bise_feature + behavior_feature
    feature_names = get_feature_names(feature_columns)
    print(' feature_names: ', feature_names)

    # # # 负采样：neg_hist_item_id 包含了每一个用户的负采样序列，
    # # # 实际环境中对于CTR数据肯定负样本要远大于正样本，这里负采样序列的意思是：对于每个用户分组，负采样其中的行为序列。
    # # # 这里方便演示，用行为序列的数据。
    # feature_dict['neg_hist_item_id'] = feature_dict['hist_item_id']
    # feature_dict['neg_hist_cate_id'] = feature_dict['hist_cate_id']
    # neg_behavior_feature = [VarLenSparseFeat(SparseFeat('neg_hist_item_id', vocabulary_size=feature_dict['neg_hist_item_id'].max() + 1, embedding_dim=parameter['hist_embedding_dim'][0], embedding_name='item_id'), maxlen=maxlen, length_name="seq_length"),
    #     VarLenSparseFeat(SparseFeat('neg_hist_cate_id', feature_dict['neg_hist_cate_id'].max() + 1, embedding_dim=parameter['hist_embedding_dim'][1], embedding_name='cate_id'), maxlen=maxlen, length_name="seq_length")]
    #
    # feature_columns = bise_feature + behavior_feature + neg_behavior_feature
    feature_columns = bise_feature + behavior_feature
    feature_names = get_feature_names(feature_columns)
    print(' feature_names: ', feature_names)

    # feed input
    train_x, train_y = get_feature_dict(train_X, train_y, [hist, 'cateID', 'seq_length'])
    val_x, val_y = get_feature_dict(val_X, val_y, [hist, 'cateID', 'seq_length'])
    test_x, test_y = get_feature_dict(test_X, test_y, [hist, 'cateID', 'seq_length'])

    print(train_x)
    print(train_y)

    print(' 数据处理完成', time.strftime("%H:%M:%S", time.localtime(time.time())))

    # train
    model = DIEN(dnn_feature_columns=feature_columns, history_feature_list=behavior_feature_list,
                 gru_type=parameter['gru_type'],
                 use_negsampling=parameter['use_negsampling'],
                 alpha=parameter['alpha'],
                 dnn_hidden_units=parameter['dnn_hidden_units'],
                 l2_reg_embedding=parameter['l2_reg_embedding'],
                 l2_reg_dnn=parameter['l2_reg_dnn'],
                 dnn_dropout=parameter['dnn_dropout'],
                 dnn_activation=parameter['dnn_activation'],
                 att_activation=parameter['att_activation'],
                 task='binary')

    # model.summary()
    print(' 训练开始 ', time.strftime("%H:%M:%S", time.localtime(time.time())))
    start_time = time.time()

    mNadam = Adam(learning_rate=parameter['lr'], beta_1=0.9, beta_2=0.999, decay=parameter['lr'] / parameter['epochs'])
    model.compile(optimizer=mNadam,
                  loss='binary_crossentropy',
                  metrics=['AUC', 'Precision', 'Recall'])

    history = model.fit(
        train_x, train_y, validation_data=(val_x, val_y),
        batch_size=parameter['batch_size'],
        epochs=parameter['epochs'],
        verbose=2,
    )

    end_time = time.time()
    print(' 训练完成', time.strftime("%H:%M:%S", time.localtime(time.time())))
    print((' 训练运行时间： {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60)))

    # # 保存成yaml文件,用于tf serving在线服务
    # deep.save(settings.save_path.format('DIEN', 'DIENmodel-11-13_serving'), save_format="tf")
    # # 保存成h5文件，用于离线评估
    # save_model(deep, settings.save_path.format('DIEN', 'DIENmodel-11-13.h5'))

    # predict_DIEN(model, sequential_data_path, encoder_paths, feature_names)

    # 训练可视化
    visualization(history, parameter['epochs'], saveflag=True, showflag=False, path=visual, eager_flag=True)

    # 测试集评估
    scores = model.evaluate(test_x, test_y, verbose=2)
    print(' %s: %.4f' % (model.metrics_names[0], scores[0]))
    print(' %s: %.4f' % (model.metrics_names[1], scores[1]))
    print(' %s: %.4f' % (model.metrics_names[2], scores[2]))
    print(' %s: %.4f' % (model.metrics_names[3], scores[3]))
    if scores[2] != 0 and scores[3] != 0:
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
    print(('DIEN 模型训练运行时间： {:.0f}分 {:.0f}秒'.format((end_time_end - start_time_start) // 60,
                                                  (end_time_end - start_time_start) % 60)))
    print(('{:.0f}小时'.format((end_time_end - start_time_start) // 60 / 60)))

if __name__ == '__main__':
    # 关闭eager模式
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    parameter = {
        'bise_embedding_dim': [4, 4],
        'hist_embedding_dim': [8, 8],
        'gru_type': 'AUGRU',
        'use_negsampling': False,
        # 'use_negsampling': True,
        'alpha': 1.0,
        'dnn_hidden_units': (256, 128, 64),
        'att_hidden_size': (64, 16),
        'l2_reg_embedding': 0.01,
        'dnn_activation': 'relu',
        'l2_reg_dnn': 0.01,
        'dnn_dropout': 0.25,
        'att_activation': "dice",
        'lr': 1e-4,
        'batch_size': 2000,
        'epochs': 1000,
    }
    encoder_paths = [encoder_path.format('item_id_enc'), encoder_path.format('cate_id_enc'),
                     encoder_path.format('hist_item_id_dict'), encoder_path.format('hist_cate_id_dict')]
    # train_DIEN(parameter, sequential_data_path, encoder_paths, visual.format('DIEN'))
    train_DIEN2(parameter, read_path_amazon, encoder_paths, visual.format('DIEN'))





