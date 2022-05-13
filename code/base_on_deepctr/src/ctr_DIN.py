# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2021/12/27 2:53 下午
# @File: ctr_DIN.py
'''
DIN
'''
import time

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from sklearn.model_selection import train_test_split
import pickle

from tools import *
from settings import *

def predict_DIN(model, sequential_data_path, encoder_paths, feature_names):
    feature_dict, label, maxlen = deal_with_sequential(sequential_data_path)
    tmp = ['item_id', 'cate_id', 'hist_item_id', 'hist_cate_id']

    for i in range(0, len(encoder_paths)//2):
        with open(encoder_paths[i], 'rb') as f:
            lab = pickle.load(f)
        feature_dict[tmp[i]] = lab.transform(feature_dict[tmp[i]]) + np.ones(len(feature_dict[tmp[i]]), dtype=int)

    for i in range(len(encoder_paths)//2, len(encoder_paths)):
        with open(encoder_paths[i], 'rb') as f:
            ldict = pickle.load(f)
            feature_dict[tmp[i]] = encDict_map(feature_dict[tmp[i]], ldict)

    x = {name: feature_dict[name] for name in feature_names}
    y = label

    y_pre = model.predict(x, batch_size=30)

'''   基于 sample_data.txt 实现的，目的是快速搭建   '''
def train_DIN(parameter, sequential_data_path, encoder_paths, visual):
    print('DIN 模型训练开始 ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    start_time_start = time.time()

    feature_dict, label, maxlen = deal_with_sequential(sequential_data_path)

    # encoder
    lab_item_id, feature_dict['item_id'] = labEncoder_map(feature_dict['item_id'], save_flag=True, save_path=encoder_paths[0])
    lab_cate_id, feature_dict['cate_id'] = labEncoder_map(feature_dict['cate_id'], save_flag=True, save_path=encoder_paths[1])
    _, feature_dict['hist_item_id'] = labEncoder_map_seq(lab_item_id, feature_dict['hist_item_id'], save_flag=True, save_path=encoder_paths[2])
    _, feature_dict['hist_cate_id'] = labEncoder_map_seq(lab_cate_id, feature_dict['hist_cate_id'], save_flag=True, save_path=encoder_paths[3])

    # 对基础特征进行 embedding
    bise_feature = [SparseFeat('item_id', vocabulary_size=len(np.unique(feature_dict['item_id'])), embedding_dim=parameter['bise_embedding_dim'][0]),
                    SparseFeat('cate_id', vocabulary_size=len(np.unique(feature_dict['cate_id'])), embedding_dim=parameter['bise_embedding_dim'][1]),]

    # _, hist_ad_goods_vocsize, _ = duplicate_seq(feature_dict['hist_item_id'])
    # _, hist_ad_class_vocsize, _ = duplicate_seq(feature_dict['hist_cate_id'])

    # 历史行为序列embedding
    behavior_feature = [VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=feature_dict['hist_item_id'].max()+1, embedding_dim=parameter['hist_embedding_dim'][0], embedding_name='item_id'), maxlen=maxlen),
        VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=feature_dict['hist_cate_id'].max()+1, embedding_dim=parameter['hist_embedding_dim'][0], embedding_name='cate_id'), maxlen=maxlen)]

    # 指定历史行为序列对应的特征
    behavior_feature_list = ['item_id', 'cate_id']

    feature_columns = bise_feature + behavior_feature
    feature_names = get_feature_names(feature_columns)
    print(' feature_names: ', feature_names)

    # feed input
    x = {name: feature_dict[name] for name in feature_names}
    y = label

    print(x)
    print(y)

    print(' 数据处理完成', time.strftime("%H:%M:%S", time.localtime(time.time())))

    # train
    model = DIN(dnn_feature_columns=feature_columns, history_feature_list=behavior_feature_list,
               dnn_hidden_units=parameter['dnn_hidden_units'],
               att_hidden_size=parameter['att_hidden_size'],
               l2_reg_embedding=parameter['l2_reg_embedding'],
               l2_reg_dnn=parameter['l2_reg_dnn'],
               dnn_dropout=parameter['dnn_dropout'],
               dnn_activation=parameter['dnn_activation'],
               att_activation=parameter['att_activation'],
               dnn_use_bn=True, task='binary')

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
    # deep.save(settings.save_path.format('DIN', 'DINmodel-11-13_serving'), save_format="tf")
    # # 保存成h5文件，用于离线评估
    # save_model(deep, settings.save_path.format('DIN', 'DINmodel-11-13.h5'))

    # predict_DIN(model, sequential_data_path, encoder_paths, feature_names)

    # 训练可视化
    visualization(history, parameter['epochs'], saveflag=True, showflag=False, path=visual)

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
    print(('DIN 模型训练运行时间： {:.0f}分 {:.0f}秒'.format((end_time_end - start_time_start) // 60,
                                                  (end_time_end - start_time_start) % 60)))
    print(('{:.0f}小时'.format((end_time_end - start_time_start) // 60 / 60)))

'''   和tf2的DIN用的数据集一样，amazon-books-100k-preprocessed.csv'''
def train_DIN2(parameter, sequential_data_path, encoder_paths, visual):
    print('DIN 模型训练开始 ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    start_time_start = time.time()

    data = pd.read_csv(sequential_data_path)

    # # 新增两列离散数据，分别模拟用户画像特征、上下文特征
    # tmp = data.shape[0]
    # discrete_1 = np.random.randint(1, 100, (tmp,))
    # discrete_2 = np.random.randint(1, 100, (tmp,))
    # data['user_profile'] = discrete_1
    # data['context'] = discrete_2
    #
    # # 重置列的顺序，前两列为新增的假数据，中间为行为序列，最后为label
    hist = []
    for i in range(40):
        hist.append('hist_cate_{}'.format(i))
    # data = data[['user_profile', 'context'] + hist + ['cateID', 'label']]

    # print(data.head(10))

    # 序列最大长度
    maxlen = 40
    # 序列emb词典最大值
    maxseqval = 0
    for i in hist:
        if data[i].max() > maxseqval:
            maxseqval = data[i].max()

    # 对基础特征进行 embedding
    bise_feature = [
        # SparseFeat('user_profile', vocabulary_size=data['user_profile'].max() + 1, embedding_dim=parameter['bise_embedding_dim'][0]),
        # SparseFeat('context', vocabulary_size=data['context'].max() + 1, embedding_dim=parameter['bise_embedding_dim'][1]),
        SparseFeat('cate_id', vocabulary_size=maxseqval + 1, embedding_dim=parameter['bise_embedding_dim'][1]),]

    # 历史行为序列embedding
    behavior_feature = [VarLenSparseFeat(
        SparseFeat('hist_cate_id', vocabulary_size=maxseqval + 1,
                   embedding_dim=parameter['hist_embedding_dim'][0], embedding_name='cate_id'), maxlen=maxlen)]

    # 指定历史行为序列对应的特征
    behavior_feature_list = ['cate_id']

    tmp_X, test_X, tmp_y, test_y = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.05, random_state=42, stratify=data.iloc[:, -1])
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.05, random_state=42, stratify=tmp_y)

    feature_columns = bise_feature + behavior_feature
    feature_names = get_feature_names(feature_columns)
    print(' feature_names: ', feature_names)

    # feed input
    # train_x, train_y = get_feature_dict(train_X, train_y, ['user_profile', 'context', hist, 'cateID'])
    # val_x, val_y = get_feature_dict(val_X, val_y, ['user_profile', 'context', hist, 'cateID'])
    # test_x, test_y = get_feature_dict(test_X, test_y, ['user_profile', 'context', hist, 'cateID'])
    train_x, train_y = get_feature_dict(train_X, train_y, [hist, 'cateID'], is_flag=True)
    val_x, val_y = get_feature_dict(val_X, val_y, [hist, 'cateID'], is_flag=True)
    test_x, test_y = get_feature_dict(test_X, test_y, [hist, 'cateID'], is_flag=True)

    print(' 数据处理完成', time.strftime("%H:%M:%S", time.localtime(time.time())))

    # train
    model = DIN(dnn_feature_columns=feature_columns, history_feature_list=behavior_feature_list,
                dnn_hidden_units=parameter['dnn_hidden_units'],
                att_hidden_size=parameter['att_hidden_size'],
                l2_reg_embedding=parameter['l2_reg_embedding'],
                l2_reg_dnn=parameter['l2_reg_dnn'],
                dnn_dropout=parameter['dnn_dropout'],
                dnn_activation=parameter['dnn_activation'],
                att_activation=parameter['att_activation'],
                dnn_use_bn=True, task='binary')

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
    # deep.save(settings.save_path.format('DIN', 'DINmodel-11-13_serving'), save_format="tf")
    # # 保存成h5文件，用于离线评估
    # save_model(deep, settings.save_path.format('DIN', 'DINmodel-11-13.h5'))

    # predict_DIN(model, sequential_data_path, encoder_paths, feature_names)

    # 训练可视化
    visualization(history, parameter['epochs'], saveflag=True, showflag=False, path=visual)

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
    print(('DIN 模型训练运行时间： {:.0f}分 {:.0f}秒'.format((end_time_end - start_time_start) // 60,
                                                  (end_time_end - start_time_start) % 60)))
    print(('{:.0f}小时'.format((end_time_end - start_time_start) // 60 / 60)))

if __name__ == '__main__':
    parameter = {
        'bise_embedding_dim': [4, 4],
        'hist_embedding_dim': [8, 8],
        'dnn_hidden_units': (128, 128, 64),
        'att_hidden_size': (80, 40),
        'l2_reg_embedding': 0.01,
        'l2_reg_dnn': 0.01,
        'dnn_dropout': 0.25,
        'dnn_activation': 'relu',
        'att_activation': "dice",
        'lr': 1e-4,
        'batch_size': 2000,
        'epochs': 4000,
    }
    encoder_paths = [encoder_path.format('item_id_enc'), encoder_path.format('cate_id_enc'),
                     encoder_path.format('hist_item_id_dict'), encoder_path.format('hist_cate_id_dict')]
    # train_DIN(parameter, sequential_data_path, encoder_paths, visual.format('DIN'))
    train_DIN2(parameter, read_path_amazon, encoder_paths, visual.format('DIN'))







