# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/1/5 5:00 下午
# @File: ctr_DIN.py
'''
DIN
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from sklearn.model_selection import train_test_split
import yaml

from tools import *

'''   Dice   '''
class Dice(layers.Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.epsilon = 1e-9

    def build(self, input_shape):
        self.apha = self.add_weight(name='dice_alpha', initializer=initializers.Zeros(), trainable=True)

    def call(self, inputs, **kwargs):
        top = inputs - tf.reduce_mean(inputs, axis=0)
        bottom = tf.sqrt(tf.math.reduce_variance(inputs, axis=0) + self.epsilon)
        z = top / bottom
        # p = tf.sigmoid(z)
        p = tf.sigmoid(-z)
        return p * inputs + (1 - p) * self.apha * inputs  # (None, 32)

'''   Activation Unit   '''
class ActivationUnit(Model):
    """
        Activation Unit结构
        + 输入
          - 用户行为作为key；候选 item embedding，作为query；
        + Out Product层：
          - 计算矩阵之间的对应元素相乘；
        + Concat层：
          - 将 query、key、query-key、query、key(Out Product层，element-wise 乘法) 的结果进行拼接；
        + Dense层：
          - 全连接层，并以PRelu或Dice作为激活函数；
        + Linear层（输出层）：
          - 全连接层，输出单元为1，即得到（query, key）相应的权重值；
    """

    def __init__(self, att_dropout, att_fc_dims):
        super(ActivationUnit, self).__init__()
        self.fc_layers = tf.keras.Sequential()
        for dim in att_fc_dims:
            self.fc_layers.add(layers.Dense(dim, activation=None))
            self.fc_layers.add(Dice())
            self.fc_layers.add(layers.Dropout(att_dropout))
        self.fc_layers.add(layers.Dense(1))

    def call(self, query, user_behavior):
        '''
        :param query: ad的embedding (2000, 1, 8)
        :param user_behavior: 行为特征 (2000, 40, 8)
        '''
        # 用户行为特征数量
        seq_len = user_behavior.shape[1]  # 40
        # 将query的维度和行为特征维度保持一致
        # [query] * seq_len：这个list有seq_len个query (2000, 1, 8)
        queries = tf.concat([query] * seq_len, axis=1)  # (2000, 40, 8)
        attn_input = tf.concat([queries,
                                user_behavior,
                                queries - user_behavior,
                                queries * user_behavior], axis=-1)  # (2000, 40, 32)

        out = self.fc_layers(attn_input)  # (2000, 40, 1)
        return out

'''   注意力池化层   '''
class AttentionPoolingLayer(Model):
    """
        Attention Pooling Layer
        对用户不同的行为的注意力是不一样的，在生成User embedding的时候，加入了Activation Unit Layer
        这一层产生了每个用户行为的权重乘上相应的物品embedding，从而生产了user interest embedding的表示
    """

    def __init__(self, att_dropout, att_fc_dims):
        super(AttentionPoolingLayer, self).__init__()
        # 注意力单元，输出注意力得分
        self.active_unit = ActivationUnit(att_dropout, att_fc_dims)

    def call(self, query, user_behavior, mask):
        '''
        :param query: ad的embedding (2000, 1, 8)
        :param user_behavior: 行为特征矩阵 (2000, 40, 8)
        :param mask: 0-1矩阵，目的是将原来为0但是emb后不为0的行为重置为0 (2000, 40, 1)
        '''
        # attn weights
        attn_weights = self.active_unit(query, user_behavior)  # (2000, 40, 1)
        # mul weights
        output = user_behavior * attn_weights * mask  # (2000, 40, 8)
        # sum pooling
        output = tf.reduce_sum(output, axis=1)  # (2000, 8)
        return output

'''   DIN   '''
class DIN(Model):
    def __init__(self, seq_size, seq_dim, feature_columns, fc_dims, fc_dropout, att_fc_dims, att_dropout):
        super(DIN, self).__init__()
        # 行为特征、queryId embedding
        self.seq_emb = layers.Embedding(seq_size + 1, seq_dim)
        # 除序列特征之外的其它特征：如画像特征、上下文信息等
        self.feature_columns = feature_columns
        # 其它特征embedding
        self.feature_emb = {'emb_{}'.format(i): layers.Embedding(feat['vocabulary_size'], feat['embed_dim'])
                            for i, feat in enumerate(self.feature_columns)}

        # 注意力池化层
        self.AttentionActivate = AttentionPoolingLayer(att_dropout, att_fc_dims)  # (2000, 8)

        # 最后拼接的全连接层
        self.fc_layers = tf.keras.Sequential()
        for dim in fc_dims:
            self.fc_layers.add(layers.Dense(dim, activation=None))
            # self.fc_layers.add(layers.BatchNormalization())
            # self.fc_layers.add(layers.Activation('relu'))
            self.fc_layers.add(Dice())
            self.fc_layers.add(layers.Dropout(fc_dropout))
        self.fc_layers.add(layers.Dense(1))

    def call(self, inputs, training=None, mask=None):
        # User Profile、Context Feature
        profile_context = inputs[:, :2]  # (2000, 2)
        # User Behaviors
        behavior_x = inputs[:, 2:-1]  # (2000, 40)

        # 将行为序列有值的变为1，0的还是0.  ex：Tensor([12 10 0 0 11] -> Tensor([1. 1. 0. 0. 1.]
        mask = tf.cast(behavior_x > 0, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)  # (2000, 40, 1)

        # Candidate Ad
        ads_x = inputs[:, -1]  # (2000,)

        # embedding
        profile_context = tf.concat([self.feature_emb['emb_{}'.format(i)](profile_context[:, i])
                                     for i in range(len(self.feature_columns))], axis=1)  # (2000, 8)
        query_ad = tf.expand_dims(self.seq_emb(ads_x), axis=1)   # (2000, 1, 8)  因为行为序列emb后是3维，所以这里增加一维。axis=1：每条行为记录对应一个query id
        user_behavior = self.seq_emb(behavior_x)  # (2000, 40, 8)

        # user_behavior = user_behavior * mask  目的：将原来为0，但是emb后不为0的部分重置为0
        user_behavior = tf.multiply(user_behavior, mask)  # (2000, 40, 8)

        # attn pooling
        user_interest = self.AttentionActivate(query_ad, user_behavior, mask)  # (2000, 8)

        # concat feature
        concat_input = tf.concat([user_interest,
                                  tf.squeeze(query_ad, axis=1),
                                  profile_context
                                  ], axis=1)

        # MLPs prediction
        out = self.fc_layers(concat_input)
        out = tf.sigmoid(out)

        return out


if __name__ == '__main__':
    # with open('config.yaml', 'r') as f:
    with open('/ad_ctr/base_on_tf2/src/config.yaml', 'r') as f:
        config = yaml.Loader(f).get_data()

    data = pd.read_csv(config['read_path_amazon'], index_col=0)

    # 新增两列离散数据，分别模拟用户画像特征、上下文特征
    tmp = data.shape[0]
    discrete_1 = np.random.randint(1, 100, (tmp,))
    discrete_2 = np.random.randint(1, 100, (tmp,))
    data['user_profile'] = discrete_1
    data['context'] = discrete_2

    # 重置列的顺序，前两列为新增的假数据，中间为行为序列，最后为label
    hist = []
    for i in range(40):
        hist.append('hist_cate_{}'.format(i))
    data = data[['user_profile', 'context'] + hist + ['cateID', 'label']]

    data_X = data.iloc[:, :-1]
    data_y = data.label.values

    # 获取行为特征embedding的字典词大小
    fields = data_X[hist].max().max()
    print('fields: ', fields)
    # 其它除行为特征的特征
    other_spare_emb = [sparseFeature(feat, data_X[feat].max() + 1, config['DIN']['fea_dim']) for feat in ['user_profile', 'context']]

    tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size=0.2, random_state=42, stratify=data_y)
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.1, random_state=42, stratify=tmp_y)

    model = DIN(seq_size=fields, seq_dim=config['DIN']['seq_dim'],
                feature_columns=other_spare_emb,
                fc_dims=config['DIN']['fc_dims'], fc_dropout=config['DIN']['fc_dropout'],
                att_fc_dims=config['DIN']['att_fc_dims'], att_dropout=config['DIN']['att_dropout'])

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
        verbose=2,
    )

    model.summary()

