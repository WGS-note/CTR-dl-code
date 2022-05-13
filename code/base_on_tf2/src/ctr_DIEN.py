# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/1/5 11:07 上午
# @File: ctr_DIEN.py
'''
DIEN
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm

from tools import *

'''   负采样生成辅助的样本   '''
def auxiliary_sample(data):
    '''
    在DIEN算法中，兴趣抽取层在用GRU来抽取interest的同时，为了使得抽取的interest的表示更加合理，
    该层设计了一个二分类的模型来计算兴趣抽取的准确性，
    用用户的下一时刻真实的行为作为positive sample,
    负采样得到的行为作为negative sample来计算一个辅助的loss
    Parameters
    :param data: pandas-df
    :return: neg_sample : negative samples   numpy.ndarray
    '''
    cate_max = np.max(data.iloc[:, 2:-1].values)  # 获取所有行为
    pos_sample = data.iloc[:, 3:-1].values  # 去掉最后的cateID和无下一时刻的第一列
    neg_sample = np.zeros_like(pos_sample)

    for i in range(pos_sample.shape[0]):
        for j in range(pos_sample.shape[1]):
            if pos_sample[i, j] > 0:
                idx = np.random.randint(low=1, high=cate_max + 1)
                while idx == pos_sample[i, j]:
                    idx = np.random.randint(low=1, high=cate_max + 1)
                neg_sample[i, j] = idx
            else:
                break  # 后面的行为都是padding的0

    return neg_sample

'''   Dice   '''
class Dice(layers.Layer):

    def __init__(self):
        super(Dice, self).__init__()
        self.epsilon = 1e-9

    def build(self, input_shape):
        self.alpha = self.add_weight(name='dice_alpha',
                                     shape=(),
                                     initializer=tf.keras.initializers.Zeros(),
                                     trainable=True)

    def call(self, x):
        top = x - tf.reduce_mean(x, axis=0)
        bottom = tf.sqrt(tf.math.reduce_variance(x, axis=0) + self.epsilon)
        norm_x = top / bottom
        p = tf.sigmoid(norm_x)
        x = self.alpha * x * (1 - p) + x * p
        return x

'''   Embedding Layer   '''
class EmbeddingLayer(layers.Layer):

    def __init__(self, seq_size, seq_dim, feature_columns):
        super(EmbeddingLayer, self).__init__()
        self.feature_dim = seq_size
        self.embed_dim = seq_dim

        # self.embedding = layers.Embedding(feature_dim + 1, embed_dim, mask_zero=True, name='item_emb')
        # 行为特征、queryId embedding
        # mask_True=True：输入矩阵中的0会被mask掉，即0为padding的值，要忽略这些的影响
        self.seq_emb = layers.Embedding(seq_size + 1, seq_dim, mask_zero=True, name='item_emb')

        # 除序列特征之外的其它特征：如画像特征、上下文信息等
        self.feature_columns = feature_columns
        # 其它特征embedding
        self.feature_emb = {'emb_{}'.format(i): layers.Embedding(feat['vocabulary_size'], feat['embed_dim'])
                            for i, feat in enumerate(self.feature_columns)}

    def call(self, x, neg_x=None):
        """
            input :
                x : (behaviors * 40, ads * 1) -> batch * (behaviors + ads) -> (2000, 40)
                neg_x : (behavior * 39) -> batch * behavior -> (2000, 39)
            output :
                query_ad : (batch * 1 * embed_dim) -> (2000, 1, 4)
                user_behavior : (batch * Time_seq_len * embed_dim) -> (2000, 40, 4)
                mask : (2000, 40, 1)
                neg_user_behavior：(2000, 39, 4)
                neg_mask：(2000, 39, 1)
                profile_context：(2000, 16)
        """
        # User Profile、Context Feature
        profile_context = x[:, :2]  # (2000, 2)
        # User Behaviors
        behaviors_x = x[:, 2:-1]  # (2000, 40)
        # Candidate Ad
        ads_x = x[:, -1]  # (2000,)

        # embedding
        profile_context = tf.concat([self.feature_emb['emb_{}'.format(i)](profile_context[:, i])
                                     for i in range(len(self.feature_columns))], axis=1)  # (2000, 4)
        query_ad = tf.expand_dims(self.seq_emb(ads_x), axis=1)  # (2000, 1, 4)  因为行为序列emb后是3维，所以这里增加一维
        user_behavior = self.seq_emb(behaviors_x)  # (2000, 40, 4)

        # 定义mask
        mask = tf.cast(behaviors_x > 0, tf.float32)  # (2000, 40)
        mask = tf.expand_dims(mask, axis=-1)   # (2000, 40, 1)

        if neg_x is not None:
            neg_mask = tf.cast(neg_x > 0, tf.float32)
            neg_mask = tf.expand_dims(neg_mask, axis=-1)
            neg_user_behavior = self.seq_emb(neg_x)

            return query_ad, user_behavior, mask, \
                   neg_user_behavior, neg_mask, profile_context

        return query_ad, user_behavior, mask, profile_context

'''   兴趣抽取层   '''
class InterestExtractLayer(Model):

    def __init__(self, embed_dim, extract_fc_dims, extract_dropout):
        super(InterestExtractLayer, self).__init__()

        # 传统的GRU来抽取时序行为的兴趣表示  return_sequences=True: 返回上次的输出
        self.GRU = layers.GRU(units=embed_dim, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)

        # 用一个mlp来计算 auxiliary loss
        self.auxiliary_mlp = tf.keras.Sequential()
        for fc_dim in extract_fc_dims:
            self.auxiliary_mlp.add(layers.Dense(fc_dim))
            self.auxiliary_mlp.add(layers.Activation('relu'))
            self.auxiliary_mlp.add(layers.Dropout(extract_dropout))
        self.auxiliary_mlp.add(layers.Dense(1))

    def call(self, user_behavior, mask, neg_user_behavior=None, neg_mask=None):
        """
            user_behavior : (2000, 40, 4)
            mask : (2000, 40, 1)
            neg_user_behavior : (2000, 39, 4)
            neg_mask : (2000, 39, 1)
        """
        # 将0-1遮罩变换bool
        mask_bool = tf.cast(tf.squeeze(mask, axis=2), tf.bool)  # (2000, 40)

        gru_interests = self.GRU(user_behavior, mask=mask_bool)  # (2000, 40, 4)

        # 计算Auxiliary Loss，只在负采样的时候计算 aux loss
        if neg_user_behavior is not None:
            # 此处用户真实行为user_behavior为图中的e，GRU抽取的状态为图中的h
            gru_embed = gru_interests[:, 1:]  # (2000, 39, 4)
            neg_mask_bool = tf.cast(tf.squeeze(neg_mask, axis=2), tf.bool)  # (2000, 39)

            # 正样本的构建  选取下一个行为作为正样本
            pos_seq = tf.concat([gru_embed, user_behavior[:, 1:]], -1)  # (2000, 39, 8)
            pos_res = self.auxiliary_mlp(pos_seq)  # (2000, 39, 1)
            pos_res = tf.sigmoid(pos_res[neg_mask_bool])  # 选择不为0的进行sigmoid  (N, 1) ex: (18290, 1)
            pos_target = tf.ones_like(pos_res, tf.float16)  # label

            # 负样本的构建  从未点击的样本中选取一个作为负样本
            neg_seq = tf.concat([gru_embed, neg_user_behavior], -1)  # (2000, 39, 8)
            neg_res = self.auxiliary_mlp(neg_seq)  # (2000, 39, 1)
            neg_res = tf.sigmoid(neg_res[neg_mask_bool])
            neg_target = tf.zeros_like(neg_res, tf.float16)

            # 计算辅助损失 二分类交叉熵
            aux_loss = tf.keras.losses.binary_crossentropy(tf.concat([pos_res, neg_res], axis=0), tf.concat([pos_target, neg_target], axis=0))
            aux_loss = tf.cast(aux_loss, tf.float32)
            aux_loss = tf.reduce_mean(aux_loss)

            return gru_interests, aux_loss

        return gru_interests, 0

'''   Activation Unit   '''
class ActivationUnit(Model):

    def __init__(self, embed_dim, att_dropout=0.2, att_fc_dims=[32, 16]):
        super(ActivationUnit, self).__init__()

        # self.fc_layers = tf.keras.Sequential()
        # input_dim = embed_dim * 4
        # for fc_dim in att_fc_dims:
        #     self.fc_layers.add(layers.Dense(fc_dim, input_shape=[input_dim, ]))
        #     self.fc_layers.add(Dice())
        #     self.fc_layers.add(layers.Dropout(att_dropout))
        #     self.input_dim = fc_dim
        # self.fc_layers.add(layers.Dense(1, input_shape=[input_dim, ]))

        self.fc_layers = tf.keras.Sequential()
        for fc_dim in att_fc_dims:
            self.fc_layers.add(layers.Dense(fc_dim))
            self.fc_layers.add(Dice())
            self.fc_layers.add(layers.Dropout(att_dropout))
        self.fc_layers.add(layers.Dense(1))

    def call(self, query, user_behavior):
        """
            query : 单独的ad的embedding mat -> batch * 1 * embed
            user_behavior : 行为特征矩阵 -> batch * seq_len * embed
        """

        # repeat ads
        seq_len = user_behavior.shape[1]
        queries = tf.concat([query] * seq_len, axis=1)
        attn_input = tf.concat([queries,
                                user_behavior,
                                queries - user_behavior,
                                queries * user_behavior], axis=-1)
        out = self.fc_layers(attn_input)
        return out

'''   Attention Pooling Layer   '''
class AttentionPoolingLayer(Model):

    def __init__(self, embed_dim, att_dropout=0.2, att_fc_dims=[32, 16], return_score=False):
        super(AttentionPoolingLayer, self).__init__()

        self.active_unit = ActivationUnit(embed_dim, att_dropout, att_fc_dims)
        self.return_score = return_score

    def call(self, query, user_behavior, mask):
        """
            query_ad : 单独的ad的embedding mat -> batch * 1 * embed
            user_behavior : 行为特征矩阵 -> batch * seq_len * embed
            mask : 被padding为0的行为置为false -> batch * seq_len * 1
        """

        # attn weights
        attn_weights = self.active_unit(query, user_behavior)
        # mul weights and sum pooling
        if not self.return_score:
            output = user_behavior * attn_weights * mask
            return output

        return attn_weights

'''   AGRU单元   '''
class AGRUCell(layers.Layer):
    """
        Attention based GRU (AGRU)
        公式如下:
            r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)
            #z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)
            h' = tanh(W_ih * x + b_ih + r * (W_hh * h + b_hh))
            h = (1 - att_score) * h + att_score * h'

    """
    def __init__(self, units):
        super().__init__()
        self.units = units
        # 作为一个 RNN 的单元，必须有state_size属性
        # state_size 表示每个时间步输出的维度
        self.state_size = units

    def build(self, input_shape):
        # 输入数据是一个tupe: (gru_embed, atten_scores)  (2000, 4)、(2000, 1)
        # 因此，t时刻输入的维度为：
        dim_xt = input_shape[0][-1]

        # 重置门中的参数
        self.w_ir = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_ir')
        self.w_hr = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='w_hr')
        self.b_ir = tf.Variable(tf.random.normal(shape=[self.units]), name='b_ir')
        self.b_hr = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hr')
        # 更新门被att_score代替

        # 候选隐藏中的参数
        self.w_ih = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_ih')
        self.w_hh = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='w_hh')
        self.b_ih = tf.Variable(tf.random.normal(shape=[self.units]), name='b_ih')
        self.b_hh = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hh')

    def call(self, inputs, states):
        x_t, att_score = inputs
        states = states[0]

        # 重置门
        r_t = tf.sigmoid(tf.matmul(x_t, self.w_ir) + self.b_ir + \
                         tf.matmul(states, self.w_hr) + self.b_hr)

        # 候选隐藏状态
        h_t_ = tf.tanh(tf.matmul(x_t, self.w_ih) + self.b_ih + \
                       tf.multiply(r_t, (tf.matmul(states, self.w_hh) + self.b_hh)))
        # 输出值
        h_t = tf.multiply(1 - att_score, states) + tf.multiply(att_score, h_t_)
        # 对gru而言，当前时刻的output与传递给下一时刻的state相同
        next_state = h_t

        return h_t, next_state

'''   AUGRU单元   '''
class AUGRUCell(layers.Layer):
    """
        GRU with attentional update gate (AUGRU)
        公式如下:
            r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)
            z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)
            z = z * att_score
            h' = tanh(W_ih * x + b_ih + r * (W_hh * h + b_hh))
            h = (1 - z) * h + z * h'

    """
    def __init__(self, units):
        super().__init__()
        self.units = units
        # 作为一个 RNN 的单元，必须有state_size属性
        # state_size 表示每个时间步输出的维度
        self.state_size = units

    def build(self, input_shape):
        # 输入数据是一个tupe: (gru_embed, atten_scores)
        # 因此，t时刻输入的维度为：
        dim_xt = input_shape[0][-1]

        # 重置门中的参数
        self.w_ir = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_ir')
        self.w_hr = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='w_hr')
        self.b_ir = tf.Variable(tf.random.normal(shape=[self.units]), name='b_ir')
        self.b_hr = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hr')

        # 更新门中的参数
        self.w_iz = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_iz')
        self.w_hz = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='W_hz')
        self.b_iz = tf.Variable(tf.random.normal(shape=[self.units]), name='b_iz')
        self.b_hz = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hz')

        # 候选隐藏中的参数
        self.w_ih = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_ih')
        self.w_hh = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='w_hh')
        self.b_ih = tf.Variable(tf.random.normal(shape=[self.units]), name='b_ih')
        self.b_hh = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hh')

    def call(self, inputs, states):
        x_t, att_score = inputs
        states = states[0]
        # 重置门
        r_t = tf.sigmoid(tf.matmul(x_t, self.w_ir) + self.b_ir + \
                         tf.matmul(states, self.w_hr) + self.b_hr)
        # 更新门
        z_t = tf.sigmoid(tf.matmul(x_t, self.w_iz) + self.b_iz + \
                         tf.matmul(states, self.w_hz) + self.b_hz)
        # 带有注意力的更新门
        z_t = tf.multiply(att_score, z_t)
        # 候选隐藏状态
        h_t_ = tf.tanh(tf.matmul(x_t, self.w_ih) + self.b_ih + \
                       tf.multiply(r_t, (tf.matmul(states, self.w_hh) + self.b_hh)))
        # 输出值
        h_t = tf.multiply(1 - z_t, states) + tf.multiply(z_t, h_t_)
        # 对gru而言，当前时刻的output与传递给下一时刻的state相同
        next_state = h_t

        return h_t, next_state

'''   兴趣进化层   '''
class InterestEvolutionLayer(Model):

    def __init__(self, input_size, gru_type='AUGRU', evolution_dropout=0.2, att_dropout=0.2, att_fc_dims=[32, 16]):
        super(InterestEvolutionLayer, self).__init__()
        self.gru_type = gru_type
        self.dropout = evolution_dropout

        if gru_type == 'GRU':
            self.attention = AttentionPoolingLayer(embed_dim=input_size, att_dropout=att_dropout, att_fc_dims=att_fc_dims)

            self.evolution = layers.GRU(units=input_size,
                                        return_sequences=True)
        elif gru_type == 'AIGRU':
            self.attention = AttentionPoolingLayer(embed_dim=input_size,
                                                   att_dropout=att_dropout,
                                                   att_fc_dims=att_fc_dims,
                                                   return_score=True)
            self.evolution = layers.GRU(units=input_size)
        elif gru_type == 'AGRU':
            self.attention = AttentionPoolingLayer(embed_dim=input_size,
                                                   att_dropout=att_dropout,
                                                   att_fc_dims=att_fc_dims,
                                                   return_score=True)
            self.evolution = layers.RNN(AGRUCell(units=input_size))
        elif gru_type == 'AUGRU':
            self.attention = AttentionPoolingLayer(embed_dim=input_size,
                                                   att_dropout=att_dropout,
                                                   att_fc_dims=att_fc_dims,
                                                   return_score=True)
            self.evolution = layers.RNN(AUGRUCell(units=input_size))

    def call(self, query_ad, gru_interests, mask):
        """
            query_ad : B * 1 * E -> (2000, 1, 4)
            gru_interests : B * T * H -> (2000, 40, 4)
            mask : B * T * 1 -> (2000, 40, 1)
        """
        mask_bool = tf.cast(tf.squeeze(mask, axis=2), tf.bool)  # (2000, 40)

        if self.gru_type == 'GRU':
            # GRU后接attention
            out = self.evolution(gru_interests, mask=mask_bool)  # (2000, 40, 4)
            out = self.attention(query_ad, out, mask)  # (2000, 40, 4)
            out = tf.reduce_sum(out, axis=1)  # (2000, 4)
        elif self.gru_type == 'AIGRU':
            # AIGRU
            att_score = self.attention(query_ad, gru_interests, mask)  # (2000, 40, 1)
            out = att_score * gru_interests  # (2000, 40, 4)
            out = self.evolution(out, mask=mask_bool)  # (2000, 4)
        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            # AGRU or AUGRU
            att_score = self.attention(query_ad, gru_interests, mask)  # (2000, 40, 1)
            out = self.evolution((gru_interests, att_score), mask=mask_bool)  # (2000, 4)

        return out

'''   DIEN   '''
class DIEN(Model):

    def __init__(self, seq_size, seq_dim, feature_columns, mlp_dims, gru_type='AUGRU',
                 extract_fc_dims=[100, 50], extract_dropout=0.,
                 evolution_dropout=0.2, att_dropout=0.2, att_fc_dims=[32, 16]):
        super(DIEN, self).__init__()
        self.feature_dim = seq_size
        self.embed_dim = seq_dim
        self.gru_type = gru_type

        # Embedding Layer
        self.embedding = EmbeddingLayer(seq_size, seq_dim, feature_columns)

        # Interest Extract Layer
        self.interest_extract = InterestExtractLayer(embed_dim=seq_dim, extract_fc_dims=extract_fc_dims, extract_dropout=extract_dropout)

        # Interest Evolution Layer
        self.interest_evolution = InterestEvolutionLayer(input_size=seq_dim,
                                                         gru_type=gru_type,
                                                         evolution_dropout=evolution_dropout,
                                                         att_dropout=att_dropout,
                                                         att_fc_dims=att_fc_dims)
        # 最后的MLP层预测
        self.final_mlp = tf.keras.Sequential()
        for fc_dim in mlp_dims:
            self.final_mlp.add(layers.Dense(fc_dim))
            self.final_mlp.add(layers.Activation('relu'))
            self.final_mlp.add(layers.Dropout(evolution_dropout))
        self.final_mlp.add(layers.Dense(1))

    def call(self, x, neg_x=None):
        """
            x : (behaviors * 40, ads * 1, other_feature*2) -> batch * (behaviors + ads + other_feature) -> (2000, 43)
            neg_x : (behaviors * 39) -> batch * (behaviors + ads) -> (2000, 39)
        """
        # Embedding   只有行为序列参与兴趣抽取和兴趣进化，其它特征embedding和兴趣concat
        _ = self.embedding(x, neg_x)
        if neg_x is not None:
            query_ad, user_behavior, mask, neg_user_behavior, neg_mask, profile_context = _
        else:
            query_ad, user_behavior, mask, profile_context = _
            neg_user_behavior = None
            neg_mask = None

        # Interest Extraction  兴趣抽取层
        gru_interest, aux_loss = self.interest_extract(user_behavior,
                                                       mask,
                                                       neg_user_behavior,
                                                       neg_mask)

        # Interest Evolution  兴趣进化层
        final_interest = self.interest_evolution(query_ad, gru_interest, mask)

        # MLP for prediction
        concat_out = tf.concat([tf.squeeze(query_ad, 1),
                                final_interest,
                                profile_context], axis=1)

        out = self.final_mlp(concat_out)
        out = tf.sigmoid(out)

        return out, aux_loss


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    with open('config.yaml', 'r') as f:
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

    fields = data_X.max().max()
    print('fields: ', fields)

    other_spare_emb = [sparseFeature(feat, data_X[feat].max() + 1, config['DIEN']['fea_dim']) for feat in ['user_profile', 'context']]

    tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size=0.2, random_state=42, stratify=data_y)
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.1, random_state=42, stratify=tmp_y)

    train_X_neg = auxiliary_sample(train_X)
    train_X = train_X.values
    val_X = val_X.values
    test_X = test_X.values

    train_loader = tf.data.Dataset.from_tensor_slices((train_X, train_X_neg, train_y)).shuffle(len(train_X)).batch(config['train']['batch_size'])
    val_loader = tf.data.Dataset.from_tensor_slices((val_X, val_y)).batch(config['train']['batch_size'])

    model = DIEN(seq_size=fields, seq_dim=config['DIEN']['seq_dim'], feature_columns=other_spare_emb,
                 mlp_dims=config['DIEN']['mlp_dims'], gru_type=config['DIEN']['gru_type'],
                 extract_fc_dims=config['DIEN']['extract_fc_dims'], extract_dropout=config['DIEN']['extract_dropout'],
                 evolution_dropout=config['DIEN']['evolution_dropout'], att_dropout=config['DIEN']['att_dropout'], att_fc_dims=config['DIEN']['att_fc_dims'])
    adam = optimizers.Adam(lr=0.001, beta_1=0.95, beta_2=0.96, decay=config['train']['adam_lr'] / config['train']['epochs'])

    epoches = config['train']['epochs']
    for epoch in range(epoches):
        epoch_train_loss = tf.keras.metrics.Mean()
        # m_auc = tf.keras.metrics.AUC()

        for batch, (x, neg_x, y) in tqdm(enumerate(train_loader)):
            with tf.GradientTape() as tape:
                out, aux_loss = model(x, neg_x)
                loss = tf.keras.losses.binary_crossentropy(y, out)
                # loss_target = loss + α * aux_loss
                loss = tf.reduce_mean(loss) + config['DIEN']['alpha_aux_loss'] * aux_loss
                loss = tf.reduce_mean(loss)
                # m_auc.update_state(y, out)

            grads = tape.gradient(loss, model.trainable_variables)
            adam.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
            epoch_train_loss(loss)
            # print('---', m_auc.result().numpy())

        epoch_val_loss = tf.keras.metrics.Mean()
        for batch, (x, y) in tqdm(enumerate(val_loader)):
            out, _ = model(x)
            loss = tf.keras.losses.binary_crossentropy(y, out)
            loss = tf.reduce_mean(loss)
            epoch_val_loss(loss)
        print('EPOCH : %s, train loss : %s, val loss: %s' % (epoch,
                                                             epoch_train_loss.result().numpy(),
                                                             epoch_val_loss.result().numpy()))

    model.summary()






