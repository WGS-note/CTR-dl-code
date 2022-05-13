# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/1/10 9:49 下午
# @File: GBDT_LR.py
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, log_loss
from sklearn.preprocessing import OneHotEncoder

from tools import *
from settings import *


data, dense_features, sparse_features = deal_with_criteo(criteo_sampled_data_path, criteo_name)
data = shuffle(data)

# train_data, test_data, val_data = split_val_test(data, n=90)
tmp_X, test_X, tmp_y, test_y = train_test_split(data.iloc[:, 1:], data.iloc[:, 0], test_size=0.05, random_state=42, stratify=data.iloc[:, 0])
train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size=0.05, random_state=42, stratify=tmp_y)

print(len(train_y))
print(len(val_y))
print(len(test_y))

gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.743, max_depth=3, min_samples_leaf=50, min_samples_split=5, min_impurity_decrease=0.2)
# gbdt = GradientBoostingClassifier()
gbdt.fit(train_X, train_y.ravel())

x_train_leaves = gbdt.apply(train_X)[:, :, 0]

ohecodel = OneHotEncoder()
x_train_trans = ohecodel.fit_transform(x_train_leaves)

lr = LogisticRegression(penalty='l2', C=0.8, max_iter=150, tol=1e-5)
lr.fit(x_train_trans, train_y)

x_test_leaves = gbdt.apply(test_X)[:, :, 0]
x_test_trans = ohecodel.transform(x_test_leaves)
y_test_preba = lr.predict_proba(x_test_trans)[:, 1]
y_pre = []
for proba in y_test_preba:
    if proba > 0.5:
        y_pre.append(1)
    else:
        y_pre.append(0)

f1 = f1_score(test_y, y_pre)
auc = roc_auc_score(test_y, y_test_preba)
# fpr, tpr, thresholds = roc_curve(y_test, y_test_pre)
# gbdt_lr_ks = max(tpr - fpr)
acc = accuracy_score(test_y, y_pre)
print(' 测试集-GBDT+LR 损失: %.5f' % (log_loss(test_y, y_pre)))
print(' 测试集-GBDT+LR 精确率: %.5f' % (precision_score(test_y, y_pre)))
print(' 测试集-GBDT+LR 召回率: %.5f' % (recall_score(test_y, y_pre)))
print(' 测试集-GBDT+LR F1: %.5f' % (f1))
print(' 测试集-GBDT+LR AUC: %.5f' % (auc))
print(' 测试集-GBDT+LR 准确率: %.5f' % (acc))
