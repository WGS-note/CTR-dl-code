# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2021/12/30 3:27 下午
# @File: offsets_exp.py
import numpy as np, pandas as pd

data = pd.DataFrame({'x0': [0, 1, 0, 1], 'x1': [0, 1, 2, 3], 'x2': [1, 1, 0, 0]})
print(data)

fields = data.max().values + 1
fields = fields.astype(np.int)
print('fields: ', fields)

offsets = np.array((0, *np.cumsum(fields)[:-1]), dtype=np.long)
print('offsets: ', offsets)

print('x: \n', data.values)
print('x + offsets: ')
print(data.values + offsets)