# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2021/12/27 2:21 下午
# @File: settings.py

# 1：GPU-docker运行、2：本地窗口运行、3：pycharm右键运行
# runFalg = 1
# runFalg = 2
runFalg = 3
if runFalg == 1:
    # criteo 数据集路径
    criteo_sampled_data_path = '/ad_ctr/data/criteo_sampled_data.csv'
    criteo_name = 'criteo_sampled_data'
    # 训练可视化
    visual = '/ad_ctr/base_on_deepctr/imgs/{}.jpg'
    # 亚马逊数据集，用于DIN、DIEN
    amazon_data_path = '/ad_ctr/data/amazon-books-100k-preprocessed.csv'
    sequential_data_path = '/ad_ctr/data/sample_data.txt'
    sequential_data_path2 = '/ad_ctr/data/222sample_data.txt'
    read_path_amazon = '/ad_ctr/data/amazon-books-100k-preprocessed.csv'
    # 编码路径，用于DIN、DIEN预测
    encoder_path = '/ad_ctr/base_on_deepctr/enc/{}.pkl'
elif runFalg == 2:
    # criteo 数据集路径
    criteo_sampled_data_path = '../data/criteo_sampled_data.csv'
    criteo_name = 'criteo_sampled_data'
    # 训练可视化
    visual = '../base_on_deepctr/imgs/{}.jpg'
    # 亚马逊数据集，用于DIN、DIEN
    amazon_data_path = '../data/amazon-books-100k-preprocessed.csv'
    sequential_data_path = '../data/sample_data.txt'
    sequential_data_path2 = '../data/222sample_data.txt'
    read_path_amazon = '../data/amazon-books-100k-preprocessed.csv'
    # 编码路径，用于DIN、DIEN预测
    encoder_path = '../base_on_deepctr/enc/{}.pkl'
else:
    # criteo 数据集路径
    criteo_sampled_data_path = '../../data/criteo_sampled_data.csv'
    criteo_name = 'criteo_sampled_data'
    # 训练可视化
    visual = '../../base_on_deepctr/imgs/{}.jpg'
    # 亚马逊数据集，用于DIN、DIEN
    amazon_data_path = '../../data/amazon-books-100k-preprocessed.csv'
    sequential_data_path = '../../data/sample_data.txt'
    sequential_data_path2 = '../../data/222sample_data.txt'
    read_path_amazon = '../../data/amazon-books-100k-preprocessed.csv'
    # 编码路径，用于DIN、DIEN预测
    encoder_path = '../../base_on_deepctr/enc/{}.pkl'







