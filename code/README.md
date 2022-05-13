**数据集介绍**

`./code/data/`

+ criteo广告展示数据集下载地址：https://www.kaggle.com/c/criteo-display-ad-challenge/data
  + criteo_sampled_data：criteo广告展示数据集(原11G数据集需解压criteo-dataset)
  + Label：待预测广告，被点击是1，没有被点击是0
  + I1-I13：总共 13 列数值型特征（主要是计数特征）
  + C1-C26：共有 26 列类别型特征。 出于匿名目的，这些功能的值已散列到 32 位

  + 特征连续型的有13个，类别型的26个

+ 阿里广告数据集下载地址：https://tianchi.aliyun.com/dataset/dataDetail?dataId=56
  + 天池DIN、DIEN数据集处理方式：https://github.com/StephenBo-China/recommendation_system_sort_model/blob/main/data_process/alibaba_data_process.ipynb

+ 亚马逊书数据集：
  + `amazon-books-100k-preprocessed.csv`、`amazon-books-100k.txt`
    + 只保留了商品特征，以及历史上的商品hist的特征
    + 处理文件在数据集同目录下的`AmazonDataPreprocess.py`里
    + 原始TXT数据为：`amazon-books-100k.txt`
  + sample_data.txt
    + 用户历史点击商品序列、用户历史点击品类序列、推荐广告商品序列、推荐广告商品品类、点击标记
      以上5项用分号分割；
      用户历史点击商品序列中，商品间用空格隔开；
      用户历史点击品类序列中，品类间用空格隔开；

**目录说明**

+ base_on_deepctr：基于deepctr实现
+ base_on_Paddle：基于飞桨实现
+ base_on_tf2：基于tf2实现



**测试集-结果对比**

> 注 --- 参数统一：
>
> batch_size = 2000
>
> epochs = 1000
>
> embedding dim = 4
>
> l2_reg_linear = 0.01
>
> l2_reg_embedding = 0.01
>
> dnn_dropout = 0.25
>
> lr = 1e-3
>
> dnn_hidden_units = (256, 128, 64)

> 数据集：
>
> criteo_sampled_data.csv
>
> 代码：
>
> base_on_deepctr/*

| Model   | Loss   | AUC    | Precision | Recall | F1     | ACC    | Time  |
| ------- | ------ | ------ | --------- | ------ | ------ | ------ | ----- |
| GBDT+LR |        | 0.7616 | 0.6955    | 0.6840 | 0.6897 | 0.689  | 5分   |
| MLR     | 0.6796 | 0.6813 | 0.6169    | 0.6753 | 0.6448 | 0.6262 | 3小时 |
| FNN     | 0.5988 | 0.7526 | 0.6842    | 0.6808 | 0.6825 | 0.6812 | 40分  |
| WDL     | 0.6034 | 0.7484 | 0.6805    | 0.6835 | 0.6820 | 0.6758 | 39分  |
| DCN     | 0.5994 | 0.7684 | 0.6645    | 0.7764 | 0.7161 | 0.6926 | 58分  |
| DeepFM  | 0.5873 | 0.7649 | 0.6902    | 0.7081 | 0.6991 | 0.6956 | 74分  |
| NFM     | 0.6105 | 0.7545 | 0.6810    | 0.6857 | 0.6834 | 0.6828 | 49分  |
| AFM     | 0.6269 | 0.7271 | 0.6512    | 0.6851 | 0.6677 | 0.6596 | 84分  |

> tf

| Model  | Loss   | AUC    | Precision | Recall | F1     | ACC    |
| ------ | ------ | ------ | --------- | ------ | ------ | ------ |
| DNN    | 0.5450 | 0.7572 | 0.4927    | 0.5341 | 0.5126 | 0.7396 |
| ResNet | 0.6777 | 0.6233 | 0.3029    | 0.9185 | 0.4556 | 0.4373 |
| GFINN  | 0.4999 | 0.7750 | 0.5312    | 0.5339 | 0.5325 | 0.7597 |
| FCINN  | 0.5926 | 0.7470 | 0.6332    | 0.8160 | 0.7131 | 0.6922 |





