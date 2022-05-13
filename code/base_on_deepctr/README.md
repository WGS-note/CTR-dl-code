### deepctr

官网：https://deepctr-doc.readthedocs.io/en/latest/Quick-Start.html

https://github.com/shenweichen/DeepCTR

```shell
pip install deepctr[cpu]
pip install deepctr[gpu]
```

> 在下deepctr之前，要保证环境里有tf、keras，否则deepctr默认下最新版的。

```python
# 本地环境如下
print(tf.__version__)
print(keras.__version__)

2.2.0
2.4.3
```



### 目录说明

+ dk：GPU运行命令
+ enc：编码保存路径
+ imgs：可视化保存路径
+ src：code



### 文件说明

+ ctr_XXX.py：XXX网络实现代码
+ tools.py：用到的工具方法
+ settings：配置路径

其中，DIN、DIEN用的是亚马逊数据集`sample_data.txt`，其它用的是`criteo_sampled_data.csv`。



### CPU本地运行

**窗口运行**

```shell
# 进入 base_on_deepctr 目录
cd XXX/推荐(广告)-精排-CTR模型/code/base_on_deepctr

# 将 settings.py 中的 runFalg 设为2
runFalg = 2

# 以MLR为例运行
python ./src/ctr_MLR.py 
```

**pycharm右键运行**

> runFalg = 3



### GPU-docker运行

**dockerfile**

```dockerfile
# # docker pull docker.mirrors.ustc.edu.cn/tensorflow/tensorflow:2.2.2-gpu-py3
FROM docker.mirrors.ustc.edu.cn/tensorflow/tensorflow:2.2.2-gpu-py3

RUN echo "" > /etc/apt/sources.list.d/cuda.list
RUN sed -i "s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list
RUN sed -i "s@/security.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list
RUN apt-get update --fix-missing && apt-get install -y fontconfig --fix-missing
RUN apt-get install -y python3.7 python3-pip

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pandas sklearn scipy matplotlib seaborn pyyaml h5py hdfs pyspark

RUN pip3 install keras==2.4.3 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip3 install deepctr[gpu] -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

WORKDIR /ad_ctr

# cd /data/wangguisen/ad_ctr
# docker build -t ad_ctr:2.0 -f ./dk/Dockerfile_tf_keras .
# docker run --gpus '"device=1"' --rm -it --name ad_ctr ad_ctr:2.0 bash
```

> 路径：/data/wangguisen/ctr_note

```shell
# 将 settings.py 中的 runFalg 设为1
runFalg = 1

# 以MLR为例运行，其它运行在run.sh里
docker run -d --gpus '"device=1"' \
    --rm -it --name ctr_deepctr \
    -v /data/wangguisen/ctr_note/base_on_deepctr:/ad_ctr/base_on_deepctr \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:2.0 \
    sh -c 'python3 -u /ad_ctr/base_on_deepctr/src/ctr_MLR.py 1>>/ad_ctr/base_on_deepctr/log/ctr_MLR.log 2>>/ad_ctr/base_on_deepctr/log/ctr_MLR.err'
```



### 保存

```python
from tensorflow.keras.models import save_model, load_model
# 保存成yaml文件,用于tf serving在线服务
deep.save(settings.save_path.format('DIN', 'DINmodel-11-13_serving'), save_format="tf")
# 保存成h5文件，用于离线评估
save_model(deep, settings.save_path.format('DIN', 'DINmodel-11-13.h5'))
```







