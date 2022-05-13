# # docker pull docker.mirrors.ustc.edu.cn/tensorflow/tensorflow:2.2.2-gpu-py3
FROM docker.mirrors.ustc.edu.cn/tensorflow/tensorflow:2.2.2-gpu-py3

RUN echo "" > /etc/apt/sources.list.d/cuda.list
RUN sed -i "s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list
RUN sed -i "s@/security.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list
RUN apt-get update --fix-missing && apt-get install -y fontconfig --fix-missing
RUN apt-get install -y python3.7 python3-pip

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pandas sklearn scipy matplotlib seaborn h5py lightgbm xgboost && \
    pip install imbalance-xgboost

RUN pip3 install keras==2.4.3 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip3 install deepctr[gpu] -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

WORKDIR /ad_ctr

# cd /data/wangguisen/ctr_note/base_on_tf2
# docker build -t ad_ctr:3.0 -f ./dk/Dockerfile .

# docker run --gpus '"device=1"' --rm -it --name ad_ctr ad_ctr:3.0 bash


