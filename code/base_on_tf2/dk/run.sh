#!/bin/bash
today=$(date -d "now" +%Y-%m-%d)
yesterday=$(date -d "yesterday" +%Y-%m-%d)

dump(){

  docker run -d --gpus '"device=1"' \
    --rm -it --name ctr_tf_DCN \
    -v /data/wangguisen/ctr_note/base_on_tf2:/ad_ctr/base_on_tf2 \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:3.0 \
    sh -c 'python3 -u /ad_ctr/base_on_tf2/src/ctr_DCN.py 1>>/ad_ctr/base_on_tf2/log/ctr_DCN.log 2>>/ad_ctr/base_on_tf2/log/ctr_DCN.err'

  docker run -d --gpus '"device=1"' \
    --rm -it --name ctr_tf_AFM \
    -v /data/wangguisen/ctr_note/base_on_tf2:/ad_ctr/base_on_tf2 \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:3.0 \
    sh -c 'python3 -u /ad_ctr/base_on_tf2/src/ctr_AFM.py 1>>/ad_ctr/base_on_tf2/log/ctr_AFM.log 2>>/ad_ctr/base_on_tf2/log/ctr_AFM.err'

  docker run -d --gpus '"device=1"' \
    --rm -it --name ctr_tf_DIN \
    -v /data/wangguisen/ctr_note/base_on_tf2:/ad_ctr/base_on_tf2 \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:3.0 \
    sh -c 'python3 -u /ad_ctr/base_on_tf2/src/ctr_DIN.py 1>>/ad_ctr/base_on_tf2/log/ctr_DIN.log 2>>/ad_ctr/base_on_tf2/log/ctr_DIN.err'

}



