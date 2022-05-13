#!/bin/bash

dump() {
  # docker run --gpus '"device=1"' --rm -it --name adCTR ad_ctr:2.0 bash

  # cd /data/wangguisen/ctr_note/base_on_deepctr

  # # # MLR
  docker run -d --gpus '"device=0"' \
    --rm -it --name ctr_deepctr_MLR \
    -v /data/wangguisen/ctr_note/base_on_deepctr:/ad_ctr/base_on_deepctr \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:2.0 \
    sh -c 'python3 -u /ad_ctr/base_on_deepctr/src/ctr_MLR.py 1>>/ad_ctr/base_on_deepctr/log/ctr_MLR.log 2>>/ad_ctr/base_on_deepctr/log/ctr_MLR.err'

  # # # FNN
  docker run -d --gpus '"device=1"' \
    --rm -it --name ctr_deepctr_FNN \
    -v /data/wangguisen/ctr_note/base_on_deepctr:/ad_ctr/base_on_deepctr \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:2.0 \
    sh -c 'python3 -u /ad_ctr/base_on_deepctr/src/ctr_FNN.py 1>>/ad_ctr/base_on_deepctr/log/ctr_FNN.log 2>>/ad_ctr/base_on_deepctr/log/ctr_FNN.err'

  # # # WDL
  docker run -d --gpus '"device=0"' \
    --rm -it --name ctr_deepctr_WDL \
    -v /data/wangguisen/ctr_note/base_on_deepctr:/ad_ctr/base_on_deepctr \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:2.0 \
    sh -c 'python3 -u /ad_ctr/base_on_deepctr/src/ctr_WDL.py 1>>/ad_ctr/base_on_deepctr/log/ctr_WDL.log 2>>/ad_ctr/base_on_deepctr/log/ctr_WDL.err'

  # # # DCN
  docker run -d --gpus '"device=1"' \
    --rm -it --name ctr_deepctr_DCN \
    -v /data/wangguisen/ctr_note/base_on_deepctr:/ad_ctr/base_on_deepctr \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:2.0 \
    sh -c 'python3 -u /ad_ctr/base_on_deepctr/src/ctr_DCN.py 1>>/ad_ctr/base_on_deepctr/log/ctr_DCN.log 2>>/ad_ctr/base_on_deepctr/log/ctr_DCN.err'

  # # # DeepFM
  docker run -d --gpus '"device=0"' \
    --rm -it --name ctr_deepctr_DeepFM \
    -v /data/wangguisen/ctr_note/base_on_deepctr:/ad_ctr/base_on_deepctr \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:2.0 \
    sh -c 'python3 -u /ad_ctr/base_on_deepctr/src/ctr_DeepFM.py 1>>/ad_ctr/base_on_deepctr/log/ctr_DeepFM.log 2>>/ad_ctr/base_on_deepctr/log/ctr_DeepFM.err'

  # # # NFM
  docker run -d --gpus '"device=1"' \
    --rm -it --name ctr_deepctr_NFM \
    -v /data/wangguisen/ctr_note/base_on_deepctr:/ad_ctr/base_on_deepctr \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:2.0 \
    sh -c 'python3 -u /ad_ctr/base_on_deepctr/src/ctr_NFM.py 1>>/ad_ctr/base_on_deepctr/log/ctr_NFM.log 2>>/ad_ctr/base_on_deepctr/log/ctr_NFM.err'

  # # # AFM
  docker run -d --gpus '"device=1"' \
    --rm -it --name ctr_deepctr_AFM \
    -v /data/wangguisen/ctr_note/base_on_deepctr:/ad_ctr/base_on_deepctr \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:2.0 \
    sh -c 'python3 -u /ad_ctr/base_on_deepctr/src/ctr_AFM.py 1>>/ad_ctr/base_on_deepctr/log/ctr_AFM.log 2>>/ad_ctr/base_on_deepctr/log/ctr_AFM.err'

  # # # FiBiNET
  docker run -d --gpus '"device=0"' \
    --rm -it --name ctr_deepctr_FiBiNET \
    -v /data/wangguisen/ctr_note/base_on_deepctr:/ad_ctr/base_on_deepctr \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:2.0 \
    sh -c 'python3 -u /ad_ctr/base_on_deepctr/src/ctr_FiBiNET.py 1>>/ad_ctr/base_on_deepctr/log/ctr_FiBiNET.log 2>>/ad_ctr/base_on_deepctr/log/ctr_FiBiNET.err'

  # # # DIN
  docker run -d --gpus '"device=0"' \
    --rm -it --name ctr_deepctr_DIN \
    -v /data/wangguisen/ctr_note/base_on_deepctr:/ad_ctr/base_on_deepctr \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:2.0 \
    sh -c 'python3 -u /ad_ctr/base_on_deepctr/src/ctr_DIN.py 1>>/ad_ctr/base_on_deepctr/log/ctr_DIN.log 2>>/ad_ctr/base_on_deepctr/log/ctr_DIN.err'

  # # # DIEN
  docker run -d --gpus '"device=1"' \
    --rm -it --name ctr_deepctr_DIEN \
    -v /data/wangguisen/ctr_note/base_on_deepctr:/ad_ctr/base_on_deepctr \
    -v /data/wangguisen/ctr_note/data:/ad_ctr/data \
    ad_ctr:2.0 \
    sh -c 'python3 -u /ad_ctr/base_on_deepctr/src/ctr_DIEN.py 1>>/ad_ctr/base_on_deepctr/log/ctr_DIEN.log 2>>/ad_ctr/base_on_deepctr/log/ctr_DIEN.err'

}






