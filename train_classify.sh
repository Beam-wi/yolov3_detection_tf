#!/bin/sh
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
echo "$0"
echo "$1" #工程参数配置文件
echo "$2" #默认参数配置文件
echo "start train"
python3 ./train_classify_auto.py --config_project "$1" --config_common "$2"
echo "end train"
