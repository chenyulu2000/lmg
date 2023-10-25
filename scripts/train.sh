#!/bin/sh
source activate lmg
cd /home/luchenyu/lmg
pwd

phase=train
config=configs/train.yaml

now=$(date +"%Y%m%d_%H%M%S")

exp_dir=exps/${phase}_$now

mkdir $exp_dir
mkdir ${exp_dir}/src_code
cp scripts/train.sh ${config} -r model/ -r components/ data/dataset.py train.py ${exp_dir}/src_code

CUDA_VISIBLE_DEVICES=4 \
  python -m torch.distributed.launch \
  --master_port=29508 \
  --nproc_per_node=1 \
  train.py \
  --config=${config} \
  --datetime=$now \
  --exp_dir=$exp_dir \
#  --is_debug=True
