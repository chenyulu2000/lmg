#!/bin/sh
source activate lmg
cd /home/luchenyu/lmg
export PYTHONPATH=/home/luchenyu/lmg
pwd

phase=test
config=configs/test.yaml

now=$(date +"%Y%m%d_%H%M%S")

exp_dir=exps/${phase}_$now

mkdir $exp_dir
cp scripts/test.sh ${config} ${exp_dir}

python test.py \
--config=${config} \
--datetime=$now \
--exp_dir=$exp_dir
