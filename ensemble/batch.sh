#!/bin/bash
#PBS -q full
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -j oe

source /home/chanju/miniconda3/etc/profile.d/conda.sh
conda activate cj

cd $PBS_O_WORKDIR

# 넘겨받은 SIZE(원자수)와 SAMPLES(샘플링 수)를 파이썬에 투입
python -u ensemble.py ${SIZE} ${SAMPLES}

exit 0