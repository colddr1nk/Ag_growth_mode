#!/bin/bash
#PBS -N gcga
#PBS -q full
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=72:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

source /home/chanju/miniconda3/etc/profile.d/conda.sh
conda activate cj

python gcga.py
