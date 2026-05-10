#!/bin/bash
#PBS -N Make_Ini_30
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
#PBS -j oe

cd "$PBS_O_WORKDIR"

source /home/chanju/miniconda3/etc/profile.d/conda.sh
conda activate cj

ENSEMBLE='ensemble.xyz'

echo "=== [1/2] ini.py started ==="
python ini.py "$ENSEMBLE" 30
echo "=== [1/2] ini.py finished ==="

echo "=== [2/2] opt.py started ==="
python opt.py
echo "=== [2/2] opt.py finished ==="
