#!/bin/bash

SAMPLES=5

jids=()
for i in {1..50}
do
    echo "Submitting job for Ag$i..."
    jid=$(qsub -v SIZE=$i,SAMPLES=$SAMPLES -N Ag$i batch.sh)
    jids+=("$jid")
    sleep 0.5
done

# Submit merge job inline after all sampling jobs complete
depend=$(printf ":%s" "${jids[@]}")
qsub -W depend=afterok${depend} << 'EOF'
#!/bin/bash
#PBS -N merge_ensemble
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:30:00
#PBS -j oe

cd $PBS_O_WORKDIR
source /home/chanju/miniconda3/etc/profile.d/conda.sh
conda activate cj
python ensemble.py --merge --output ensemble.xyz
EOF

echo "Done! Submitted ${#jids[@]} sampling jobs + 1 merge job."
echo "Check queue with: qstat"
