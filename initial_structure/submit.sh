#!/bin/bash

echo "Submitting job to generate 30 initial structures..."

# batch.sh를 qsub로 제출
qsub batch.sh

echo "Job submitted! Check your queue status with 'qstat'."