#!/bin/bash

echo "Submitting GCGA job..."
qsub batch.sh
echo "Job submitted! Check queue with: qstat"
