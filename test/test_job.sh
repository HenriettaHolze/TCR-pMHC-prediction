#!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J myJob
 #BSUB -n 1
 #BSUB -W 01:00
 #BSUB -R "rusage[mem=32GB]"
 #BSUB -o logs/test.out
 #BSUB -e logs/test.err

 echo "Running script..."
 cd ~/TCR-pMHC-prediction/test
 python3 test.py
