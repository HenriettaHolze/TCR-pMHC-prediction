#!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J myJob
 #BSUB -n 1
 #BSUB -W 01:00
 #BSUB -R "rusage[mem=32GB]"
 #BSUB -o logs/esm_pca.out
 #BSUB -e logs/esm_pca.err

 echo "Running script..."
 cd ~/TCR-pMHC-prediction/005.ESM_pca
 python3 esm_pca.py
