#!/bin/bash
#SBATCH --nodes=2
#SBATCH --time=30-00:00:00
#SBATCH --job-name=C3D_FP
#SBATCH --partition=ce-mri
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=372G
#SBATCH --exclusive
#SBATCH --nodelist=d[3093,3094]
#SBATCH --output=/home/diaconu.d/mywork/script/%j.out
#SBATCH --error=/home/diaconu.d/mywork/script/%j.err

# Copy your slurm script for reproducing results in the future
CURRENT_JOBID=$(dirname $0)
cp $(dirname $0)/slurm.sh /home/diaconu.d/mywork/script/${CURRENT_JOBID:21}.sh
cd /home/diaconu.d/mywork/R2Plus1D-C3D/
pwd

# export NCCL_P2P_DISABLE=1  # IN AMD+A100 cluster
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$((RANDOM % 1000 + 25000))

srun torchrun \
--nnodes $SLURM_NNODES \
--nproc_per_node gpu \
--rdzv_id %j \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \ 
python train.py \
--num_epochs 100 \
--gpu_ids 0 \
--batch_size 8 


awk -v t=$SECONDS 'BEGIN{printf "Elapsed Time (HH:MM:SS): %d:%02d:%02d\n", t/3600, t/60%60, t%60}'
