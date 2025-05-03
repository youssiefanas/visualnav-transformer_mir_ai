#!/bin/bash
#SBATCH --job-name=nomad_train
#SBATCH --partition=mundus,besteffort,all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-20:1               #a100-40:1 - a40-48:1 -  l40-48:1     
#SBATCH --cpus-per-task=24             # 16 CPUs per task (1/4 of total)
#SBATCH --mem=32G                      # 64GB RAM
#SBATCH --time=24:00:00                # 24 hour time limit
#SBATCH --output=logs/slurm-%j.out     # Output log file
#SBATCH --error=logs/slurm-%j.err      # Error log file


# Load required modules
module purge
module load conda

# Activate conda environment
conda activate nomad_update

# Navigate to project directory
cd /home/mundus/ymorsy172/visualnav-transformer/train/

# Run training with both GPUs
python train.py -c config/nomad.yaml #--gpu_ids 0 1 
 #   --batch_size 8 \                  # Increased from 4 for multi-GPU
  #  --num_workers 8 \                 # Adjusted for 16 CPUs
#    --wandb_mode online               # Change to "offline" if no internet

echo "Job completed with status $?"
