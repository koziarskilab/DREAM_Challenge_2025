#!/bin/bash
#SBATCH --job-name=DREAM_TEST_STEP1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=./runs/DREAM/%x_%j.log
#SBATCH --qos=nopreemption
#SBATCH --mail-user=yifan.jiang@sickkids.ca
#SBATCH --mail-type=ALL

# Load necessary modules and environment
source ~/.zshrc
mamba activate dream

# Run the test prediction script
echo "Starting DREAM Challenge Step1 test predictions..."
python3 test_step1.py

echo "Test predictions completed."