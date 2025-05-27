#!/bin/bash
#SBATCH --job-name=FP_CORRELATION
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=./runs/DREAM/fingerprint_correlation_%j.log
#SBATCH --qos=nopreemption
#SBATCH --mail-user=yifan.jiang@sickkids.ca
#SBATCH --mail-type=ALL

# Load necessary modules
source ~/.zshrc
mamba activate dream

# Get the job ID and job name
JOB_ID=${SLURM_JOB_ID}
JOB_NAME=${SLURM_JOB_NAME}

# Set output directory with job ID
OUTPUT_DIR="./runs/DREAM/${JOB_NAME}_${JOB_ID}"

# Define fingerprints to analyze
FPS_TYPES=("MACCS" "RDK" "AVALON" "ATOMPAIR")
MODEL_TYPES=("lgbm" "xgboost" "rf" "extra_tree" "histgb" "xgb_limitdepth" "lrl1" "catboost" "kneighbor")

# You can comment out elements you don't want to analyze
# For example:
# FPS_TYPES=("MACCS" "AVALON")

echo "Starting fingerprint correlation analysis..."
echo "Analyzing fingerprints: ${FPS_TYPES[@]}"
echo "Model types: ${MODEL_TYPES[@]}"

# Join array elements for passing to Python script
FPS_ARGS=""
for fps in "${FPS_TYPES[@]}"; do
    FPS_ARGS+="$fps "
done

MODEL_ARGS=""
for model in "${MODEL_TYPES[@]}"; do
    MODEL_ARGS+="$model "
done

# Run fingerprint correlation analysis
python evaluate_fingerprint_correlation.py \
    --output_dir ${OUTPUT_DIR} \
    --fps_types ${FPS_ARGS} \
    --model_types ${MODEL_ARGS}

echo "Analysis completed. Results saved to ${OUTPUT_DIR}"