#!/bin/bash
#SBATCH --job-name=DREAM_BASELINE_SEL_PROXY_Part8   # General job name
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32           # Number of CPU cores
#SBATCH --mem=128G                   # Memory allocation
#SBATCH --time=4-00:00:00            # Time limit
#SBATCH --output=./runs/DREAM/%x_%j.log    # Output log file
#SBATCH --qos=nopreemption
#SBATCH --mail-user=yifan.jiang@sickkids.ca
#SBATCH --mail-type=ALL

# Load necessary modules (if applicable)
source ~/.zshrc
mamba activate dream

# Get the job ID and job name
JOB_ID=${SLURM_JOB_ID}
JOB_NAME=${SLURM_JOB_NAME}

# Define the output directory using job name and job ID
BASE_LOG_DIR="./runs/DREAM/${JOB_NAME}_${JOB_ID}"

# Define the options for fps_type, model_type, and metrics
FPS_TYPES=("AVALON" "ATOMPAIR")
# FPS_TYPES=("AVALON")
MODEL_TYPES=(
  "lgbm"              # LightGBM
  "xgboost"           # XGBoost with default settings
  "rf"                # Random Forest
  "extra_tree"        # Extra Trees Classifier
  "histgb"            # Histogram-based Gradient Boosting
)
METRICS=(
  # Classification metrics
#   "accuracy"          # 1 - accuracy (to minimize)
#   "log_loss"          # Default for multiclass classification
#   "roc_auc"           # 1 - roc_auc_score (Default for binary classification)
#   "roc_auc_weighted"  # ROC AUC with average="weighted"
#   "f1"                # 1 - f1_score
#   "micro_f1"          # 1 - f1_score with average="micro"
#   "macro_f1"          # 1 - f1_score with average="macro"
  "ap"                # 1 - average_precision_score (PRAUC)
)

# Iterate over all combinations of metric, fps_type, and model_type
for metric in "${METRICS[@]}"; do
    for fps_type in "${FPS_TYPES[@]}"; do
        for model_type in "${MODEL_TYPES[@]}"; do
            # Define the specific log directory for this combination
            LOG_DIR="${BASE_LOG_DIR}/${metric}_${fps_type}_${model_type}"
            mkdir -p ${LOG_DIR}

            # Run the Python script with the current combination
            echo "Running baseline_automl_proxy.py with --metric=${metric} --fps_type=${fps_type} and --model_type=${model_type}"
            python3 baseline_automl_proxy.py --log_dir ${LOG_DIR} --fps_type ${fps_type} --model_type ${model_type} --metric ${metric}
        done
    done
done

echo "All tasks completed."