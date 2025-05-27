#!/bin/bash
#SBATCH --job-name=DREAM_BASELINE_SEL   # General job name
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

# Define the options for fps_type and model_type
# FPS_TYPES=("MACCS" "RDK" "AVALON" "ATOMPAIR")
FPS_TYPES=("ATOMPAIR")
MODEL_TYPES=(
  # "lgbm"              # LightGBM
  # "xgboost"           # XGBoost with default settings
  # "rf"                # Random Forest
  # "extra_tree"        # Extra Trees Classifier
  # "histgb"            # Histogram-based Gradient Boosting
  #-----------------------------------------------------------#
  "xgb_limitdepth"    # XGBoost with max_depth parameter
  "lrl1"              # Logistic Regression with L1 regularization
  "catboost"          # CatBoost Classifier
  "kneighbor"         # K-Nearest Neighbors
)

# Iterate over all combinations of fps_type and model_type
for fps_type in "${FPS_TYPES[@]}"; do
    for model_type in "${MODEL_TYPES[@]}"; do
        # Define the specific log directory for this combination
        LOG_DIR="${BASE_LOG_DIR}/${fps_type}_${model_type}"
        mkdir -p ${LOG_DIR}

        # Run the Python script with the current combination
        echo "Running baseline_automl.py with --fps_type=${fps_type} and --model_type=${model_type}"
        python3 baseline_automl.py --log_dir ${LOG_DIR} --fps_type ${fps_type} --model_type ${model_type}
    done
done

echo "All tasks completed."