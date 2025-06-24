#!/bin/bash
#SBATCH --job-name=DREAM_ENSEMBLE_FPS_xgb_limitdepth
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=../runs/DREAM/%x_%j.log
#SBATCH --qos=nopreemption
#SBATCH --mail-user=yifan.jiang@sickkids.ca
#SBATCH --mail-type=ALL

# Load necessary modules
source ~/.zshrc
mamba activate dream

# Get the job ID and job name
JOB_ID=${SLURM_JOB_ID}
JOB_NAME=${SLURM_JOB_NAME}

# Define the output directory using job name and job ID
BASE_LOG_DIR="../runs/DREAM/${JOB_NAME}_${JOB_ID}"

# Define model types to test
MODEL_TYPES=(
    # "lgbm"              # LightGBM
    # "xgboost"           # XGBoost with default settings
    # "rf"                # Random Forest
    # "extra_tree"        # Extra Trees Classifier
    # "histgb"            # Histogram-based Gradient Boosting
    "xgb_limitdepth"    # XGBoost with max_depth parameter
    # "lrl1"              # Logistic Regression with L1 regularization
    # "catboost"          # CatBoost Classifier
    # "kneighbor"         # K-Nearest Neighbors
)

# Define all 15 possible combinations of fingerprints (excluding empty set)
FPS_COMBINATIONS=(
    # "MACCS"                           # Single fingerprints
    # "RDK"
    # "AVALON"
    # "ATOMPAIR"
    # "MACCS,RDK"                       # Pairs
    # "MACCS,AVALON"
    # "MACCS,ATOMPAIR"
    # "RDK,AVALON"
    # "RDK,ATOMPAIR"
    # "AVALON,ATOMPAIR"
    # "MACCS,RDK,AVALON"               # Triplets
    # "MACCS,RDK,ATOMPAIR"
    # "MACCS,AVALON,ATOMPAIR"
    # "RDK,AVALON,ATOMPAIR"
    "MACCS,RDK,AVALON,ATOMPAIR"      # All four
)

# Define time budget per combination (in seconds)
TIME_BUDGET=18000  # 5 hour per combination

# Process each combination with each model type
for model_type in "${MODEL_TYPES[@]}"; do
    for fps_combination in "${FPS_COMBINATIONS[@]}"; do
        # Create a safe directory name by replacing commas with underscores
        fps_dir_name=$(echo ${fps_combination} | sed 's/,/_/g')
        LOG_DIR="${BASE_LOG_DIR}/${fps_dir_name}_${model_type}"
        mkdir -p ${LOG_DIR}

        echo "Running with fingerprint combination: ${fps_combination}, model: ${model_type}"
        python3 baseline_fingerprint_concatenate.py \
            --log_dir ${LOG_DIR} \
            --fps_type "${fps_combination}" \
            --model_type ${model_type} \
            --time_budget ${TIME_BUDGET}
    done
done

echo "All combination tasks completed."