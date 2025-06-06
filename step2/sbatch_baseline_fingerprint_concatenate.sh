#!/bin/bash
#SBATCH --job-name=DREAM_ENSEMBLE_FPS_extra_tree   # Job name
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32              # Number of CPU cores
#SBATCH --mem=128G                       # Memory allocation
#SBATCH --time=4-00:00:00                 # Time limit (48 hours for all combinations)
#SBATCH --output=../runs/DREAM/%x_%j.log    # Output log file
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
BASE_LOG_DIR="../runs/DREAM/${JOB_NAME}_${JOB_ID}"

# Define the fingerprint types to ensemble
FPS_TYPES="MACCS,RDK,AVALON,ATOMPAIR"  # All four fingerprints
# FPS_TYPES="MACCS,RDK,AVALON"         # Uncomment for subset testing

# Define model types to test
MODEL_TYPES=(
    # "lgbm"              # LightGBM
    # "xgboost"           # XGBoost
    # "rf"                # Random Forest
    "extra_tree"        # Extra Trees
    # "catboost"        # CatBoost (uncomment if needed)
    # "histgb"          # Histogram-based GB (uncomment if needed)
)

# Define time budget per combination (in seconds)
TIME_BUDGET=18000  # 5 hour per combination (adjust based on your needs)

# Iterate over all model types
for model_type in "${MODEL_TYPES[@]}"; do
    # Define the specific log directory for this model type
    LOG_DIR="${BASE_LOG_DIR}/${model_type}_ensemble"
    mkdir -p ${LOG_DIR}
    
    echo "==========================================================="
    echo "Running ensemble evaluation with model: ${model_type}"
    echo "Fingerprint types: ${FPS_TYPES}"
    echo "Time budget per combination: ${TIME_BUDGET} seconds"
    echo "Log directory: ${LOG_DIR}"
    echo "==========================================================="
    
    # Run the ensemble fingerprint evaluation
    python3 baseline_fingerprint_concatenate.py \
        --log_dir ${LOG_DIR} \
        --fps_types ${FPS_TYPES} \
        --model_type ${model_type} \
        --time_budget ${TIME_BUDGET}
    
    echo "Completed ensemble evaluation for model: ${model_type}"
    echo "Results saved in: ${LOG_DIR}/../ensemble_model_results.csv"
    echo "Summary saved in: ${LOG_DIR}/../combinations_summary.csv"
    echo "-----------------------------------------------------------"
done

echo "All ensemble evaluations completed!"