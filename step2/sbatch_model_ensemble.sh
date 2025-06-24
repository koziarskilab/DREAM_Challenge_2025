#!/bin/bash
#SBATCH --job-name=DREAM_MODEL_ENSEMBLE
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
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

# Define ensemble methods to test
ENSEMBLE_METHODS=(
    "simple_average"
    "weighted_average_clusters"
    "majority_vote"
    "max_vote"
    "min_vote"
    "median_vote"
    "stacking_logistic"
    "stacking_rf"
)

echo "Starting ensemble evaluation with four pretrained models:"
echo "  1. histgb with RDK_AVALON_ATOMPAIR"
echo "  2. lgbm with MACCS_RDK_AVALON"
echo "  3. xgb_limitdepth with MACCS_ATOMPAIR"
echo "  4. xgboost with MACCS_RDK_AVALON_ATOMPAIR"
echo "Results will be collected in: ${BASE_LOG_DIR}/ensemble_methods_results.csv"
echo "----------------------------------------"

# Process each ensemble method
for ensemble_method in "${ENSEMBLE_METHODS[@]}"; do
    LOG_DIR="${BASE_LOG_DIR}/${ensemble_method}"
    mkdir -p ${LOG_DIR}
    
    echo "Running ensemble method: ${ensemble_method}"
    python3 model_ensemble.py \
        --log_dir ${LOG_DIR} \
        --ensemble_method ${ensemble_method}
    
    echo "Completed ensemble method: ${ensemble_method}"
    echo "----------------------------------------"
done

echo "All ensemble methods completed."
echo "Master results CSV: ${BASE_LOG_DIR}/ensemble_methods_results.csv"
echo "Individual results in subdirectories: ${BASE_LOG_DIR}/<method_name>/"