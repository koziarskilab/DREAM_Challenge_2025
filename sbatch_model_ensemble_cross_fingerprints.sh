#!/bin/bash
#SBATCH --job-name=MODEL_ENSEMBLE_CROSS_FPS_simple_average   # Updated job name
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
LOG_DIR="./runs/DREAM/${JOB_NAME}_${JOB_ID}"
mkdir -p ${LOG_DIR}

# Define ensemble methods to test
ENSEMBLE_METHODS=("simple_average")

# Define different max_pairs values to test
MAX_PAIRS_VALUES=(5)

echo "Running cross-fingerprint ensemble with predefined low-correlation pairs..."
echo "The script will automatically use the 8 predefined pairs from correlation analysis:"
echo "  1. MACCS_kneighbor & ATOMPAIR_catboost"
echo "  2. MACCS_kneighbor & RDK_catboost"
echo "  3. AVALON_lrl1 & ATOMPAIR_xgboost"
echo "  4. RDK_histgb & ATOMPAIR_lgbm"
echo "  5. AVALON_rf & RDK_extra_tree"
echo "  6. AVALON_xgb_limitdepth & ATOMPAIR_rf"
echo "  7. AVALON_histgb & ATOMPAIR_histgb"
echo "  8. AVALON_xgboost & AVALON_extra_tree"
echo "----------------------------------------"

# Run experiments for each ensemble method and max_pairs combination
for method in "${ENSEMBLE_METHODS[@]}"; do 
    for max_pairs in "${MAX_PAIRS_VALUES[@]}"; do
        echo "Starting ${method} ensemble method with max_pairs=${max_pairs}..."
        
        # Create method-specific log directory
        METHOD_LOG_DIR="${LOG_DIR}/${method}_maxpairs${max_pairs}"
        
        # Construct and execute the command
        CMD="python model_ensemble_cross_fingerprints.py --log_dir ${METHOD_LOG_DIR} --ensemble_method ${method} --max_pairs ${max_pairs}"
        
        echo "Running: ${CMD}"
        eval ${CMD}
        
        echo "Completed ${method} ensemble with max_pairs=${max_pairs}"
        echo "Results saved in ${METHOD_LOG_DIR}"
        echo "----------------------------------------"
    done
done

echo "All cross-fingerprint ensemble experiments completed."
echo "Check results in: ${LOG_DIR}"

# Print summary of what was run
echo ""
echo "EXPERIMENT SUMMARY:"
echo "==================="
echo "Ensemble methods tested: ${ENSEMBLE_METHODS[@]}"
echo "Max pairs values tested: ${MAX_PAIRS_VALUES[@]}"
echo "Total experiments: $((${#ENSEMBLE_METHODS[@]} * ${#MAX_PAIRS_VALUES[@]}))"
echo ""
echo "Each experiment evaluates all combinations from 1 to max_pairs using the 8 predefined low-correlation pairs."
echo "Check the combination_summary.csv files in each method directory for detailed results."