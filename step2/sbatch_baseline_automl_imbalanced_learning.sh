#!/bin/bash
#SBATCH --job-name=DREAM_BASELINE_IMBALANCED_LEARNING_ATOMPAIR   # Updated job name
#SBATCH --cpus-per-task=16           # Number of CPU cores
#SBATCH --mem=64G                    # Memory allocation
#SBATCH --gres=gpu:rtx6000:1         # Request 1 GPU
#SBATCH --time=16:00:00              # Time limit
#SBATCH --output=../runs/DREAM/%x_%j.log    # Output log file
#SBATCH --qos=normal
#SBATCH --mail-user=yifan.jiang@sickkids.ca
#SBATCH --mail-type=ALL

# Load necessary modules (if applicable)
source ~/.zshrc
mamba activate dream311

# Get the job ID and job name
JOB_ID=${SLURM_JOB_ID}
JOB_NAME=${SLURM_JOB_NAME}

# Define the output directory using job name and job ID
BASE_LOG_DIR="../runs/DREAM/${JOB_NAME}_${JOB_ID}"

# Define the options for fps_type and imbalanced_methods
# FPS_TYPES=("MACCS" "RDK" "AVALON" "ATOMPAIR")
FPS_TYPES=("ATOMPAIR")  # Uncomment for single fingerprint testing

# Define imbalanced learning types
IMBALANCED_METHODS=(
  "CE"                # Standard cross-entropy loss (no imbalanced learning)
  "CB_F"              # Class-balanced focal loss
  "BS"                # Balanced softmax
  "CB_CE"             # Class-balanced cross-entropy loss
  "CS"                # Cost-sensitive cross-entropy loss
  "IB"                # Influence-balanced loss
  "CDT"               # Class-dependent temperatures
  "MIXUP"             # Info augmentation with mixup
  "REMIX"             # Info augmentation with remix
  "DECOUPLING"        # Decoupling representation and classifier
  "BBN"               # Bilateral-branch network
)

# Iterate over all combinations of fps_type and loss_type
for fps_type in "${FPS_TYPES[@]}"; do
    for imbalanced_method in "${IMBALANCED_METHODS[@]}"; do
        # Define the specific log directory for this combination
        LOG_DIR="${BASE_LOG_DIR}/${fps_type}_MLP_${imbalanced_method}"
        mkdir -p ${LOG_DIR}

        # Run the Python script with the current combination
        echo "Running baseline_automl_imbalanced_learning.py with --fps_type=${fps_type}, --imbalanced=${imbalanced_method}"
        python3 baseline_automl_imbalanced_learning.py \
            --log_dir ${LOG_DIR} \
            --fps_type ${fps_type} \
            --imbalanced ${imbalanced_method}
    done
done

echo "All tasks completed."