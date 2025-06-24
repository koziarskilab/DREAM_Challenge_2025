#!/bin/bash
#SBATCH --job-name=DREAM_BASELINE_IMBALANCED_LEARNING_p1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=16:00:00
#SBATCH --output=../runs/DREAM/%x_%j.log
#SBATCH --qos=normal
#SBATCH --mail-user=yifan.jiang@sickkids.ca
#SBATCH --mail-type=ALL

# Load necessary modules
source ~/.zshrc
mamba activate dream311

# Get the job ID and job name
JOB_ID=${SLURM_JOB_ID}
JOB_NAME=${SLURM_JOB_NAME}

# Define the output directory using job name and job ID
BASE_LOG_DIR="../runs/DREAM/${JOB_NAME}_${JOB_ID}"

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

# Option 1: Process all possible combinations of all 4 fingerprints
FPS_COMBINATIONS=(
    "MACCS"                           # Single fingerprints
    "RDK"
    "AVALON"
    "ATOMPAIR"
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
    # "MACCS,RDK,AVALON,ATOMPAIR"      # All four
)

# Process each combination with each imbalanced method
for fps_combination in "${FPS_COMBINATIONS[@]}"; do
    for imbalanced_method in "${IMBALANCED_METHODS[@]}"; do
        # Create a safe directory name by replacing commas with underscores
        fps_dir_name=$(echo ${fps_combination} | sed 's/,/_/g')
        LOG_DIR="${BASE_LOG_DIR}/${fps_dir_name}_MLP_${imbalanced_method}"
        mkdir -p ${LOG_DIR}

        echo "Running with fingerprint combination: ${fps_combination}, method: ${imbalanced_method}"
        python3 baseline_imbalanced_learning.py \
            --log_dir ${LOG_DIR} \
            --fps_type "${fps_combination}" \
            --imbalanced ${imbalanced_method}
    done
done

echo "All combination tasks completed."