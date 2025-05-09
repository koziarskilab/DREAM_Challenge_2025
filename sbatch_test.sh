#!/bin/bash
#SBATCH --job-name=DREAM_TEST   # General job name
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32      # Number of CPU cores
#SBATCH --mem=128G              # Memory allocation
#SBATCH --time=4-00:00:00       # Time limit
#SBATCH --output=./runs/DREAM/%x_%j.log    # Output log file
#SBATCH --qos=nopreemption

# Load necessary modules (if applicable)
source ~/.zshrc
mamba activate dream

# Get the job ID
JOB_ID=${SLURM_JOB_ID}
echo "Starting test evaluation job with ID: ${JOB_ID}"

# Get the source directory that contains model subdirectories
SOURCE_DIR=${1:-"./runs/DREAM/DREAM_BASELINE_SEL_16135584"}

# Check if the directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Directory $SOURCE_DIR does not exist."
  echo "Usage: sbatch sbatch_test.sh [source_directory]"
  exit 1
fi

# Define the available fingerprint types and model types
FPS_TYPES=("MACCS" "RDK" "AVALON" "ATOMPAIR")
MODEL_TYPES=(
  "lgbm"              # LightGBM
  "xgboost"           # XGBoost with default settings
  "xgb_limitdepth"    # XGBoost with max_depth parameter
  "rf"                # Random Forest
  "extra_tree"        # Extra Trees Classifier
  "histgb"            # Histogram-based Gradient Boosting
  "lrl1"              # Logistic Regression with L1 regularization
  "lrl2"              # Logistic Regression with L2 regularization
  "catboost"          # CatBoost Classifier
  "kneighbor"         # K-Nearest Neighbors
)

# You can comment out elements you don't want to test
# For example, to test only MACCS and AVALON fingerprints:
FPS_TYPES=("MACCS")

# Similarly, to test only lgbm and rf models:
MODEL_TYPES=("lgbm")

echo "Processing models in directory: $SOURCE_DIR"
echo "Selected fingerprint types: ${FPS_TYPES[@]}"
echo "Selected model types: ${MODEL_TYPES[@]}"

# Initialize counters
TOTAL_MODELS=0
SUCCESSFUL_MODELS=0
FAILED_MODELS=0

# Process each combination
for fps_type in "${FPS_TYPES[@]}"; do
  for model_type in "${MODEL_TYPES[@]}"; do
    # Construct the model directory path
    MODEL_DIR="${SOURCE_DIR}/${fps_type}_${model_type}"
    MODEL_PATH="${MODEL_DIR}/best_model.pkl"
    
    # Check if the model exists
    if [ -f "$MODEL_PATH" ]; then
      TOTAL_MODELS=$((TOTAL_MODELS+1))
      
      echo "Processing model: ${fps_type}_${model_type}"
      echo "Model path: $MODEL_PATH"
      
      # Run the test.py script
      if python test.py --model_path "$MODEL_PATH"; then
        echo "Successfully tested ${fps_type}_${model_type}"
        SUCCESSFUL_MODELS=$((SUCCESSFUL_MODELS+1))
      else
        echo "Failed to test ${fps_type}_${model_type}"
        FAILED_MODELS=$((FAILED_MODELS+1))
      fi
      
      echo "----------------------------------------"
    else
      echo "Skipping ${fps_type}_${model_type} - model file not found at $MODEL_PATH"
    fi
  done
done

echo "Testing completed."
echo "Total models processed: $TOTAL_MODELS"
echo "Successful tests: $SUCCESSFUL_MODELS"
echo "Failed tests: $FAILED_MODELS"
echo "Results are saved in their respective model directories."