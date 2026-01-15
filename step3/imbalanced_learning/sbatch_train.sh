#!/bin/bash
#SBATCH --job-name=DREAM_TRAIN_step3_transformer
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=2-00:00:00
#SBATCH --output=../../runs/DREAM/%x_%j.log
#SBATCH --qos=long
#SBATCH --mail-user=yifan.jiang@sickkids.ca
#SBATCH --mail-type=ALL

# Load necessary modules
source ~/.zshrc
mamba activate dream311

# Get the job ID and job name
JOB_ID=${SLURM_JOB_ID}
JOB_NAME=${SLURM_JOB_NAME}

# Define the output directory using job name and job ID
BASE_LOG_DIR="../../runs/DREAM/${JOB_NAME}_${JOB_ID}"

# Define model types to train
MODEL_TYPES=(
    # "mlp"
    # "gcn" 
    "transformer"
)

# Define fingerprint types for MLP
FP_TYPES=(
    "MACCS"
    "RDK"
    "AVALON"
    "ATOMPAIR"
)

# Define data paths
TRAIN_CSV="../../datasets/DREAM/Train_Dataset_DREAM_Step3.parquet"
VAL_CSV="../../datasets/DREAM/Val_Dataset_DREAM_Step3.csv"

# Create base log directory
mkdir -p ${BASE_LOG_DIR}

# Process each model type
for model_type in "${MODEL_TYPES[@]}"; do
    echo "Training ${model_type} models..."
    
    if [[ "$model_type" == "mlp" ]]; then
        # For MLP, iterate through different fingerprint types
        for fp_type in "${FP_TYPES[@]}"; do
            LOG_DIR="${BASE_LOG_DIR}/${model_type}_${fp_type}"
            mkdir -p ${LOG_DIR}
            
            echo "Running MLP with fingerprint: ${fp_type}"
            python3 train.py \
                --train_csv ${TRAIN_CSV} \
                --val_csv ${VAL_CSV} \
                --model_type ${model_type} \
                --fp_type ${fp_type} \
                --batch_size 64 \
                --lr 0.001 \
                --epochs 200 \
                --dropout 0.1 \
                --mlp_hidden_dims 512 256 128 \
                --save_dir ${LOG_DIR}/checkpoints \
                --device cuda \
                2>&1 | tee ${LOG_DIR}/training.log
                
            echo "Completed MLP training with ${fp_type} fingerprint"
        done
    else
        # For GCN and Transformer
        LOG_DIR="${BASE_LOG_DIR}/${model_type}"
        mkdir -p ${LOG_DIR}
        
        if [[ "$model_type" == "gcn" ]]; then
            echo "Running GCN model"
            python3 train.py \
                --train_csv ${TRAIN_CSV} \
                --val_csv ${VAL_CSV} \
                --model_type ${model_type} \
                --batch_size 32 \
                --lr 0.001 \
                --epochs 100 \
                --dropout 0.1 \
                --hidden_feats 128 64 32 \
                --save_dir ${LOG_DIR}/checkpoints \
                --device cuda \
                2>&1 | tee ${LOG_DIR}/training.log
                
        elif [[ "$model_type" == "transformer" ]]; then
            echo "Running Transformer model"
            python3 train.py \
                --train_csv ${TRAIN_CSV} \
                --val_csv ${VAL_CSV} \
                --model_type ${model_type} \
                --batch_size 32 \
                --lr 0.001 \
                --epochs 100 \
                --dropout 0.1 \
                --vocab_size 65 \
                --max_length 100 \
                --hidden_size 128 \
                --num_layers 6 \
                --intermediate_size 512 \
                --num_attention_heads 8 \
                --attention_probs_dropout 0.1 \
                --hidden_dropout_rate 0.1 \
                --save_dir ${LOG_DIR}/checkpoints \
                --device cuda \
                2>&1 | tee ${LOG_DIR}/training.log
        fi
        
        echo "Completed ${model_type} training"
    fi
done

echo "All model training tasks completed."

# Print summary
echo "=================================================="
echo "Training Summary:"
echo "Job ID: ${JOB_ID}"
echo "Log Directory: ${BASE_LOG_DIR}"
echo "Models trained:"
for model_type in "${MODEL_TYPES[@]}"; do
    if [[ "$model_type" == "mlp" ]]; then
        for fp_type in "${FP_TYPES[@]}"; do
            echo "  - MLP with ${fp_type} fingerprint"
        done
    else
        echo "  - ${model_type}"
    fi
done
echo "=================================================="