from helper import Dataset, ProcessData
import argparse
import os
import pandas as pd
from flaml.automl.automl import AutoML
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)  # PRAUC and ROC-AUC metrics
import datetime


# Function to update the results CSV file
def update_results_csv(parent_dir, model_type, fps_type, prauc, roc_auc):
    # Define the CSV file path
    results_file = os.path.join(parent_dir, "model_results.csv")

    # Prepare the new data row
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {
        "Timestamp": timestamp,
        "ModelType": model_type,
        "FingerprintType": fps_type,
        "PRAUC": prauc,
        "ROC-AUC": roc_auc,
    }

    # Check if the file exists
    if os.path.exists(results_file):
        # Read existing data and append new row
        df_results = pd.read_csv(results_file)
        df_results = pd.concat(
            [df_results, pd.DataFrame([new_data])], ignore_index=True
        )
    else:
        # Create new dataframe with the headers
        df_results = pd.DataFrame([new_data])

    # Save the updated dataframe
    df_results.to_csv(results_file, index=False)
    print(f"Results updated in {results_file}")


# Main function
def main(args):
    # Ensure log_dir exists
    os.makedirs(args.log_dir, exist_ok=True)

    # Get the parent directory of log_dir
    parent_dir = os.path.dirname(os.path.abspath(args.log_dir))

    # Load datasets
    df_train = Dataset("./datasets/DEARM/Train_Dataset_DREAM.parquet").get_dataframe()
    df_val = Dataset("./datasets/DEARM/Val_Dataset_DREAM.csv").get_dataframe()

    # Select fingerprint type
    selected_fps = args.fps_type
    if selected_fps not in ["MACCS", "RDK", "AVALON", "ATOMPAIR"]:
        raise ValueError(f"Unsupported fingerprint type: {selected_fps}")

    # Process data
    TrainData = ProcessData(df_train, selected_fps).get_data()
    ValData = ProcessData(df_val, selected_fps).get_data()
    TrainLabel = df_train["LABEL"]
    DevLabel = df_val["LABEL"]

    # Train the model using FLAML AutoML
    automl = AutoML()
    automl_settings = {
        "time_budget": 3600,
        "metric": "ap",  # Use average precision (PRAUC) as the metric
        "task": "classification",
        "estimator_list": [args.model_type],
        "log_file_name": os.path.join(args.log_dir, "flaml_experiment.log"),
        "eval_method": "cv",
        "n_splits": 5,  # Replaced 'split_type' with 'n_splits' for proper stratified CV
        "seed": 42,
    }
    automl.fit(X_train=TrainData, y_train=TrainLabel, **automl_settings)

    # Save the model
    import pickle

    best_model_path = os.path.join(args.log_dir, "best_model.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    # Evaluate the final model
    probabilities = automl.predict_proba(ValData)[:, 1]
    predictions = automl.predict(ValData)

    # Calculate PRAUC
    prauc = average_precision_score(DevLabel, probabilities)
    print(f"PRAUC on validation set: {prauc:.4f}")

    # Calculate ROC-AUC
    roc_auc = roc_auc_score(DevLabel, probabilities)
    print(f"ROC-AUC on validation set: {roc_auc:.4f}")

    # Update the results CSV file
    update_results_csv(parent_dir, args.model_type, args.fps_type, prauc, roc_auc)

    # Save predictions
    df_predictions_val = pd.DataFrame(
        {
            "SMILES": df_val["SMILES"],
            "PredictedScore": probabilities,
            "PredictedLabel": predictions,
        }
    )
    predictions_path = os.path.join(args.log_dir, "val_predictions.csv")
    df_predictions_val.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")


# Argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model for DREAM Challenge 2025"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory to save logs and model checkpoints",
    )
    parser.add_argument(
        "--fps_type",
        type=str,
        required=True,
        help="Fingerprint type: ['MACCS', 'RDK', 'AVALON', 'ATOMPAIR']",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type: 'lgbm' (LightGBM), 'xgboost' (XGBoost with default settings), "
        "'xgb_limitdepth' (XGBoost with max_depth parameter), 'rf' (Random Forest), "
        "'extra_tree' (Extra Trees Classifier), 'histgb' (Histogram-based Gradient Boosting), "
        "'lrl1' (Logistic Regression with L1 regularization), 'lrl2' (Logistic Regression with L2 regularization), "
        "'catboost' (CatBoost Classifier), 'kneighbor' (K-Nearest Neighbors)",
    )
    args = parser.parse_args()

    main(args)
