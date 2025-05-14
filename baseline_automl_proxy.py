from helper import Dataset, ProcessData
import argparse
import os
import pandas as pd
from flaml.automl.automl import AutoML
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)  # PRAUC and ROC-AUC metrics
import datetime

def update_results_csv(parent_dir, model_type, fps_type, prauc, roc_auc, metric,
                       hits_200=None, clusters_200=None, cluster_prauc_200=None,
                       hits_500=None, clusters_500=None, cluster_prauc_500=None):
    # Define the CSV file path
    results_file = os.path.join(parent_dir, "model_results.csv")

    # Prepare the new data row
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {
        "Timestamp": timestamp,
        "ModelType": model_type,
        "FingerprintType": fps_type,
        "OptimizationMetric": metric,
        "PRAUC": prauc,
        "ROC-AUC": roc_auc,
    }
    
    # Add cluster metrics if provided
    if hits_200 is not None:
        new_data["Hits_Top200"] = hits_200
    if clusters_200 is not None:
        new_data["Clusters_Top200"] = clusters_200
    if cluster_prauc_200 is not None:
        new_data["ClusterPRAUC_Top200"] = cluster_prauc_200
    if hits_500 is not None:
        new_data["Hits_Top500"] = hits_500
    if clusters_500 is not None:
        new_data["Clusters_Top500"] = clusters_500
    if cluster_prauc_500 is not None:
        new_data["ClusterPRAUC_Top500"] = cluster_prauc_500

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
    df_train = Dataset("./datasets/DREAM/LRRK2_DEL.parquet").get_dataframe()
    df_val = Dataset("./datasets/DREAM/LRRK2_ASMS_clustered.csv").get_dataframe()

    print("Number of binders in the training set:", (df_train["LABEL"] == 1).sum())
    print("Number of non-binders in the training set:", (df_train["LABEL"] == 0).sum())
    print("------------------------------------------------------------")
    print("Number of binders in the validation set:", (df_val["LABEL"] == 1).sum())
    print("Number of non-binders in the validation set:", (df_val["LABEL"] == 0).sum())

    # Select fingerprint type
    selected_fps = args.fps_type
    if selected_fps not in ["MACCS", "RDK", "AVALON", "ATOMPAIR"]:
        raise ValueError(f"Unsupported fingerprint type: {selected_fps}")

    # Process data
    TrainData = ProcessData(df_train, selected_fps).get_data()
    ValData = ProcessData(df_val, selected_fps).get_data()
    TrainLabel = df_train["LABEL"]
    ValLabel = df_val["LABEL"]
    print("TrainLabel shape:", TrainLabel.shape)
    print("ValLabel shape:", ValLabel.shape)

    # Train the model using FLAML AutoML
    automl = AutoML()
    automl_settings = {
        "time_budget": 100,
        "metric": args.metric,  # Use the metric provided via command-line
        "task": "classification",
        "estimator_list": [args.model_type],
        "log_file_name": os.path.join(args.log_dir, "flaml_experiment.log"),
        "eval_method": "cv",
        "n_splits": 5,
        "seed": 42,
    }
    automl.fit(X_train=TrainData, y_train=TrainLabel, **automl_settings)

    # No need for saving the model
    # import pickle

    # best_model_path = os.path.join(args.log_dir, "best_model.pkl")
    # with open(best_model_path, "wb") as f:
    #     pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    # Evaluate the final model
    probabilities = automl.predict_proba(ValData)[:, 1]
    predictions = automl.predict(ValData)

    # Calculate PRAUC
    prauc = average_precision_score(ValLabel, probabilities)
    print(f"PRAUC on validation set: {prauc:.4f}")

    # Replace the section after calculating ROC-AUC with:
    # Calculate ROC-AUC
    roc_auc = roc_auc_score(ValLabel, probabilities)
    print(f"ROC-AUC on validation set: {roc_auc:.4f}")

    # Calculate cluster-based metrics
    # Get the number of all clusters with positive labels
    all_clusters = df_val[df_val["LABEL"] == 1].drop_duplicates("CLUSTER_LABEL").shape[0]
    
    # Create sorted index array based on probabilities (descending)
    sorted_indices = probabilities.argsort()[::-1]
    
    # Create selections at different thresholds
    selection_200 = df_val.iloc[sorted_indices[:200]].copy()
    selection_200["Score"] = probabilities[sorted_indices[:200]]
    
    selection_500 = df_val.iloc[sorted_indices[:500]].copy()
    selection_500["Score"] = probabilities[sorted_indices[:500]]
    
    # Calculate metrics for top 200 compounds
    hits_200 = selection_200[selection_200["LABEL"] == 1]
    n_hits_200 = hits_200.shape[0]
    clusters_200 = hits_200.drop_duplicates("CLUSTER_LABEL").shape[0]
    
    # Calculate metrics for top 500 compounds
    hits_500 = selection_500[selection_500["LABEL"] == 1]
    n_hits_500 = hits_500.shape[0]
    clusters_500 = hits_500.drop_duplicates("CLUSTER_LABEL").shape[0]
    
    print(f"All positive clusters in validation set: {all_clusters}")
    print(f"Top 200 selection - Hits: {n_hits_200}, Unique clusters: {clusters_200}")
    print(f"Top 500 selection - Hits: {n_hits_500}, Unique clusters: {clusters_500}")
    
    # Calculate cluster PRAUC for top 200
    cluster_prauc_200 = None
    if clusters_200 > 1:
        # Calculate cluster PRAUC
        cluster_recall_200 = []
        cluster_precision_200 = []
        
        for th in sorted(hits_200["Score"].unique(), reverse=True):
            found = hits_200[hits_200["Score"] >= th].drop_duplicates("CLUSTER_LABEL").shape[0]
            cluster_recall_200.append(found/all_clusters)
            selected = selection_200[selection_200["Score"] >= th].shape[0]
            cluster_precision_200.append(found/selected if selected > 0 else 0)
        
        cluster_prauc_200 = auc(cluster_recall_200, cluster_precision_200)
        print(f"Cluster PRAUC for top 200: {cluster_prauc_200:.4f}")
    elif clusters_200 == 1:
        # Single cluster case
        th = hits_200["Score"].min()
        selected = selection_200[selection_200["Score"] >= th].shape[0]
        cluster_prauc_200 = 1/selected if selected > 0 else 0
        print(f"Cluster PRAUC for top 200 (single cluster): {cluster_prauc_200:.4f}")
    
    # Calculate cluster PRAUC for top 500
    cluster_prauc_500 = None
    if clusters_500 > 1:
        # Calculate cluster PRAUC
        cluster_recall_500 = []
        cluster_precision_500 = []
        
        for th in sorted(hits_500["Score"].unique(), reverse=True):
            found = hits_500[hits_500["Score"] >= th].drop_duplicates("CLUSTER_LABEL").shape[0]
            cluster_recall_500.append(found/all_clusters)
            selected = selection_500[selection_500["Score"] >= th].shape[0]
            cluster_precision_500.append(found/selected if selected > 0 else 0)
        
        cluster_prauc_500 = auc(cluster_recall_500, cluster_precision_500)
        print(f"Cluster PRAUC for top 500: {cluster_prauc_500:.4f}")
    elif clusters_500 == 1:
        # Single cluster case
        th = hits_500["Score"].min()
        selected = selection_500[selection_500["Score"] >= th].shape[0]
        cluster_prauc_500 = 1/selected if selected > 0 else 0
        print(f"Cluster PRAUC for top 500 (single cluster): {cluster_prauc_500:.4f}")
    
    # Update the results CSV file with all metrics
    update_results_csv(
        parent_dir, args.model_type, args.fps_type, prauc, roc_auc, args.metric,
        n_hits_200, clusters_200, cluster_prauc_200,
        n_hits_500, clusters_500, cluster_prauc_500
    )

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
    parser.add_argument(
        "--metric",
        type=str,
        default="ap",
        help="Metric to optimize: 'ap' (Average Precision/PRAUC), 'roc_auc' (ROC-AUC), "
        "'accuracy', 'log_loss', 'f1', 'micro_f1', 'macro_f1', etc.",
    )
    args = parser.parse_args()
    main(args)
