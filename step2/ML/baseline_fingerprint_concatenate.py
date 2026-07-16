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
)
import datetime
import pickle
import numpy as np
from itertools import combinations


def concatenate_fingerprints(df, fps_types):
    """
    Concatenate multiple fingerprint types for a given dataframe.
    
    Args:
        df: DataFrame containing the data
        fps_types: List of fingerprint types to concatenate
    
    Returns:
        Concatenated fingerprint features
    """
    concatenated_data = []
    
    for fps_type in fps_types:
        data = ProcessData(df, fps_type).get_data()
        concatenated_data.append(data)
    
    # Concatenate along feature axis (axis=1)
    return np.concatenate(concatenated_data, axis=1)


def update_results_csv(parent_dir, model_type, fps_combination, prauc, roc_auc, metric,
                       hits_50=None, clusters_50=None, cluster_prauc_50=None,
                       hits_200=None, clusters_200=None, cluster_prauc_200=None,
                       hits_500=None, clusters_500=None, cluster_prauc_500=None):
    # Define the CSV file path
    results_file = os.path.join(parent_dir, "ensemble_model_results.csv")

    # Prepare the new data row
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {
        "Timestamp": timestamp,
        "ModelType": model_type,
        "FingerprintCombination": fps_combination,
        "OptimizationMetric": metric,
        "PRAUC": prauc,
        "ROC-AUC": roc_auc,
    }
    
    # Add cluster metrics if provided
    if hits_50 is not None:
        new_data["Hits_Top50"] = hits_50
    if clusters_50 is not None:
        new_data["Clusters_Top50"] = clusters_50
    if cluster_prauc_50 is not None:
        new_data["ClusterPRAUC_Top50"] = cluster_prauc_50
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


def calculate_cluster_metrics(df_val, probabilities, top_n, all_clusters):
    """Calculate cluster-based metrics for top N compounds."""
    # Create sorted index array based on probabilities (descending)
    sorted_indices = probabilities.argsort()[::-1]
    
    # Create selection for top N
    selection = df_val.iloc[sorted_indices[:top_n]].copy()
    selection["Score"] = probabilities[sorted_indices[:top_n]]
    
    # Calculate hits and clusters
    hits = selection[selection["LABEL"] == 1]
    n_hits = hits.shape[0]
    clusters = hits.drop_duplicates("CLUSTER_LABEL").shape[0]
    
    # Calculate cluster PRAUC
    cluster_prauc = None
    if clusters > 1:
        cluster_recall = []
        cluster_precision = []
        
        for th in sorted(hits["Score"].unique(), reverse=True):
            found = hits[hits["Score"] >= th].drop_duplicates("CLUSTER_LABEL").shape[0]
            cluster_recall.append(found/all_clusters)
            selected = selection[selection["Score"] >= th].shape[0]
            cluster_precision.append(found/selected if selected > 0 else 0)
        
        if len(cluster_recall) >= 2:
            cluster_prauc = auc(cluster_recall, cluster_precision)
        else:
            cluster_prauc = cluster_precision[0] if cluster_precision else 0
    elif clusters == 1:
        th = hits["Score"].min()
        selected = selection[selection["Score"] >= th].shape[0]
        cluster_prauc = 1/selected if selected > 0 else 0
    elif clusters == 0:
        cluster_prauc = 0
    
    return n_hits, clusters, cluster_prauc


def train_and_evaluate_combination(fps_combination, df_train, df_val, args, parent_dir):
    """Train and evaluate a single fingerprint combination."""
    print(f"\n{'='*60}")
    print(f"Evaluating combination: {','.join(fps_combination)}")
    print(f"{'='*60}")
    
    # Create combination string for file naming
    combo_str = "_".join(fps_combination)
    
    # Create log directory for this combination
    combo_log_dir = os.path.join(args.log_dir, f"combo_{combo_str}")
    os.makedirs(combo_log_dir, exist_ok=True)
    
    try:
        # Process data with concatenated fingerprints
        print("Processing fingerprints...")
        TrainData = concatenate_fingerprints(df_train, fps_combination)
        ValData = concatenate_fingerprints(df_val, fps_combination)
        TrainLabel = df_train["LABEL"]
        DevLabel = df_val["LABEL"]
        
        print(f"Training data shape: {TrainData.shape}")
        print(f"Validation data shape: {ValData.shape}")
        
        # Train the model using FLAML AutoML
        print("Training model...")
        automl = AutoML()
        automl_settings = {
            "time_budget": args.time_budget,
            "metric": "roc_auc_weighted",
            "task": "classification",
            "estimator_list": [args.model_type],
            "log_file_name": os.path.join(combo_log_dir, "flaml_experiment.log"),
            "eval_method": "cv",
            "n_splits": 5,
            "seed": 42,
        }
        
        automl.fit(X_train=TrainData, y_train=TrainLabel, **automl_settings)
        
        # Save the model
        best_model_path = os.path.join(combo_log_dir, "best_model.pkl")
        with open(best_model_path, "wb") as f:
            pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
        
        # Evaluate the final model
        print("Evaluating model...")
        probabilities = automl.predict_proba(ValData)[:, 1]
        predictions = automl.predict(ValData)
        
        # Calculate PRAUC and ROC-AUC
        prauc = average_precision_score(DevLabel, probabilities)
        roc_auc = roc_auc_score(DevLabel, probabilities)
        
        print(f"PRAUC: {prauc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Calculate cluster metrics
        all_clusters = df_val[df_val["LABEL"] == 1].drop_duplicates("CLUSTER_LABEL").shape[0]
        
        # Calculate metrics for different top N values
        n_hits_50, clusters_50, cluster_prauc_50 = calculate_cluster_metrics(df_val, probabilities, 50, all_clusters)
        n_hits_200, clusters_200, cluster_prauc_200 = calculate_cluster_metrics(df_val, probabilities, 200, all_clusters)
        n_hits_500, clusters_500, cluster_prauc_500 = calculate_cluster_metrics(df_val, probabilities, 500, all_clusters)
        
        print(f"Top 50 - Hits: {n_hits_50}, Clusters: {clusters_50}, Cluster PRAUC: {cluster_prauc_50:.4f}")
        print(f"Top 200 - Hits: {n_hits_200}, Clusters: {clusters_200}, Cluster PRAUC: {cluster_prauc_200:.4f}")
        print(f"Top 500 - Hits: {n_hits_500}, Clusters: {clusters_500}, Cluster PRAUC: {cluster_prauc_500:.4f}")
        
        # Update results CSV
        combo_name = ",".join(fps_combination)
        update_results_csv(
            parent_dir, args.model_type, combo_name, prauc, roc_auc, "roc_auc_weighted",
            n_hits_50, clusters_50, cluster_prauc_50,
            n_hits_200, clusters_200, cluster_prauc_200,
            n_hits_500, clusters_500, cluster_prauc_500
        )
        
        # Save predictions
        df_predictions_val = pd.DataFrame({
            "SMILES": df_val["SMILES"],
            "PredictedScore": probabilities,
            "PredictedLabel": predictions,
        })
        predictions_path = os.path.join(combo_log_dir, "val_predictions.csv")
        df_predictions_val.to_csv(predictions_path, index=False)
        
        return {
            'combination': combo_name,
            'prauc': prauc,
            'roc_auc': roc_auc,
            'hits_50': n_hits_50,
            'clusters_50': clusters_50,
            'cluster_prauc_50': cluster_prauc_50
        }
        
    except Exception as e:
        print(f"Error processing combination {combo_str}: {str(e)}")
        return None


def main(args):
    # Ensure log_dir exists
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Get the parent directory of log_dir
    parent_dir = os.path.dirname(os.path.abspath(args.log_dir))
    
    # Load datasets
    print("Loading datasets...")
    df_train = Dataset("../../datasets/DREAM/Train_Dataset_DREAM.parquet").get_dataframe()
    # Check and standardize the LABEL column name
    if 'Label' in df_train.columns and 'LABEL' not in df_train.columns:
        df_train.rename(columns={'Label': 'LABEL'}, inplace=True)
    elif 'LABEL' not in df_train.columns and 'Label' not in df_train.columns:
        raise ValueError("Neither 'LABEL' nor 'Label' column found in training data")
    df_val = Dataset("../../datasets/DREAM/Val_Dataset_DREAM_Step2.csv").get_dataframe()
    
    print("------------------------------------------------------------")
    print("Number of binders in the training set:", (df_train["LABEL"] == 1).sum())
    print("Number of non-binders in the training set:", (df_train["LABEL"] == 0).sum())
    print("------------------------------------------------------------")
    print("Number of binders in the validation set:", (df_val["LABEL"] == 1).sum())
    print("Number of non-binders in the validation set:", (df_val["LABEL"] == 0).sum())
    print("------------------------------------------------------------")
    
    # Check if this is a combination run
    if "," in args.fps_type:
        # This is a combination run - evaluate ONLY the exact combination specified
        fps_types = [fp.strip() for fp in args.fps_type.split(',')]
        
        # Validate fingerprint types
        valid_fps = ["MACCS", "RDK", "AVALON", "ATOMPAIR"]
        for fp in fps_types:
            if fp not in valid_fps:
                raise ValueError(f"Unsupported fingerprint type: {fp}. Valid types: {valid_fps}")
        
        print(f"Evaluating exact fingerprint combination: {fps_types} with {args.model_type}")
        
        # Evaluate only the specified combination
        result = train_and_evaluate_combination(tuple(fps_types), df_train, df_val, args, parent_dir)
        
        if result:
            print(f"\n{'='*80}")
            print(f"RESULT FOR FINGERPRINT COMBINATION: {result['combination']} WITH {args.model_type}")
            print(f"{'='*80}")
            print(f"PRAUC: {result['prauc']:.4f}")
            print(f"ROC-AUC: {result['roc_auc']:.4f}")
            print(f"Top 50 - Hits: {result['hits_50']}, Clusters: {result['clusters_50']}")
        else:
            print("Combination evaluation failed.")
    
    else:
        # Single fingerprint type - use original logic
        selected_fps = args.fps_type
        if selected_fps not in ["MACCS", "RDK", "AVALON", "ATOMPAIR"]:
            raise ValueError(f"Unsupported fingerprint type: {selected_fps}")

        print(f"Single fingerprint processing with {args.model_type}: {selected_fps}")
        
        # Evaluate the single fingerprint type
        result = train_and_evaluate_combination((selected_fps,), df_train, df_val, args, parent_dir)
        
        if result:
            print(f"\n{'='*80}")
            print(f"RESULT FOR SINGLE FINGERPRINT: {result['combination']} WITH {args.model_type}")
            print(f"{'='*80}")
            print(f"PRAUC: {result['prauc']:.4f}")
            print(f"ROC-AUC: {result['roc_auc']:.4f}")
            print(f"Top 50 - Hits: {result['hits_50']}, Clusters: {result['clusters_50']}")
        else:
            print("Single fingerprint evaluation failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ensemble models with fingerprint combinations for DREAM Challenge 2025"
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
        help="Fingerprint type: single ['MACCS', 'RDK', 'AVALON', 'ATOMPAIR'] or comma-separated combinations (e.g., 'MACCS,RDK,AVALON,ATOMPAIR')",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type: 'lgbm', 'xgboost', 'xgb_limitdepth', 'rf', 'extra_tree', "
        "'histgb', 'lrl1', 'lrl2', 'catboost', 'kneighbor'",
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        default=3600,
        help="Time budget for each model training (seconds). Default: 3600 (1 hour)",
    )
    args = parser.parse_args()
    
    main(args)