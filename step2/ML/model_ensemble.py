from helper import Dataset, ProcessData
import argparse
import os
import numpy as np
import pandas as pd
from flaml.automl.automl import AutoML
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import datetime
import pickle


def load_pretrained_model(model_type, fingerprint_combo):
    """Load a pretrained model from the specified path"""
    model_path = f"../../runs/DREAM/step2/{model_type}/{fingerprint_combo}/best_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        automl = pickle.load(f)
    
    return automl


def calculate_cluster_metrics(predictions, df_val, top_k):
    """Calculate cluster metrics for a given model's predictions"""
    # Create sorted index array based on probabilities (descending)
    sorted_indices = predictions.argsort()[::-1]
    
    # Create selection for top k
    selection = df_val.iloc[sorted_indices[:top_k]].copy()
    selection["Score"] = predictions[sorted_indices[:top_k]]
    
    # Calculate hits and unique clusters
    hits = selection[selection["LABEL"] == 1]
    n_hits = hits.shape[0]
    clusters = hits.drop_duplicates("CLUSTER_LABEL").shape[0]
    
    return n_hits, clusters


def calculate_model_weights(predictions_array, val_y, df_val, method="weighted_clusters"):
    """Calculate weights for models based on hierarchical cluster performance"""
    weights = []
    cluster_metrics = []
    
    # First, calculate all cluster metrics for each model
    for i in range(predictions_array.shape[0]):
        _, clusters_50 = calculate_cluster_metrics(predictions_array[i], df_val, 50)
        _, clusters_200 = calculate_cluster_metrics(predictions_array[i], df_val, 200)
        _, clusters_500 = calculate_cluster_metrics(predictions_array[i], df_val, 500)
        cluster_metrics.append((clusters_50, clusters_200, clusters_500))
    
    if method == "weighted_clusters":
        # Hierarchical scoring based on Top50 > Top200 > Top500 priority
        scores = []
        
        for i, (c50, c200, c500) in enumerate(cluster_metrics):
            # Create a hierarchical score: prioritize Top50, then Top200, then Top500
            # Use a large multiplier to ensure higher priority metrics dominate
            score = c50 * 1000000 + c200 * 1000 + c500
            scores.append(score)
        
        weights = scores
        print(f"Cluster metrics (Top50, Top200, Top500): {cluster_metrics}")
        print(f"Hierarchical scores: {scores}")
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights to sum to 1
    weights = np.array(weights, dtype=float)
    if np.sum(weights) == 0:
        # If all weights are 0, use equal weights
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights = weights / np.sum(weights)
    
    print(f"Model weights ({method}): {weights}")
    return weights


def ensemble_predictions(predictions_array, method, val_y=None, df_val=None, train_predictions=None, train_y=None):
    """
    Ensemble predictions using different methods
    
    Args:
        predictions_array: Array of shape (n_models, n_samples) with model predictions
        method: Ensemble method to use
        val_y: True labels for validation set (needed for weighted methods)
        df_val: Validation dataframe (needed for cluster-based weighting)
        train_predictions: Training predictions for stacking (shape: n_models, n_train_samples)
        train_y: Training labels for stacking
    """
    
    if method == "simple_average":
        ensemble_probs = np.mean(predictions_array, axis=0)
        
    elif method == "weighted_average_clusters":
        if val_y is None or df_val is None:
            raise ValueError("val_y and df_val required for cluster-based weighted averaging")
        weights = calculate_model_weights(predictions_array, val_y, df_val, method="weighted_clusters")
        ensemble_probs = np.average(predictions_array, axis=0, weights=weights)
        
    elif method == "majority_vote":
        # Convert probabilities to hard predictions and take majority vote
        hard_predictions = (predictions_array > 0.5).astype(int)
        ensemble_probs = np.mean(hard_predictions, axis=0)
        
    elif method == "max_vote":
        # Take maximum probability across all models
        ensemble_probs = np.max(predictions_array, axis=0)
        
    elif method == "min_vote":
        # Take minimum probability across all models
        ensemble_probs = np.min(predictions_array, axis=0)
        
    elif method == "median_vote":
        # Take median probability across all models
        ensemble_probs = np.median(predictions_array, axis=0)
        
    elif method == "stacking_logistic":
        if train_predictions is None or train_y is None:
            raise ValueError("Training data required for stacking")
        
        # Train meta-learner on training predictions
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        meta_learner.fit(train_predictions.T, train_y)  # Transpose to get (n_samples, n_models)
        
        # Get ensemble predictions
        ensemble_probs = meta_learner.predict_proba(predictions_array.T)[:, 1]
        
    elif method == "stacking_rf":
        if train_predictions is None or train_y is None:
            raise ValueError("Training data required for stacking")
        
        # Train meta-learner on training predictions
        meta_learner = RandomForestClassifier(n_estimators=100, random_state=42)
        meta_learner.fit(train_predictions.T, train_y)  # Transpose to get (n_samples, n_models)
        
        # Get ensemble predictions
        ensemble_probs = meta_learner.predict_proba(predictions_array.T)[:, 1]
        
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_probs


def prediction_level_ensemble(df_train, df_val, log_dir, ensemble_method):
    """
    Ensemble the four specific pretrained models at prediction level.
    """
    
    # Define the four specific models with their fingerprint combinations
    model_specs = [
        ("histgb", "RDK_AVALON_ATOMPAIR"),
        ("lgbm", "MACCS_RDK_AVALON"), 
        ("xgb_limitdepth", "MACCS_ATOMPAIR"),
        ("xgboost", "MACCS_RDK_AVALON_ATOMPAIR")
    ]
    
    def concatenate_fingerprints(df, fps_types):
        """
        Concatenate multiple fingerprint types for a given dataframe.
        Same logic as in baseline_fingerprint_concatenate.py
        """
        concatenated_data = []
        
        for fps_type in fps_types:
            data = ProcessData(df, fps_type).get_data()
            concatenated_data.append(data)
        
        # Concatenate along feature axis (axis=1)
        return np.concatenate(concatenated_data, axis=1)
    
    model_predictions_val = []
    model_predictions_train = []
    model_info = []
    
    for model_type, fingerprint_combo in model_specs:
        try:
            # Load the model
            model = load_pretrained_model(model_type, fingerprint_combo)
            
            # Parse fingerprint combination to get individual fingerprint types
            fps_types = fingerprint_combo.split("_")
            
            # Prepare validation data using concatenated fingerprints
            # Same approach as baseline_fingerprint_concatenate.py
            val_X = concatenate_fingerprints(df_val, fps_types)
            val_predictions = model.predict_proba(val_X)[:, 1]
            model_predictions_val.append(val_predictions)
            
            # Prepare training data if needed for stacking
            if ensemble_method.startswith("stacking"):
                train_X = concatenate_fingerprints(df_train, fps_types)
                train_predictions = model.predict_proba(train_X)[:, 1]
                model_predictions_train.append(train_predictions)
            
            model_info.append((model_type, fingerprint_combo))
            print(f"Successfully loaded and got predictions from {model_type} with {fingerprint_combo}")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        except Exception as e:
            print(f"Warning: Error processing model {model_type} with {fingerprint_combo}: {e}")
            continue
    
    if not model_predictions_val:
        print("No valid model predictions could be obtained. Skipping.")
        return None
    
    print(f"Successfully obtained predictions from {len(model_predictions_val)} models")
    
    # Convert to numpy arrays for easier manipulation
    predictions_array_val = np.array(model_predictions_val)  # Shape: (n_models, n_samples)
    predictions_array_train = np.array(model_predictions_train) if model_predictions_train else None
    
    # Get validation and training labels
    val_y = df_val["LABEL"].values
    train_y = df_train["LABEL"].values if predictions_array_train is not None else None
    
    print(f"Using ensemble method: {ensemble_method}")
    
    # Generate ensemble predictions
    ensemble_probabilities = ensemble_predictions(
        predictions_array_val, 
        ensemble_method, 
        val_y=val_y,
        df_val=df_val,
        train_predictions=predictions_array_train,
        train_y=train_y
    )
    
    ensemble_predictions_binary = (ensemble_probabilities > 0.5).astype(int)
    
    print(f"Ensemble predictions shape: {ensemble_probabilities.shape}")
    
    # Calculate metrics
    prauc = average_precision_score(val_y, ensemble_probabilities)
    print(f"PRAUC on validation set: {prauc:.4f}")

    roc_auc = roc_auc_score(val_y, ensemble_probabilities)
    print(f"ROC-AUC on validation set: {roc_auc:.4f}")

    # Get the number of all clusters with positive labels
    all_clusters = df_val[df_val["LABEL"] == 1].drop_duplicates("CLUSTER_LABEL").shape[0]
    
    # Create sorted index array based on probabilities (descending)
    sorted_indices = ensemble_probabilities.argsort()[::-1]
    
    # Create selections at different thresholds
    selection_50 = df_val.iloc[sorted_indices[:50]].copy()
    selection_50["Score"] = ensemble_probabilities[sorted_indices[:50]]
    
    selection_200 = df_val.iloc[sorted_indices[:200]].copy()
    selection_200["Score"] = ensemble_probabilities[sorted_indices[:200]]
    
    selection_500 = df_val.iloc[sorted_indices[:500]].copy()
    selection_500["Score"] = ensemble_probabilities[sorted_indices[:500]]
    
    # Calculate metrics for different thresholds
    def calculate_metrics(selection, threshold_name):
        hits = selection[selection["LABEL"] == 1]
        n_hits = hits.shape[0]
        clusters = hits.drop_duplicates("CLUSTER_LABEL").shape[0]
        
        print(f"Top {threshold_name} selection - Hits: {n_hits}, Unique clusters: {clusters}")
        
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
                print(f"Cluster PRAUC for top {threshold_name}: {cluster_prauc:.4f}")
            else:
                cluster_prauc = cluster_precision[0] if cluster_precision else 0
                print(f"Using single point precision for top {threshold_name}: {cluster_prauc:.4f}")
        elif clusters == 1:
            th = hits["Score"].min()
            selected = selection[selection["Score"] >= th].shape[0]
            cluster_prauc = 1/selected if selected > 0 else 0
            print(f"Cluster PRAUC for top {threshold_name} (single cluster): {cluster_prauc:.4f}")
        else:
            cluster_prauc = 0
            print(f"No clusters found in top {threshold_name} selection. Cluster PRAUC set to 0.")
        
        return n_hits, clusters, cluster_prauc
    
    # Calculate metrics for all thresholds
    n_hits_50, clusters_50, cluster_prauc_50 = calculate_metrics(selection_50, "50")
    n_hits_200, clusters_200, cluster_prauc_200 = calculate_metrics(selection_200, "200")
    n_hits_500, clusters_500, cluster_prauc_500 = calculate_metrics(selection_500, "500")
    
    print(f"All positive clusters in validation set: {all_clusters}")

    # Save predictions
    df_predictions_val = pd.DataFrame(
        {
            "SMILES": df_val["SMILES"],
            "PredictedScore": ensemble_probabilities,
            "PredictedLabel": ensemble_predictions_binary,
        }
    )
    predictions_path = os.path.join(log_dir, "val_predictions.csv")
    df_predictions_val.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Save individual model predictions for analysis
    individual_predictions_df = pd.DataFrame({
        "SMILES": df_val["SMILES"],
        "TrueLabel": val_y,
        "EnsemblePrediction": ensemble_probabilities,
    })
    
    # Add individual model predictions as columns
    for i, (model_type, fingerprint_combo) in enumerate(model_info):
        individual_predictions_df[f"{model_type}_{fingerprint_combo}"] = predictions_array_val[i]
    
    individual_path = os.path.join(log_dir, "individual_predictions.csv")
    individual_predictions_df.to_csv(individual_path, index=False)
    print(f"Individual predictions saved to {individual_path}")
    
    # Save ensemble info
    ensemble_info = {
        "ensemble_method": ensemble_method,
        "models": model_info,
        "n_models": len(model_info),
        "prauc": prauc,
        "roc_auc": roc_auc,
    }
    
    ensemble_info_path = os.path.join(log_dir, "ensemble_info.pkl")
    with open(ensemble_info_path, "wb") as f:
        pickle.dump(ensemble_info, f, pickle.HIGHEST_PROTOCOL)
    print(f"Ensemble info saved to {ensemble_info_path}")
    
    # Return comprehensive metrics
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ensemble_method": ensemble_method,
        "n_models": len(model_info),
        "models_used": [f"{model_type}_{fingerprint_combo}" for model_type, fingerprint_combo in model_info],
        "prauc": prauc,
        "roc_auc": roc_auc,
        "hits_50": n_hits_50,
        "clusters_50": clusters_50,
        "cluster_prauc_50": cluster_prauc_50,
        "hits_200": n_hits_200,
        "clusters_200": clusters_200,
        "cluster_prauc_200": cluster_prauc_200,
        "hits_500": n_hits_500,
        "clusters_500": clusters_500,
        "cluster_prauc_500": cluster_prauc_500,
    }


def main(args):
    # Ensure log_dir exists
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Get the parent directory of log_dir for the summary CSV
    parent_dir = os.path.dirname(os.path.abspath(args.log_dir))

    # Load datasets
    df_train = Dataset("../../datasets/DREAM/Train_Dataset_DREAM.parquet").get_dataframe()
    df_val = Dataset("../../datasets/DREAM/Val_Dataset_DREAM_Step2.csv").get_dataframe()

    print("------------------------------------------------------------")
    print("Number of binders in the training set:", (df_train["LABEL"] == 1).sum())
    print("Number of non-binders in the training set:", (df_train["LABEL"] == 0).sum())
    print("------------------------------------------------------------")
    print("Number of binders in the validation set:", (df_val["LABEL"] == 1).sum())
    print("Number of non-binders in the validation set:", (df_val["LABEL"] == 0).sum())
    print("------------------------------------------------------------")

    print("Using the following four pretrained models:")
    print("  1. histgb with RDK_AVALON_ATOMPAIR")
    print("  2. lgbm with MACCS_RDK_AVALON")
    print("  3. xgb_limitdepth with MACCS_ATOMPAIR")
    print("  4. xgboost with MACCS_RDK_AVALON_ATOMPAIR")
    
    print(f"Using ensemble method: {args.ensemble_method}")
    
    # Train and evaluate the ensemble
    results = prediction_level_ensemble(
        df_train, 
        df_val, 
        args.log_dir,
        args.ensemble_method
    )
    
    if results:
        print(f"\n=== ENSEMBLE RESULTS ===")
        print(f"Ensemble method: {results['ensemble_method']}")
        print(f"Number of models: {results['n_models']}")
        print(f"Models used: {', '.join(results['models_used'])}")
        print(f"PRAUC: {results['prauc']:.4f}")
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
        print(f"Hits@50: {results['hits_50']}, Clusters@50: {results['clusters_50']}")
        print(f"Hits@200: {results['hits_200']}, Clusters@200: {results['clusters_200']}")
        print(f"Hits@500: {results['hits_500']}, Clusters@500: {results['clusters_500']}")
        
        # Save results summary
        results_df = pd.DataFrame([results])
        summary_path = os.path.join(args.log_dir, "ensemble_summary.csv")
        results_df.to_csv(summary_path, index=False)
        print(f"Results summary saved to {summary_path}")
        
        # Update master CSV in parent directory
        update_master_results_csv(parent_dir, results)
        
    else:
        print("Ensemble evaluation failed.")

    print(f"\nEnsemble evaluation completed.")


def update_master_results_csv(parent_dir, results):
    """Update the master CSV file with ensemble results"""
    # Define the CSV file path
    results_file = os.path.join(parent_dir, "ensemble_methods_results.csv")
    
    # Prepare the new data row with all metrics
    new_data = {
        "Timestamp": results["timestamp"],
        "EnsembleMethod": results["ensemble_method"],
        "NumModels": results["n_models"],
        "ModelsUsed": ", ".join(results["models_used"]),
        "PRAUC": results["prauc"],
        "ROC_AUC": results["roc_auc"],
        "Hits_50": results["hits_50"],
        "Clusters_50": results["clusters_50"],
        "ClusterPRAUC_50": results["cluster_prauc_50"],
        "Hits_200": results["hits_200"],
        "Clusters_200": results["clusters_200"],
        "ClusterPRAUC_200": results["cluster_prauc_200"],
        "Hits_500": results["hits_500"],
        "Clusters_500": results["clusters_500"],
        "ClusterPRAUC_500": results["cluster_prauc_500"],
    }
    
    # Check if the file exists
    if os.path.exists(results_file):
        # Load existing data
        df_results = pd.read_csv(results_file)
        # Append new data
        df_results = pd.concat([df_results, pd.DataFrame([new_data])], ignore_index=True)
    else:
        # Create new dataframe
        df_results = pd.DataFrame([new_data])
    
    # Save the updated dataframe
    df_results.to_csv(results_file, index=False)
    print(f"Master results updated in {results_file}")
    
    # Print summary of all methods evaluated so far
    if len(df_results) > 1:
        print(f"\n=== MASTER RESULTS SUMMARY ({len(df_results)} methods evaluated) ===")
        
        # Sort by PRAUC for display
        df_sorted = df_results.sort_values('PRAUC', ascending=False)
        
        print("Ranking by PRAUC:")
        for idx, row in df_sorted.iterrows():
            print(f"  {idx+1}. {row['EnsembleMethod']}: PRAUC={row['PRAUC']:.4f}, "
                  f"Clusters@50={row['Clusters_50']}, ROC-AUC={row['ROC_AUC']:.4f}")
        
        # Also show best by Clusters@50
        best_clusters = df_results.loc[df_results['Clusters_50'].idxmax()]
        print(f"\nBest Clusters@50: {best_clusters['EnsembleMethod']} "
              f"({best_clusters['Clusters_50']} clusters, PRAUC={best_clusters['PRAUC']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train prediction-level ensemble model using four specific pretrained models"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory to save logs and model checkpoints",
    )
    parser.add_argument(
        "--ensemble_method",
        type=str,
        default="simple_average",
        choices=[
            "simple_average", 
            "weighted_average_clusters",
            "majority_vote", 
            "max_vote", 
            "min_vote", 
            "median_vote",
            "stacking_logistic", 
            "stacking_rf"
        ],
        help="Ensemble method to use",
    )
    args = parser.parse_args()
    
    main(args)