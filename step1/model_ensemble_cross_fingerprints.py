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
from sklearn.model_selection import cross_val_score
import datetime
import pickle
from itertools import combinations


def load_pretrained_model(fingerprint_type, model_type):
    """Load a pretrained model from the specified path"""
    model_path = f"../runs/DREAM/DREAM_BASELINE_SEL/{fingerprint_type}_{model_type}/best_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        automl = pickle.load(f)
    
    return automl


def parse_model_spec(spec):
    """Parse a model specification, handling models with multiple underscores"""
    parts = spec.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid model specification format: {spec}")
    
    # First part is the fingerprint type
    fps_type = parts[0]
    
    # Everything after the first underscore is the model type
    model_type = "_".join(parts[1:])
    
    return fps_type, model_type


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


def prediction_level_ensemble(ensemble_specs, df_train, df_val, log_dir, ensemble_method,
                             model_combination_name=None):
    """
    Ensemble models at prediction level using various methods.
    Each model uses its own fingerprint type for predictions.
    """
    
    # Load pretrained models with their respective fingerprint types
    model_predictions_val = []
    model_predictions_train = []
    model_info = []
    
    for fps_type, model_type in ensemble_specs:
        try:
            # Load the model
            model = load_pretrained_model(fps_type, model_type)
            
            # Prepare validation data using the model's specific fingerprint type
            val_X = ProcessData(df_val, fps_type).get_data()
            val_predictions = model.predict_proba(val_X)[:, 1]
            model_predictions_val.append(val_predictions)
            
            # Prepare training data if needed for stacking
            if ensemble_method.startswith("stacking"):
                train_X = ProcessData(df_train, fps_type).get_data()
                train_predictions = model.predict_proba(train_X)[:, 1]
                model_predictions_train.append(train_predictions)
            
            model_info.append((fps_type, model_type))
            print(f"Successfully loaded and got predictions from {fps_type}_{model_type}")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        except Exception as e:
            print(f"Warning: Error processing model {fps_type}_{model_type}: {e}")
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

    # Create a directory for the specific combination if needed
    combo_log_dir = log_dir
    if model_combination_name:
        combo_log_dir = os.path.join(log_dir, model_combination_name)
        os.makedirs(combo_log_dir, exist_ok=True)
    
    # Save predictions
    df_predictions_val = pd.DataFrame(
        {
            "SMILES": df_val["SMILES"],
            "PredictedScore": ensemble_probabilities,
            "PredictedLabel": ensemble_predictions_binary,
        }
    )
    predictions_path = os.path.join(combo_log_dir, "val_predictions.csv")
    df_predictions_val.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Save individual model predictions for analysis
    individual_predictions_df = pd.DataFrame({
        "SMILES": df_val["SMILES"],
        "TrueLabel": val_y,
        "EnsemblePrediction": ensemble_probabilities,
    })
    
    # Add individual model predictions as columns
    for i, (fps_type, model_type) in enumerate(model_info):
        individual_predictions_df[f"{fps_type}_{model_type}"] = predictions_array_val[i]
    
    individual_path = os.path.join(combo_log_dir, "individual_predictions.csv")
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
    
    ensemble_info_path = os.path.join(combo_log_dir, "ensemble_info.pkl")
    with open(ensemble_info_path, "wb") as f:
        pickle.dump(ensemble_info, f, pickle.HIGHEST_PROTOCOL)
    print(f"Ensemble info saved to {ensemble_info_path}")
    
    # Return comprehensive metrics for summary with reordered columns
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "combo_size": None,  # Will be filled in main()
        "n_unique_models": None,  # Will be filled in main()
        "pairs_used": None,  # Will be filled in main()
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

    # Load datasets
    df_train = Dataset("../datasets/DREAM/Train_Dataset_DREAM.parquet").get_dataframe()
    df_val = Dataset("../datasets/DREAM/Val_Dataset_DREAM.csv").get_dataframe()

    print("------------------------------------------------------------")
    print("Number of binders in the training set:", (df_train["LABEL"] == 1).sum())
    print("Number of non-binders in the training set:", (df_train["LABEL"] == 0).sum())
    print("------------------------------------------------------------")
    print("Number of binders in the validation set:", (df_val["LABEL"] == 1).sum())
    print("Number of non-binders in the validation set:", (df_val["LABEL"] == 0).sum())
    print("------------------------------------------------------------")

    # Use the 8 specific low-correlation model pairs from our analysis
    # These are the pairs that ensure all 15 top models appear at least once
    predefined_pairs = [
        [("AVALON", "xgboost"), ("MACCS", "kneighbor")],
        [("AVALON", "rf"), ("MACCS", "kneighbor")],
        [("ATOMPAIR", "rf"), ("AVALON", "xgb_limitdepth")],
        [("AVALON", "histgb"), ("ATOMPAIR", "xgboost")],
        [("AVALON", "extra_tree"), ("ATOMPAIR", "histgb")],
        [("RDK", "catboost"), ("ATOMPAIR", "lgbm")],
        [("ATOMPAIR", "catboost"), ("RDK", "extra_tree")],
        [("ATOMPAIR", "lgbm"), ("RDK", "histgb")]
    ]
    
    print(f"Using {len(predefined_pairs)} predefined low-correlation pairs")
    print("Predefined pairs:")
    for i, pair in enumerate(predefined_pairs, 1):
        pair_str = " & ".join([f"{fps}_{model}" for fps, model in pair])
        print(f"  {i}. {pair_str}")
    
    print(f"Using ensemble method: {args.ensemble_method}")
    print(f"Maximum pairs per combination: {args.max_pairs}")
    
    # Initialize results list and summary file path
    all_results = []
    summary_path = os.path.join(args.log_dir, "combination_summary.csv")
    
    print(f"\nEvaluating all combinations from size 1 to {args.max_pairs}")
    
    for combo_size in range(1, min(args.max_pairs + 1, len(predefined_pairs) + 1)):
        print(f"\n=== Evaluating combinations of {combo_size} pair(s) ===")
        
        for combo in combinations(predefined_pairs, combo_size):
            # Flatten the combination to get all individual models
            all_models_in_combo = []
            pair_names = []
            
            for pair in combo:
                all_models_in_combo.extend(pair)
                pair_name = "_".join([f"{fps}_{model}" for fps, model in pair])
                pair_names.append(f"({pair_name})")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_models = []
            for model in all_models_in_combo:
                if model not in seen:
                    unique_models.append(model)
                    seen.add(model)
            
            # Create model combination name
            model_combination_name = f"PAIRS{'_'.join(pair_names)}"
            individual_models_name = "_".join([f"{fps}_{model}" for fps, model in unique_models])
            
            print(f"\nProcessing combination of {combo_size} pair(s):")
            print(f"  Pairs: {' + '.join(pair_names)}")
            print(f"  Unique models: {individual_models_name}")
            print(f"  Total unique models: {len(unique_models)}")
            
            # Skip if only one unique model (can't ensemble with itself)
            if len(unique_models) < 2:
                print("  Skipping: Need at least 2 unique models for ensemble")
                continue
            
            # Train and evaluate the ensemble
            results = prediction_level_ensemble(
                unique_models,
                df_train, 
                df_val, 
                args.log_dir,
                args.ensemble_method,
                model_combination_name
            )
            
            if results:
                # Add additional info about the combination
                results['combo_size'] = combo_size
                results['n_unique_models'] = len(unique_models)
                results['pairs_used'] = ' + '.join(pair_names)
                all_results.append(results)
                print(f"Combination completed successfully with PRAUC: {results['prauc']:.4f}")
                
                # Save/update the CSV file after each successful combination
                summary_df = pd.DataFrame(all_results)
                summary_df.to_csv(summary_path, index=False)
                print(f"Updated summary saved to {summary_path} ({len(all_results)} combinations)")
                
            else:
                print("Combination failed.")
    
    # Final summary and analysis (keeping the existing end summary)
    if all_results:
        print(f"\nComprehensive summary of all combinations saved to {summary_path}")
        print(f"Final summary contains {len(all_results)} combinations with all metrics")
        
        # Print best combinations for each size
        summary_df = pd.DataFrame(all_results)
        print(f"\n=== SUMMARY OF BEST COMBINATIONS ===")
        for size in sorted(summary_df['combo_size'].unique()):
            size_df = summary_df[summary_df['combo_size'] == size]
            best_prauc = size_df.loc[size_df['prauc'].idxmax()]
            best_clusters_50 = size_df.loc[size_df['clusters_50'].idxmax()]
            
            print(f"\nCombination size {size} ({len(size_df)} combinations evaluated):")
            print(f"  Best PRAUC: {best_prauc['prauc']:.4f} - {best_prauc['pairs_used']}")
            print(f"  Best Clusters@50: {best_clusters_50['clusters_50']} - {best_clusters_50['pairs_used']}")
        
        # Overall best
        overall_best_prauc = summary_df.loc[summary_df['prauc'].idxmax()]
        overall_best_clusters = summary_df.loc[summary_df['clusters_50'].idxmax()]
        
        print(f"\n=== OVERALL BEST COMBINATIONS ===")
        print(f"Best PRAUC: {overall_best_prauc['prauc']:.4f}")
        print(f"  Pairs: {overall_best_prauc['pairs_used']}")
        print(f"  Size: {overall_best_prauc['combo_size']} pairs, {overall_best_prauc['n_unique_models']} unique models")
        
        print(f"\nBest Clusters@50: {overall_best_clusters['clusters_50']}")
        print(f"  Pairs: {overall_best_clusters['pairs_used']}")
        print(f"  Size: {overall_best_clusters['combo_size']} pairs, {overall_best_clusters['n_unique_models']} unique models")
    
    print(f"\nEvaluation of all combinations completed. Total combinations evaluated: {len(all_results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train prediction-level ensemble models using predefined low-correlation pairs"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory to save logs and model checkpoints",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=4,
        help="Maximum number of pairs to combine (default: 4)",
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