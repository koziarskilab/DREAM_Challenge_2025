import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
from tqdm import tqdm
import glob

def calculate_prediction_correlation(fps_types, model_types, output_dir):
    """
    Calculate and visualize the correlation between predictions from models trained with
    different fingerprint types.
    """
    # Hardcoded base directory
    base_dir = "../runs/DREAM/DREAM_BASELINE_SEL"
    
    print("Loading prediction files...")
    print(f"Using base directory: {base_dir}")
    
    # Create a directory for the output files
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store predictions for each fingerprint and model type
    predictions_by_fingerprint = {fps: {} for fps in fps_types}
    all_smiles = None
    
    # Load prediction files
    for fps_type in fps_types:
        for model_type in model_types:
            prediction_file = os.path.join(base_dir, f"{fps_type}_{model_type}", "val_predictions.csv")
            
            if os.path.exists(prediction_file):
                print(f"Loading {prediction_file}")
                df = pd.read_csv(prediction_file)
                
                # Store predictions
                if 'PredictedScore' in df.columns and 'SMILES' in df.columns:
                    # Store SMILES and prediction
                    predictions_df = df[['SMILES', 'PredictedScore']].copy()
                    predictions_by_fingerprint[fps_type][model_type] = predictions_df
                    
                    # Track all SMILES we've seen
                    if all_smiles is None:
                        all_smiles = set(df['SMILES'])
                    else:
                        all_smiles = all_smiles.intersection(set(df['SMILES']))
                else:
                    print(f"Warning: Required columns not found in {prediction_file}")
            else:
                print(f"Warning: File not found: {prediction_file}")
    
    if not all_smiles:
        print("Error: No valid prediction files found")
        return
    
    print(f"Found {len(all_smiles)} common molecules across all prediction files")
    
    # Create a merged DataFrame with predictions from all models
    merged_predictions = []
    
    # Start with SMILES column
    for smiles in tqdm(all_smiles, desc="Merging predictions"):
        row = {'SMILES': smiles}
        
        # Add predictions from each fingerprint and model
        for fps_type in fps_types:
            for model_type in model_types:
                if model_type in predictions_by_fingerprint[fps_type]:
                    # Find the prediction for this SMILES
                    pred_df = predictions_by_fingerprint[fps_type][model_type]
                    pred_row = pred_df[pred_df['SMILES'] == smiles]
                    
                    if not pred_row.empty:
                        column_name = f"{fps_type}_{model_type}"
                        row[column_name] = pred_row.iloc[0]['PredictedScore']
        
        merged_predictions.append(row)
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(merged_predictions)
    
    # Save merged predictions
    predictions_path = os.path.join(output_dir, "merged_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved merged predictions to {predictions_path}")
    
    # Get list of prediction columns
    prediction_columns = [col for col in predictions_df.columns if col != 'SMILES']
    
    # Calculate correlation matrix
    print("Calculating prediction correlations...")
    correlation_matrix = predictions_df[prediction_columns].corr()
    
    # Create a grouping by fingerprint type for visualization
    groups = {fps: [col for col in prediction_columns if col.startswith(fps)] 
              for fps in fps_types}
    
    # Save raw correlation data
    correlation_matrix.to_csv(os.path.join(output_dir, "prediction_correlations.csv"))
    
    # Plot correlation heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', 
                xticklabels=correlation_matrix.columns, 
                yticklabels=correlation_matrix.columns,
                vmin=-1, vmax=1)
    plt.title('Correlation Between Predictions from Different Fingerprints & Models')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    corr_path = os.path.join(output_dir, 'prediction_correlations_heatmap.png')
    plt.savefig(corr_path)
    print(f"Saved correlation heatmap to {corr_path}")
    
    # Calculate average correlation between fingerprint types
    fps_correlation = pd.DataFrame(index=fps_types, columns=fps_types)
    
    for fps1 in fps_types:
        for fps2 in fps_types:
            # Get all columns for each fingerprint
            cols1 = groups[fps1]
            cols2 = groups[fps2]
            
            # Extract correlations between these groups
            corrs = []
            for c1 in cols1:
                for c2 in cols2:
                    if c1 != c2:  # Avoid self-correlations
                        corrs.append(correlation_matrix.loc[c1, c2])
            
            # Store average correlation
            fps_correlation.loc[fps1, fps2] = np.mean(corrs) if corrs else np.nan
    
    # Plot fingerprint correlation heatmap
    plt.figure(figsize=(10, 8))
    mask = np.zeros_like(fps_correlation, dtype=bool)
    np.fill_diagonal(mask, True)  # Mask the diagonal for better visualization
    
    sns.heatmap(fps_correlation, annot=True, cmap='coolwarm', 
                mask=mask, vmin=-1, vmax=1)
    plt.title('Average Correlation Between Predictions from Different Fingerprint Types')
    plt.tight_layout()
    fps_corr_path = os.path.join(output_dir, 'fingerprint_prediction_correlation.png')
    plt.savefig(fps_corr_path)
    print(f"Saved fingerprint correlation heatmap to {fps_corr_path}")
    
    # Save fingerprint correlation data
    fps_correlation.to_csv(os.path.join(output_dir, "fingerprint_prediction_correlation.csv"))
    
    # For each model type, calculate correlation between fingerprints
    for model in model_types:
        # Check if this model exists for multiple fingerprints
        model_cols = [col for col in prediction_columns if col.endswith(f"_{model}")]
        if len(model_cols) > 1:
            # Calculate correlation for this model across fingerprints
            model_corr = predictions_df[model_cols].corr()
            
            # Plot model-specific correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(model_corr, annot=True, cmap='viridis', vmin=-1, vmax=1)
            plt.title(f'Correlation Between Fingerprints for {model} Model')
            plt.tight_layout()
            model_corr_path = os.path.join(output_dir, f'{model}_fingerprint_correlation.png')
            plt.savefig(model_corr_path)
            print(f"Saved {model} model correlation heatmap to {model_corr_path}")
            
            # Save model correlation data
            model_corr.to_csv(os.path.join(output_dir, f"{model}_fingerprint_correlation.csv"))
    
    # Create scatterplots for pairs with interesting correlations
    # Find highest and lowest correlation pairs
    pairs = []
    for i, col1 in enumerate(prediction_columns):
        for col2 in prediction_columns[i+1:]:
            corr = correlation_matrix.loc[col1, col2]
            pairs.append((col1, col2, corr))
    
    # Sort by absolute correlation
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Plot the top 5 and bottom 5 correlation pairs
    top_pairs = pairs[:5]
    bottom_pairs = pairs[-5:]
    
    os.makedirs(os.path.join(output_dir, "scatter_plots"), exist_ok=True)
    
    # Plot top correlated pairs
    for col1, col2, corr in top_pairs:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=predictions_df, x=col1, y=col2, alpha=0.6)
        plt.title(f'Correlation: {corr:.3f}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "scatter_plots", f"high_corr_{col1}_vs_{col2}.png"))
        plt.close()
    
    # Plot bottom correlated pairs
    for col1, col2, corr in bottom_pairs:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=predictions_df, x=col1, y=col2, alpha=0.6)
        plt.title(f'Correlation: {corr:.3f}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "scatter_plots", f"low_corr_{col1}_vs_{col2}.png"))
        plt.close()
    
    print("Prediction correlation analysis completed!")
    return correlation_matrix, fps_correlation


def main():
    parser = argparse.ArgumentParser(description="Analyze correlations between predictions from different fingerprints")
    parser.add_argument('--output_dir', type=str, default="./prediction_correlation_analysis",
                      help='Directory to save analysis results')
    parser.add_argument('--fps_types', nargs='+', default=["MACCS", "RDK", "AVALON", "ATOMPAIR"],
                      help='List of fingerprint types to analyze')
    parser.add_argument('--model_types', nargs='+', default=["lgbm", "xgboost", "rf", "extra_tree", "histgb"],
                      help='List of model types to analyze')
    
    args = parser.parse_args()
    
    # Display configurations
    print(f"Fingerprint types: {args.fps_types}")
    print(f"Model types: {args.model_types}")
    
    # Run correlation analysis
    calculate_prediction_correlation(args.fps_types, args.model_types, args.output_dir)


if __name__ == "__main__":
    main()