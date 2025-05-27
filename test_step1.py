from helper import Dataset, ProcessData
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

def load_pretrained_model(fingerprint_type, model_type):
    """Load a pretrained model from the specified path"""
    model_path = f"./runs/DREAM/DREAM_BASELINE_SEL/{fingerprint_type}_{model_type}/best_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        automl = pickle.load(f)
    
    return automl

def generate_test_predictions():
    """Generate predictions for the test set using the best ensemble model"""
    
    # Load test dataset
    print("Loading test dataset...")
    df_test = Dataset("./datasets/DREAM/Step1_TestData_Target2035.parquet").get_dataframe()
    print(f"Test set shape: {df_test.shape}")
    print(f"Test set columns: {list(df_test.columns)}")
    print(f"First few column names: {df_test.columns[:10].tolist()}")
    
    # Define your best ensemble model components
    # New ensemble: 5 pairs = 10 models total
    best_ensemble_specs = [
        ("AVALON", "histgb"),
        ("ATOMPAIR", "xgboost"),
        ("AVALON", "extra_tree"),
        ("ATOMPAIR", "histgb"),
        ("RDK", "catboost"),
        ("ATOMPAIR", "lgbm"),
        ("ATOMPAIR", "catboost"),
        ("RDK", "extra_tree"),
        ("ATOMPAIR", "lgbm"),
        ("RDK", "histgb")
    ]
    
    print(f"Best ensemble consists of {len(best_ensemble_specs)} models:")
    for fps_type, model_type in best_ensemble_specs:
        print(f"  - {fps_type}_{model_type}")
    
    # Load models and get predictions with progress bar
    model_predictions = []
    model_info = []
    
    print("\nLoading models and generating predictions...")
    for fps_type, model_type in tqdm(best_ensemble_specs, desc="Processing models"):
        try:
            # Load the model
            model = load_pretrained_model(fps_type, model_type)
            
            # Prepare test data using the model's specific fingerprint type
            test_X = ProcessData(df_test, fps_type).get_data()
            test_predictions = model.predict_proba(test_X)[:, 1]
            model_predictions.append(test_predictions)
            
            model_info.append((fps_type, model_type))
            print(f"Successfully got predictions from {fps_type}_{model_type}")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Error processing model {fps_type}_{model_type}: {e}")
            return None
    
    if len(model_predictions) != len(best_ensemble_specs):
        print("Error: Could not load all required models")
        return None
    
    print(f"Successfully obtained predictions from all {len(model_predictions)} models")
    
    # Convert to numpy array for ensemble
    predictions_array = np.array(model_predictions)  # Shape: (n_models, n_samples)
    
    # Apply median_vote ensemble method
    print("Applying median_vote ensemble method...")
    ensemble_scores = np.median(predictions_array, axis=0)
    
    print(f"Ensemble predictions shape: {ensemble_scores.shape}")
    print(f"Score range: [{ensemble_scores.min():.6f}, {ensemble_scores.max():.6f}]")
    
    # Create sorted indices based on ensemble scores (descending order)
    sorted_indices = ensemble_scores.argsort()[::-1]
    
    # Create result dataframe
    result_df = pd.DataFrame({
        'RandomID': df_test['RandomID'].values,
        'Score': ensemble_scores
    })
    
    # Add ranking information
    result_df['Rank'] = 0
    result_df.loc[sorted_indices, 'Rank'] = range(1, len(sorted_indices) + 1)
    
    # Create Sel_200 and Sel_500 columns
    result_df['Sel_200'] = 0
    result_df['Sel_500'] = 0
    
    # Mark top 200 and top 500
    top_200_indices = sorted_indices[:200]
    top_500_indices = sorted_indices[:500]
    
    result_df.loc[top_200_indices, 'Sel_200'] = 1
    result_df.loc[top_500_indices, 'Sel_500'] = 1
    
    # Sort by score (descending) for final output
    result_df = result_df.sort_values('Score', ascending=False).reset_index(drop=True)
    
    # Select only required columns in the specified order
    final_result = result_df[['RandomID', 'Sel_200', 'Sel_500', 'Score']].copy()
    
    # Ensure the output directory exists
    output_dir = "./runs/DREAM"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to the specified DREAM challenge submission path
    output_path = "./runs/DREAM/TeamKoziarskiLab.csv"
    final_result.to_csv(output_path, index=False)
    
    print(f"\nDREAM Challenge Step1 results saved to: {output_path}")
    print(f"Total compounds: {len(final_result)}")
    print(f"Top 200 selected: {final_result['Sel_200'].sum()}")
    print(f"Top 500 selected: {final_result['Sel_500'].sum()}")
    
    # Print some statistics
    print(f"\nTop 10 compounds:")
    print(final_result.head(10)[['RandomID', 'Score', 'Sel_200', 'Sel_500']].to_string(index=False))
    
    # Save individual model predictions for analysis (optional, for debugging)
    individual_predictions_df = pd.DataFrame({
        'RandomID': df_test['RandomID'].values,
        'EnsembleScore': ensemble_scores,
    })
    
    # Add individual model predictions as columns
    for i, (fps_type, model_type) in enumerate(model_info):
        individual_predictions_df[f"{fps_type}_{model_type}"] = predictions_array[i]
    
    individual_path = "./runs/DREAM/test_predictions_individual_models.csv"
    individual_predictions_df.to_csv(individual_path, index=False)
    print(f"Individual model predictions saved to: {individual_path}")
    
    return final_result

if __name__ == "__main__":
    print("=== DREAM Challenge Step1 - Generating Test Set Predictions ===")
    print("Team: KoziarskiLab")
    print("Best ensemble: (AVALON_histgb + ATOMPAIR_xgboost) + (AVALON_extra_tree + ATOMPAIR_histgb) + (RDK_catboost + ATOMPAIR_lgbm) + (ATOMPAIR_catboost + RDK_extra_tree) + (ATOMPAIR_lgbm + RDK_histgb)")
    print("Ensemble method: median_vote")
    print("=" * 80)
    
    results = generate_test_predictions()
    
    if results is not None:
        print("\n" + "=" * 80)
        print("SUCCESS: DREAM Challenge Step1 predictions generated successfully!")
        print("Submission file created: ./runs/DREAM/TeamKoziarskiLab.csv")
        print("File format: RandomID, Sel_200, Sel_500, Score")
        print("Ready for DREAM Challenge submission!")
    else:
        print("\nFAILED: Could not generate test predictions.")