from helper import Dataset, ProcessData
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import datetime


def load_pretrained_model(model_type, fingerprint_combo):
    """Load a pretrained model from the specified path"""
    model_path = f"../../runs/DREAM/step2/{model_type}/{fingerprint_combo}/best_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        automl = pickle.load(f)
    
    return automl


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


def generate_test_predictions_step2():
    """Generate predictions for step 2 test set using stacking_logistic ensemble"""
    
    # Load datasets
    print("Loading datasets...")
    df_train = Dataset("../../datasets/DREAM/Train_Dataset_DREAM.parquet").get_dataframe()
    df_test = Dataset("../../datasets/DREAM/Step2_TestData_Target2035.parquet").get_dataframe()
    
    print(f"Training set shape: {df_train.shape}")
    print(f"Test set shape: {df_test.shape}")
    print(f"Test set columns: {list(df_test.columns)}")
    
    # Define the four specific models with their fingerprint combinations
    # Based on model_ensemble.py
    model_specs = [
        ("histgb", "RDK_AVALON_ATOMPAIR"),
        ("lgbm", "MACCS_RDK_AVALON"), 
        ("xgb_limitdepth", "MACCS_ATOMPAIR"),
        ("xgboost", "MACCS_RDK_AVALON_ATOMPAIR")
    ]
    
    print(f"Using stacking_logistic ensemble with {len(model_specs)} models:")
    for model_type, fingerprint_combo in model_specs:
        print(f"  - {model_type} with {fingerprint_combo}")
    
    # Load models and get training predictions for meta-learner
    print("\nGenerating training predictions for meta-learner...")
    model_predictions_train = []
    model_predictions_test = []
    model_info = []
    
    for model_type, fingerprint_combo in model_specs:
        try:
            # Load the model
            model = load_pretrained_model(model_type, fingerprint_combo)
            
            # Parse fingerprint combination to get individual fingerprint types
            fps_types = fingerprint_combo.split("_")
            
            # Prepare training data using concatenated fingerprints
            train_X = concatenate_fingerprints(df_train, fps_types)
            train_predictions = model.predict_proba(train_X)[:, 1]
            model_predictions_train.append(train_predictions)
            
            # Prepare test data using concatenated fingerprints
            test_X = concatenate_fingerprints(df_test, fps_types)
            test_predictions = model.predict_proba(test_X)[:, 1]
            model_predictions_test.append(test_predictions)
            
            model_info.append((model_type, fingerprint_combo))
            print(f"Successfully got predictions from {model_type} with {fingerprint_combo}")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Error processing model {model_type} with {fingerprint_combo}: {e}")
            return None
    
    if len(model_predictions_train) != len(model_specs):
        print("Error: Could not load all required models")
        return None
    
    print(f"Successfully obtained predictions from all {len(model_predictions_train)} models")
    
    # Convert to numpy arrays for stacking
    predictions_array_train = np.array(model_predictions_train)  # Shape: (n_models, n_train_samples)
    predictions_array_test = np.array(model_predictions_test)    # Shape: (n_models, n_test_samples)
    
    # Get training labels
    train_y = df_train["LABEL"].values
    
    print("Training meta-learner with stacking_logistic...")
    # Train meta-learner (Logistic Regression) on training predictions
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    meta_learner.fit(predictions_array_train.T, train_y)  # Transpose to get (n_samples, n_models)
    
    # Get ensemble predictions on test set
    print("Generating ensemble predictions on test set...")
    ensemble_scores = meta_learner.predict_proba(predictions_array_test.T)[:, 1]
    
    print(f"Ensemble predictions shape: {ensemble_scores.shape}")
    print(f"Score range: [{ensemble_scores.min():.6f}, {ensemble_scores.max():.6f}]")
    
    # Create sorted indices based on ensemble scores (descending order)
    sorted_indices = ensemble_scores.argsort()[::-1]
    
    # Create result dataframe with required columns for step 2
    result_df = pd.DataFrame({
        'RandomID': df_test['RandomID'].values,  # Add RandomID
        'SMILES': df_test['SMILES'].values,
        'Score': ensemble_scores
    })
    
    # Add ranking information
    result_df['Rank'] = 0
    result_df.loc[sorted_indices, 'Rank'] = range(1, len(sorted_indices) + 1)
    
    # Create selection columns for top 50, 200, 500, and 5000
    result_df['Sel_50'] = 0
    result_df['Sel_200'] = 0
    result_df['Sel_500'] = 0
    result_df['Sel_5000'] = 0
    
    # Mark top compounds
    top_50_indices = sorted_indices[:50]
    top_200_indices = sorted_indices[:200]
    top_500_indices = sorted_indices[:500]
    top_5000_indices = sorted_indices[:5000]
    
    result_df.loc[top_50_indices, 'Sel_50'] = 1
    result_df.loc[top_200_indices, 'Sel_200'] = 1
    result_df.loc[top_500_indices, 'Sel_500'] = 1
    result_df.loc[top_5000_indices, 'Sel_5000'] = 1
    
    # Sort by score (descending) for final output
    result_df = result_df.sort_values('Score', ascending=False).reset_index(drop=True)
    
    # Select required columns in the specified order for step 2 (including RandomID and Sel_5000)
    final_result = result_df[['RandomID', 'SMILES', 'Sel_50', 'Sel_200', 'Sel_500', 'Sel_5000', 'Score']].copy()
    
    # Ensure the output directory exists
    output_dir = "../../runs/DREAM/step2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to the specified DREAM challenge step 2 submission path
    output_path = "../../runs/DREAM/step2/TeamKoziarskiLab_Step2.csv"
    final_result.to_csv(output_path, index=False)
    
    print(f"\nDREAM Challenge Step2 results saved to: {output_path}")
    print(f"Total compounds: {len(final_result)}")
    print(f"Top 50 selected: {final_result['Sel_50'].sum()}")
    print(f"Top 200 selected: {final_result['Sel_200'].sum()}")
    print(f"Top 500 selected: {final_result['Sel_500'].sum()}")
    print(f"Top 5000 selected: {final_result['Sel_5000'].sum()}")
    
    # Print some statistics
    print(f"\nTop 10 compounds:")
    print(final_result.head(10)[['RandomID', 'SMILES', 'Score', 'Sel_50', 'Sel_200', 'Sel_500', 'Sel_5000']].to_string(index=False))
    
    # Save individual model predictions for analysis (optional, for debugging)
    individual_predictions_df = pd.DataFrame({
        'RandomID': df_test['RandomID'].values,  # Add RandomID here too
        'SMILES': df_test['SMILES'].values,
        'EnsembleScore': ensemble_scores,
    })
    
    # Add individual model predictions as columns
    for i, (model_type, fingerprint_combo) in enumerate(model_info):
        individual_predictions_df[f"{model_type}_{fingerprint_combo}"] = predictions_array_test[i]
    
    individual_path = "../../runs/DREAM/step2/test_predictions_individual_models_step2.csv"
    individual_predictions_df.to_csv(individual_path, index=False)
    print(f"Individual model predictions saved to: {individual_path}")
    
    # Save meta-learner info
    meta_learner_info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ensemble_method": "stacking_logistic",
        "models_used": [f"{model_type}_{fingerprint_combo}" for model_type, fingerprint_combo in model_info],
        "meta_learner_type": "LogisticRegression",
        "meta_learner_params": meta_learner.get_params(),
        "n_models": len(model_info),
        "test_compounds": len(final_result)
    }
    
    meta_info_path = "../../runs/DREAM/step2/meta_learner_info.pkl"
    with open(meta_info_path, "wb") as f:
        pickle.dump(meta_learner_info, f, pickle.HIGHEST_PROTOCOL)
    print(f"Meta-learner info saved to: {meta_info_path}")
    
    return final_result


if __name__ == "__main__":
    print("=== DREAM Challenge Step2 - Generating Test Set Predictions ===")
    print("Team: KoziarskiLab")
    print("Best ensemble method: stacking_logistic")
    print("Models used:")
    print("  1. histgb with RDK_AVALON_ATOMPAIR")
    print("  2. lgbm with MACCS_RDK_AVALON")
    print("  3. xgb_limitdepth with MACCS_ATOMPAIR")
    print("  4. xgboost with MACCS_RDK_AVALON_ATOMPAIR")
    print("Meta-learner: Logistic Regression")
    print("=" * 80)
    
    results = generate_test_predictions_step2()
    
    if results is not None:
        print("\n" + "=" * 80)
        print("SUCCESS: DREAM Challenge Step2 predictions generated successfully!")
        print("Submission file created: ../../runs/DREAM/step2/TeamKoziarskiLab_Step2.csv")
        print("File format: RandomID, SMILES, Sel_50, Sel_200, Sel_500, Sel_5000, Score")
        print("Ready for DREAM Challenge Step 2 submission!")
    else:
        print("\nFAILED: Could not generate test predictions for Step 2.")