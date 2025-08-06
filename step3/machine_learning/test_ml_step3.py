from helper import Dataset, ProcessData
import os
import numpy as np
import pandas as pd
import pickle
import datetime


def load_pretrained_model(model_type, fingerprint_combo):
    """Load a pretrained model from the specified path"""
    model_path = f"./runs/DREAM/step2/{model_type}/{fingerprint_combo}/best_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        automl = pickle.load(f)
    
    return automl


def concatenate_fingerprints(df, fps_types):
    """
    Concatenate multiple fingerprint types for a given dataframe.
    Memory-optimized version with chunking and dtype optimization.
    """
    concatenated_data = []
    
    for fps_type in fps_types:
        data = ProcessData(df, fps_type).get_data()
        # Convert to float32 to save memory (instead of float64)
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        concatenated_data.append(data)
    
    # Concatenate and immediately free intermediate arrays
    result = np.concatenate(concatenated_data, axis=1)
    
    # Clean up intermediate arrays
    for data in concatenated_data:
        del data
    del concatenated_data
    
    return result


def majority_vote_ensemble(predictions_array):
    """
    Ensemble predictions using majority voting
    
    Args:
        predictions_array: Array of shape (n_models, n_samples) with model predictions
    
    Returns:
        ensemble_probs: Array of ensemble scores between 0 and 1
    """
    # Convert probabilities to hard predictions and take majority vote
    hard_predictions = (predictions_array > 0.5).astype(int)
    ensemble_probs = np.mean(hard_predictions, axis=0)
    
    return ensemble_probs


def max_vote_ensemble(predictions_array):
    """
    Ensemble predictions using max voting
    
    Args:
        predictions_array: Array of shape (n_models, n_samples) with model predictions
    
    Returns:
        ensemble_probs: Array of ensemble scores between 0 and 1
    """
    # Take maximum probability across all models
    ensemble_probs = np.max(predictions_array, axis=0)
    
    return ensemble_probs


def process_models_in_chunks(df_test, model_specs, chunk_size=10000):
    """
    Process test predictions in chunks to reduce memory usage
    """
    n_samples = len(df_test)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    print(f"Processing {n_samples} samples in {n_chunks} chunks of size {chunk_size}")
    
    # Initialize result arrays
    all_predictions = np.zeros((len(model_specs), n_samples), dtype=np.float32)
    model_info = []
    
    # Load all models first (they're relatively small)
    print("Loading all models...")
    loaded_models = []
    for model_type, fingerprint_combo in model_specs:
        try:
            model = load_pretrained_model(model_type, fingerprint_combo)
            fps_types = fingerprint_combo.split("_")
            loaded_models.append((model, fps_types, model_type, fingerprint_combo))
            model_info.append((model_type, fingerprint_combo))
            print(f"Loaded {model_type} with {fingerprint_combo}")
        except Exception as e:
            print(f"Error loading model {model_type} with {fingerprint_combo}: {e}")
            return None, None
    
    # Process data in chunks
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_samples)
        
        print(f"Processing chunk {chunk_idx + 1}/{n_chunks} (samples {start_idx}:{end_idx})")
        
        # Get chunk of data
        df_chunk = df_test.iloc[start_idx:end_idx].copy()
        
        # Process each model for this chunk
        for model_idx, (model, fps_types, model_type, fingerprint_combo) in enumerate(loaded_models):
            try:
                # Prepare chunk data
                chunk_X = concatenate_fingerprints(df_chunk, fps_types)
                
                # Get predictions for this chunk
                chunk_predictions = model.predict_proba(chunk_X)[:, 1].astype(np.float32)
                
                # Store in result array
                all_predictions[model_idx, start_idx:end_idx] = chunk_predictions
                
                # Clean up chunk data
                del chunk_X, chunk_predictions
                
            except Exception as e:
                print(f"Error processing chunk {chunk_idx} for model {model_type}: {e}")
                return None, None
        
        # Clean up chunk dataframe
        del df_chunk
        
        # Force garbage collection
        import gc
        gc.collect()
    
    return all_predictions, model_info


def generate_test_predictions_step3():
    """Generate predictions for step 3 test set using max voting ensemble"""
    
    # Load datasets with memory optimization
    print("Loading datasets...")
    try:
        # Try to load with specific dtypes to save memory
        dtype_dict = {
            'Catalog_ID': 'category',  # Use category for string IDs
            'SMILES': 'string',  # Use string dtype for SMILES
            'Molecular_Weight': 'float32',
            'aLogP': 'float32'
        }
        df_test = pd.read_csv("./datasets/DREAM/Test_Step3_Dataset_DREAM_w_FP.csv", dtype=dtype_dict)
    except:
        # Fallback to default loading
        df_test = Dataset("./datasets/DREAM/Test_Step3_Dataset_DREAM_w_FP.csv").get_dataframe()
    
    print(f"Test set shape: {df_test.shape}")
    print(f"Memory usage: {df_test.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Define the four specific models with their fingerprint combinations
    model_specs = [
        ("histgb", "RDK_AVALON_ATOMPAIR"),
        ("lgbm", "MACCS_RDK_AVALON"), 
        ("xgb_limitdepth", "MACCS_ATOMPAIR"),
        ("xgboost", "MACCS_RDK_AVALON_ATOMPAIR")
    ]
    
    print(f"Using max_vote ensemble with {len(model_specs)} models:")
    for model_type, fingerprint_combo in model_specs:
        print(f"  - {model_type} with {fingerprint_combo}")
    
    # Process models in chunks to reduce memory usage
    print("\nGenerating test predictions with chunked processing...")
    
    # Determine chunk size based on dataset size
    n_samples = len(df_test)
    if n_samples > 100000:
        chunk_size = 5000
    elif n_samples > 50000:
        chunk_size = 10000
    else:
        chunk_size = 20000
    
    predictions_array_test, model_info = process_models_in_chunks(df_test, model_specs, chunk_size)
    
    if predictions_array_test is None:
        print("Error: Could not generate predictions")
        return None
    
    print(f"Successfully obtained predictions from all {len(model_info)} models")
    print(f"Predictions array shape: {predictions_array_test.shape}")
    print(f"Predictions memory usage: {predictions_array_test.nbytes / 1024**2:.2f} MB")
    
    print("Generating ensemble predictions using max voting...")
    # Get ensemble predictions using max voting
    ensemble_scores = max_vote_ensemble(predictions_array_test)
    
    print(f"Ensemble predictions shape: {ensemble_scores.shape}")
    print(f"Score range: [{ensemble_scores.min():.6f}, {ensemble_scores.max():.6f}]")
    
    # Create sorted indices based on ensemble scores (descending order)
    sorted_indices = ensemble_scores.argsort()[::-1]
    
    # Only keep top 5000 compounds
    top_5000_indices = sorted_indices[:5000]
    
    # Create result dataframe with only top 5000 compounds
    result_df = pd.DataFrame({
        'Catalog_ID': df_test['Catalog_ID'].iloc[top_5000_indices].values,
        'SMILES': df_test['SMILES'].iloc[top_5000_indices].values,
        'Score': ensemble_scores[top_5000_indices].astype(np.float32),
        'Molecular_Weight': df_test['Molecular_Weight'].iloc[top_5000_indices].values,
        'aLogP': df_test['aLogP'].iloc[top_5000_indices].values
    })
    
    # Add ranking information (1 to 5000)
    result_df['Rank'] = range(1, 5001)
    
    # Select required columns in the specified order for step 3
    final_result = result_df[['Catalog_ID', 'SMILES', 'Molecular_Weight', 'aLogP', 'Score']].copy()
    
    # Save individual model predictions for analysis (only for top 5000) BEFORE deleting predictions_array_test
    individual_predictions_df = pd.DataFrame({
        'Catalog_ID': df_test['Catalog_ID'].iloc[top_5000_indices].values,
        'SMILES': df_test['SMILES'].iloc[top_5000_indices].values,
        'EnsembleScore': ensemble_scores[top_5000_indices].astype(np.float32),
    })
    
    # Add individual model predictions as columns (only for top 5000)
    for i, (model_type, fingerprint_combo) in enumerate(model_info):
        individual_predictions_df[f"{model_type}_{fingerprint_combo}"] = predictions_array_test[i, top_5000_indices]
    
    # Clean up large intermediate variables AFTER using them
    del result_df, predictions_array_test
    import gc
    gc.collect()
    
    # Ensure the output directory exists
    output_dir = "./runs/DREAM/step3"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to the specified DREAM challenge step 3 submission path
    output_path = "./runs/DREAM/step3/TeamKoziarskiLab_Step3.csv"
    final_result.to_csv(output_path, index=False)
    
    print(f"\nDREAM Challenge Step3 results saved to: {output_path}")
    print(f"Total compounds (top 5000): {len(final_result)}")
    
    # Print some statistics
    print(f"\nTop 10 compounds:")
    print(final_result.head(10).to_string(index=False))
    
    # Save individual model predictions for analysis (only for top 5000)
    individual_path = "./runs/DREAM/step3/test_predictions_individual_models_step3.csv"
    individual_predictions_df.to_csv(individual_path, index=False)
    print(f"Individual model predictions (top 5000) saved to: {individual_path}")
    
    # Save ensemble info
    ensemble_info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ensemble_method": "max_vote",
        "models_used": [f"{model_type}_{fingerprint_combo}" for model_type, fingerprint_combo in model_info],
        "n_models": len(model_info),
        "test_compounds": len(final_result),
        "compounds_saved": "top_5000_only"
    }
    
    meta_info_path = "./runs/DREAM/step3/ensemble_info.pkl"
    with open(meta_info_path, "wb") as f:
        pickle.dump(ensemble_info, f, pickle.HIGHEST_PROTOCOL)
    print(f"Ensemble info saved to: {meta_info_path}")
    
    return final_result


if __name__ == "__main__":
    print("=== DREAM Challenge Step3 - Generating Test Set Predictions ===")
    print("Team: KoziarskiLab")
    print("Best ensemble method: max_vote")
    print("Models used:")
    print("  1. histgb with RDK_AVALON_ATOMPAIR")
    print("  2. lgbm with MACCS_RDK_AVALON")
    print("  3. xgb_limitdepth with MACCS_ATOMPAIR")
    print("  4. xgboost with MACCS_RDK_AVALON_ATOMPAIR")
    print("Ensemble method: Max Voting")
    print("Output: Top 5000 compounds only")
    print("=" * 80)
    
    results = generate_test_predictions_step3()
    
    if results is not None:
        print("\n" + "=" * 80)
        print("SUCCESS: DREAM Challenge Step3 predictions generated successfully!")
        print("Submission file created: ./runs/DREAM/step3/TeamKoziarskiLab_Step3.csv")
        print("File format: Catalog_ID, SMILES, Molecular_Weight, aLogP, Score")
        print("Contains only top 5000 compounds ranked by max voting ensemble")
        print("Ready for DREAM Challenge Step 3 submission!")
    else:
        print("\nFAILED: Could not generate test predictions for Step 3.")