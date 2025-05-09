from helper import Dataset, ProcessData
import argparse
import os
import pandas as pd
import numpy as np
import pickle
import joblib
import re

def extract_model_info(model_path):
    """Extract fingerprint type and model type from the model path."""
    # Extract the directory containing the model file
    model_dir = os.path.dirname(model_path)
    
    # Get the last directory component which should be like "MACCS_lgbm"
    config_dir = os.path.basename(model_dir)
    
    # Split by underscore to get fingerprint type and model type
    parts = config_dir.split('_')
    if len(parts) >= 2:
        fps_type = parts[0]
        model_type = parts[1]
        
        # Validate fingerprint type
        if fps_type not in ["MACCS", "RDK", "AVALON", "ATOMPAIR"]:
            raise ValueError(f"Extracted fingerprint type '{fps_type}' is not supported. Must be one of: 'MACCS', 'RDK', 'AVALON', 'ATOMPAIR'")
        
        return fps_type, model_type
    else:
        raise ValueError(f"Could not extract fingerprint and model types from path: {model_path}")

def main(args):
    # Fixed test path
    test_path = "./datasets/DEARM/Step1_TestData_Target2035.parquet"
    
    # Extract fingerprint type from model path
    fps_type, model_type = extract_model_info(args.model_path)
    print(f"Extracted fingerprint type: {fps_type}, model type: {model_type}")
    
    # Set output directory to the model directory
    output_dir = os.path.dirname(args.model_path)
    
    # Load the test dataset
    print(f"Loading test dataset from {test_path}")
    df_test = pd.read_parquet(test_path)
    
    # Load the trained model
    print(f"Loading model from {args.model_path}")
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Process the test data with the extracted fingerprint type
    print(f"Processing test data with fingerprint type: {fps_type}")
    TestData = ProcessData(df_test, fps_type).get_data()
    
    # Generate predictions
    print("Generating predictions...")
    probabilities = model.predict_proba(TestData)[:, 1]
    
    # Create results dataframe
    print("Creating results dataframe...")
    df_results = pd.DataFrame({
        "COMPOUND_ID": df_test["RandomID"],
        "PredictedScore": probabilities
    })
    
    # Save results
    output_path = os.path.join(output_dir, f"test_predictions.csv")
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Print summary statistics
    print("\nPrediction Summary:")
    print(f"Total predictions: {len(df_results)}")
    print(f"Average score: {np.mean(probabilities):.4f}")
    print(f"Median score: {np.median(probabilities):.4f}")
    print(f"Min score: {np.min(probabilities):.4f}")
    print(f"Max score: {np.max(probabilities):.4f}")
    
    # Return path for potential follow-up processing
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model file (either .pkl or joblib format)")
    
    args = parser.parse_args()
    
    main(args)