import pandas as pd
import yaml
import os
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import copy
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

def load_template_yaml(template_path):
    """Load the template YAML file"""
    with open(template_path, 'r') as f:
        return yaml.safe_load(f)

def create_molecule_yaml(template_config, smiles, molecule_id, template_yaml_name, current_datetime, template_path):
    """Create a YAML config file for a specific molecule"""
    # Read the original template as text to preserve formatting
    with open(template_path, 'r') as f:
        template_text = f.read()
    
    # Find the original SMILES in the template
    original_smiles = template_config['sequences'][1]['ligand']['smiles']
    
    # Replace the SMILES while preserving formatting
    modified_text = template_text.replace(original_smiles, smiles)
    
    # Create config directory in the same structure as output
    config_dir = Path(f'../../runs/DREAM/{template_yaml_name}_{current_datetime}/{molecule_id}')
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Write the YAML file
    yaml_path = config_dir / f"{molecule_id}.yaml"
    with open(yaml_path, 'w') as f:
        f.write(modified_text)
    
    return yaml_path

def extract_metrics(molecule_id, template_yaml_name, current_datetime):
    """Extract prediction metrics from JSON files"""
    base_path = Path(f'../../runs/DREAM/{template_yaml_name}_{current_datetime}/{molecule_id}/boltz_results_{molecule_id}/predictions/{molecule_id}')
    
    metrics = {}
    
    # Extract confidence metrics
    confidence_file = base_path / f'confidence_{molecule_id}_model_0.json'
    if confidence_file.exists():
        with open(confidence_file, 'r') as f:
            confidence_data = json.load(f)
            metrics.update({
                'confidence_score': confidence_data.get('confidence_score'),
                'ptm': confidence_data.get('ptm'),
                'iptm': confidence_data.get('iptm'),
                'ligand_iptm': confidence_data.get('ligand_iptm'),
                'protein_iptm': confidence_data.get('protein_iptm'),
                'complex_plddt': confidence_data.get('complex_plddt'),
                'complex_iplddt': confidence_data.get('complex_iplddt'),
                'complex_pde': confidence_data.get('complex_pde'),
                'complex_ipde': confidence_data.get('complex_ipde')
            })
    
    # Extract affinity metrics
    affinity_file = base_path / f'affinity_{molecule_id}.json'
    if affinity_file.exists():
        with open(affinity_file, 'r') as f:
            affinity_data = json.load(f)
            metrics.update({
                'affinity_pred_value': affinity_data.get('affinity_pred_value'),
                'affinity_probability_binary': affinity_data.get('affinity_probability_binary')
            })
    
    return metrics

def run_boltz_prediction(yaml_path, molecule_id, template_yaml_name, current_datetime):
    """Run boltz prediction for a single molecule"""
    out_dir = f"../../runs/DREAM/{template_yaml_name}_{current_datetime}/{molecule_id}"
    
    cmd = [
        'boltz', 'predict', str(yaml_path),
        '--use_msa_server', '--use_potentials',
        '--out_dir', out_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"

def update_results_csv(result_row, csv_path):
    """Update the results CSV file with a new row"""
    if csv_path.exists():
        # Read existing data and append
        existing_df = pd.read_csv(csv_path)
        updated_df = pd.concat([existing_df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        # Create new DataFrame
        updated_df = pd.DataFrame([result_row])
    
    # Save to CSV
    updated_df.to_csv(csv_path, index=False)

def calculate_correlations(df, label_col='LABEL'):
    """Calculate correlations between LABEL and prediction metrics"""
    # Check if label column exists
    if label_col not in df.columns:
        print(f"Warning: '{label_col}' column not found in data. Skipping correlation calculation.")
        return {}
    
    metric_cols = ['confidence_score', 'ptm', 'iptm', 'ligand_iptm', 'protein_iptm', 
                   'complex_plddt', 'complex_iplddt', 'complex_pde', 'complex_ipde', 
                   'affinity_pred_value', 'affinity_probability_binary']
    
    correlations = {}
    
    for metric in metric_cols:
        if metric in df.columns and not df[metric].isna().all():
            # Remove rows with NaN values for this metric
            valid_data = df[[label_col, metric]].dropna()
            
            if len(valid_data) > 1:
                try:
                    # Calculate Pearson correlation
                    pearson_r, pearson_p = pearsonr(valid_data[label_col], valid_data[metric])
                    # Calculate Spearman correlation
                    spearman_r, spearman_p = spearmanr(valid_data[label_col], valid_data[metric])
                    
                    correlations[metric] = {
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'n_samples': len(valid_data)
                    }
                except:
                    correlations[metric] = {
                        'pearson_r': np.nan,
                        'pearson_p': np.nan,
                        'spearman_r': np.nan,
                        'spearman_p': np.nan,
                        'n_samples': len(valid_data)
                    }
    
    return correlations

def save_cif_file(molecule_id, template_yaml_name, current_datetime):
    """Save the CIF file to visualization directory"""
    # Source CIF file path
    source_cif = Path(f'../../runs/DREAM/{template_yaml_name}_{current_datetime}/{molecule_id}/boltz_results_{molecule_id}/predictions/{molecule_id}/{molecule_id}_model_0.cif')
    
    # Create visualization directory
    viz_dir = Path(f'../../runs/DREAM/{template_yaml_name}_{current_datetime}/visualization')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Destination CIF file path
    dest_cif = viz_dir / f'{molecule_id}.cif'
    
    # Copy the file if it exists
    if source_cif.exists():
        shutil.copy2(source_cif, dest_cif)
        return True, str(dest_cif)
    else:
        return False, f"CIF file not found: {source_cif}"

def main():
    parser = argparse.ArgumentParser(description='Run Boltz predictions in batch')
    parser.add_argument('--csv_file', default='datasets/boltz/WDR91/known_active_molecules_Boltz-2.csv',
                       help='Path to CSV file with molecules')
    parser.add_argument('--template_yaml', default='config/wdr91_8hsj_pocket_template.yaml',
                       help='Path to template YAML file')
    
    args = parser.parse_args()
    
    # Get template yaml name without extension and current datetime
    template_yaml_name = Path(args.template_yaml).stem
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Load the CSV file
    print(f"Loading molecules from {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    
    # Load the template YAML
    print(f"Loading template from {args.template_yaml}")
    template_config = load_template_yaml(args.template_yaml)
    
    # Create base directories
    Path(f'../../runs/DREAM/{template_yaml_name}_{current_datetime}').mkdir(parents=True, exist_ok=True)
    
    # Set up results CSV path
    results_csv_path = Path(f'../../runs/DREAM/{template_yaml_name}_{current_datetime}/final_results.csv')
    
    # Process each molecule
    successful_predictions = 0
    failed_predictions = []
    
    print(f"Starting batch prediction for {len(df)} molecules...")
    print(f"Output directory: ../../runs/DREAM/{template_yaml_name}_{current_datetime}/")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
        smiles = row['SMILES']
        molecule_id = row['ID']
        label = row.get('LABEL', np.nan)  # Use get() to handle missing LABEL column
        
        try:
            # Create YAML config for this molecule
            yaml_path = create_molecule_yaml(template_config, smiles, molecule_id, template_yaml_name, current_datetime, args.template_yaml)
            tqdm.write(f"Created config: {yaml_path}")
            
            # Run prediction
            tqdm.write(f"Running prediction for {molecule_id}...")
            success, output = run_boltz_prediction(yaml_path, molecule_id, template_yaml_name, current_datetime)
            
            if success:
                successful_predictions += 1
                tqdm.write(f"✓ Successfully predicted {molecule_id}")
                
                # Extract metrics
                tqdm.write(f"Extracting metrics for {molecule_id}...")
                metrics = extract_metrics(molecule_id, template_yaml_name, current_datetime)
                
                # Save CIF file to visualization directory
                cif_saved, cif_path = save_cif_file(molecule_id, template_yaml_name, current_datetime)
                if cif_saved:
                    tqdm.write(f"✓ CIF file saved: {cif_path}")
                else:
                    tqdm.write(f"⚠ CIF file not saved: {cif_path}")
                
                # Combine all data
                result_row = {
                    'SMILES': smiles,
                    'ID': molecule_id,
                    'LABEL': label,
                    **metrics
                }
                
            else:
                failed_predictions.append((molecule_id, output))
                tqdm.write(f"✗ Failed to predict {molecule_id}: {output}")
                
                # Add row with NaN metrics for failed predictions
                result_row = {
                    'SMILES': smiles,
                    'ID': molecule_id,
                    'LABEL': label,
                    'confidence_score': np.nan,
                    'ptm': np.nan,
                    'iptm': np.nan,
                    'ligand_iptm': np.nan,
                    'protein_iptm': np.nan,
                    'complex_plddt': np.nan,
                    'complex_iplddt': np.nan,
                    'complex_pde': np.nan,
                    'complex_ipde': np.nan,
                    'affinity_pred_value': np.nan,
                    'affinity_probability_binary': np.nan
                }
            
            # Update CSV after each prediction
            update_results_csv(result_row, results_csv_path)
            tqdm.write(f"Updated results CSV: {results_csv_path}")
                
        except Exception as e:
            failed_predictions.append((molecule_id, str(e)))
            tqdm.write(f"✗ Error processing {molecule_id}: {str(e)}")
            
            # Add row with NaN metrics for error cases
            result_row = {
                'SMILES': smiles,
                'ID': molecule_id,
                'LABEL': label,
                'confidence_score': np.nan,
                'ptm': np.nan,
                'iptm': np.nan,
                'ligand_iptm': np.nan,
                'protein_iptm': np.nan,
                'complex_plddt': np.nan,
                'complex_iplddt': np.nan,
                'complex_pde': np.nan,
                'complex_ipde': np.nan,
                'affinity_pred_value': np.nan,
                'affinity_probability_binary': np.nan
            }
            
            # Update CSV after each error
            update_results_csv(result_row, results_csv_path)
            tqdm.write(f"Updated results CSV: {results_csv_path}")
    
    # Calculate and save correlations at the end only if LABEL column exists
    final_results_df = pd.read_csv(results_csv_path)
    if 'LABEL' in final_results_df.columns:
        print("\nCalculating correlations...")
        correlations = calculate_correlations(final_results_df)
        if correlations:  # Only save if correlations were calculated
            correlations_df = pd.DataFrame(correlations).T
            correlations_csv_path = f'../../runs/DREAM/{template_yaml_name}_{current_datetime}/correlations.csv'
            correlations_df.to_csv(correlations_csv_path)
            print(f"Correlations saved to: {correlations_csv_path}")
    else:
        print("\nNo LABEL column found - skipping correlation analysis.")
        correlations = {}
    
    # Summary
    print("\n" + "="*50)
    print("BATCH PREDICTION SUMMARY")
    print("="*50)
    print(f"Total molecules: {len(df)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {len(failed_predictions)}")
    
    if failed_predictions:
        print("\nFailed molecules:")
        for mol_id, error in failed_predictions:
            print(f"  - {mol_id}: {error}")
    
    print(f"\nResults saved to: ../../runs/DREAM/{template_yaml_name}_{current_datetime}/")
    
    # Print correlation summary
    if correlations:
        print("\nCORRELATION SUMMARY (Pearson r):")
        print("-" * 30)
        for metric, corr_data in correlations.items():
            print(f"{metric:25}: r={corr_data['pearson_r']:.4f}, p={corr_data['pearson_p']:.4f}, n={corr_data['n_samples']}")
    else:
        print("\nNo correlations calculated.")

if __name__ == "__main__":
    main()