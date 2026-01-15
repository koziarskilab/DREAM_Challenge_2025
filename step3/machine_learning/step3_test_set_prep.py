from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors
import pandas as pd
import numpy as np
from tqdm import tqdm

def compute_molecular_properties(smiles):
    """Compute molecular weight and aLogP for a given SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        alogp = Descriptors.MolLogP(mol)
    else:
        mw = np.nan
        alogp = np.nan
    return mw, alogp

# Read the SDF file
sdf_file = "/h/yfjiang/research/DREAM_Challenge_2025/datasets/DREAM/datasets/DREAM/Enamine_screening_collection_202506.sdf"

print("Loading SDF file...")
df = PandasTools.LoadSDF(sdf_file)
print(f"Loaded {len(df)} molecules from SDF file")

# Load known active molecules to exclude
known_actives_file = "/h/yfjiang/research/DREAM_Challenge_2025/datasets/DREAM/known_active_molecules.csv"
known_actives_df = pd.read_csv(known_actives_file)
known_active_smiles = set(known_actives_df['SMILES'].tolist())
print(f"Loaded {len(known_active_smiles)} known active molecules to exclude")

# Extract SMILES from the DataFrame
# The RDKit mol objects should be in a column, let's check the column names
print("DataFrame columns:", df.columns.tolist())

# Convert RDKit mol objects to SMILES if needed
if 'ROMol' in df.columns:
    df['SMILES'] = df['ROMol'].apply(lambda mol: Chem.MolToSmiles(mol) if mol is not None else None)
elif 'SMILES' not in df.columns:
    # If no SMILES column exists, we need to extract from mol objects
    # Assuming the first column contains mol objects
    mol_column = df.columns[0]
    df['SMILES'] = df[mol_column].apply(lambda mol: Chem.MolToSmiles(mol) if mol is not None else None)

# Remove rows with invalid SMILES
df = df.dropna(subset=['SMILES'])
print(f"After removing invalid SMILES: {len(df)} molecules")

# Exclude known active molecules
initial_count = len(df)
df = df[~df['SMILES'].isin(known_active_smiles)]
excluded_count = initial_count - len(df)
print(f"Excluded {excluded_count} known active molecules. Remaining: {len(df)} molecules")

# Compute molecular properties for all molecules
print("Computing molecular properties...")
properties_list = []
valid_indices = []

for idx, smiles in tqdm(enumerate(df['SMILES']), total=len(df), desc="Processing molecules"):
    mw, alogp = compute_molecular_properties(smiles)
    
    # Apply filters: MW < 500 Da and aLogP < 4.0
    if not np.isnan(mw) and not np.isnan(alogp) and mw < 500 and alogp < 4.0:
        properties_list.append({
            'Catalog_ID': df.iloc[idx].get('Catalog_ID', f'Unknown_{idx}'),
            'SMILES': smiles,
            'Molecular_Weight': mw,
            'aLogP': alogp
        })
        valid_indices.append(idx)

print(f"Filtered molecules based on MW < 500 and aLogP < 4.0: {len(properties_list)} molecules")

# Create final DataFrame with results
filtered_df = pd.DataFrame(properties_list)

# Save the results with SMILES included
output_file = "/h/yfjiang/research/DREAM_Challenge_2025/datasets/DREAM/datasets/DREAM/Test_Step3_Dataset_DREAM.csv"
filtered_df[['Catalog_ID', 'SMILES', 'Molecular_Weight', 'aLogP']].to_csv(output_file, index=False)

print(f"Screened results saved to {output_file}")
print(f"Final dataset contains {len(filtered_df)} molecules")

# Display summary statistics
if len(filtered_df) > 0:
    print("\nSummary Statistics:")
    print(f"Molecular Weight - Min: {filtered_df['Molecular_Weight'].min():.2f}, Max: {filtered_df['Molecular_Weight'].max():.2f}, Mean: {filtered_df['Molecular_Weight'].mean():.2f}")
    print(f"aLogP - Min: {filtered_df['aLogP'].min():.2f}, Max: {filtered_df['aLogP'].max():.2f}, Mean: {filtered_df['aLogP'].mean():.2f}")
    
    print("\nFirst 5 filtered molecules:")
    print(filtered_df[['Catalog_ID', 'SMILES', 'Molecular_Weight', 'aLogP']].head())