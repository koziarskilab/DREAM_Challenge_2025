import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np

# Load all three datasets
file_path_0 = '../../datasets/DREAM/Val_Dataset_DREAM_0.parquet'
file_path_1 = '../../datasets/DREAM/Val_Dataset_DREAM_1.parquet'
file_path_2 = '../../datasets/DREAM/public_ligends_WDR91_with_fps.csv'

# Load parquet files
data_0 = pd.read_parquet(file_path_0)
data_0 = data_0.rename(columns={'BINARY_LABEL': 'LABEL'})  # For val data
data_1 = pd.read_parquet(file_path_1)

# Load CSV file
data_2 = pd.read_csv(file_path_2)

# Print the column names of all datasets
print("Columns in data_0:", data_0.columns.tolist())
print("Columns in data_1:", data_1.columns.tolist())
print("Columns in data_2:", data_2.columns.tolist())

# Find common columns across all three datasets
common_columns_01 = data_0.columns.intersection(data_1.columns)
common_columns_all = common_columns_01.intersection(data_2.columns)
print("Common columns across all datasets:", common_columns_all.tolist())

# If there are insufficient common columns, we need to standardize the datasets
if len(common_columns_all) < 3:  # Need at least SMILES, LABEL, and one fingerprint column
    print("Insufficient common columns. Standardizing datasets...")
    
    # Standardize column names and select required columns
    def standardize_dataset(df, dataset_name):
        df = df.copy()
        
        # Check for SMILES column variations
        smiles_cols = [col for col in df.columns if 'SMILES' in col.upper()]
        if smiles_cols:
            df = df.rename(columns={smiles_cols[0]: 'SMILES'})
        
        # Check for LABEL column variations
        label_cols = [col for col in df.columns if any(x in col.upper() for x in ['LABEL', 'BINARY', 'ACTIVE'])]
        if label_cols:
            df = df.rename(columns={label_cols[0]: 'LABEL'})
        
        # Check for COMPOUND_ID column variations
        id_cols = [col for col in df.columns if any(x in col.upper() for x in ['ID', 'COMPOUND'])]
        if id_cols:
            df = df.rename(columns={id_cols[0]: 'COMPOUND_ID'})
        elif 'COMPOUND_ID' not in df.columns:
            # Create COMPOUND_ID if it doesn't exist
            df['COMPOUND_ID'] = f"{dataset_name}_" + df.index.astype(str)
        
        # Check for ECFP4 column variations
        ecfp_cols = [col for col in df.columns if 'ECFP' in col.upper()]
        if ecfp_cols:
            df = df.rename(columns={ecfp_cols[0]: 'ECFP4'})
        
        print(f"Standardized columns for {dataset_name}:", df.columns.tolist())
        return df
    
    # Standardize all datasets
    data_0 = standardize_dataset(data_0, "data_0")
    data_1 = standardize_dataset(data_1, "data_1")
    data_2 = standardize_dataset(data_2, "data_2")
    
    # Define required columns
    required_columns = ['SMILES', 'LABEL', 'COMPOUND_ID', 'ECFP4']
    
    # Filter datasets to only include required columns that exist
    def filter_columns(df, required_cols):
        available_cols = [col for col in required_cols if col in df.columns]
        return df[available_cols]
    
    data_0_filtered = filter_columns(data_0, required_columns)
    data_1_filtered = filter_columns(data_1, required_columns)
    data_2_filtered = filter_columns(data_2, required_columns)
    
    print("Filtered columns:")
    print("data_0:", data_0_filtered.columns.tolist())
    print("data_1:", data_1_filtered.columns.tolist())
    print("data_2:", data_2_filtered.columns.tolist())
    
    # Find common columns after standardization
    common_columns = set(data_0_filtered.columns) & set(data_1_filtered.columns) & set(data_2_filtered.columns)
    common_columns = list(common_columns)
    print("Final common columns:", common_columns)
    
    # Merge all three datasets
    if common_columns:
        data = pd.concat([
            data_0_filtered[common_columns], 
            data_1_filtered[common_columns], 
            data_2_filtered[common_columns]
        ], ignore_index=True)
    else:
        print("No common columns found. Using only data_0 and data_1.")
        common_columns_01 = list(set(data_0_filtered.columns) & set(data_1_filtered.columns))
        data = pd.concat([
            data_0_filtered[common_columns_01], 
            data_1_filtered[common_columns_01]
        ], ignore_index=True)
else:
    # Merge all three datasets using common columns
    data = pd.concat([
        data_0[common_columns_all], 
        data_1[common_columns_all], 
        data_2[common_columns_all]
    ], ignore_index=True)

print(f"Combined dataset shape: {data.shape}")
print(f"Combined dataset columns: {data.columns.tolist()}")

# Remove duplicate rows based on the 'SMILES' column
print(f"Before removing duplicates: {len(data)} rows")
data = data.drop_duplicates(subset='SMILES')
print(f"After removing duplicates: {len(data)} rows")

# Check label distribution
print("Label distribution:")
print(data['LABEL'].value_counts())

# Filter active molecules
active_molecules = data[data['LABEL'] == 1].copy()
print(f"Number of active molecules: {len(active_molecules)}")

# Convert ECFP4 fingerprint strings to binary fingerprints
def process_fingerprint(fp_str):
    try:
        # Handle different possible formats
        if isinstance(fp_str, str):
            # Convert string representation to Python object
            fp_data = eval(fp_str)
        elif isinstance(fp_str, (list, tuple, np.ndarray)):
            fp_data = fp_str
        else:
            print(f"Unexpected fingerprint format: {type(fp_str)}")
            return np.zeros((2048,), dtype=int)
        
        # Create binary array
        arr = np.zeros((2048,), dtype=int)
        
        # Set bits for fingerprint indices
        for idx in fp_data:
            arr[int(idx) % 2048] = 1
        
        return arr
    except Exception as e:
        print(f"Error processing fingerprint: {e}")
        return np.zeros((2048,), dtype=int)

# Print a sample to debug
if len(active_molecules) > 0:
    print("Sample ECFP4 data:", active_molecules['ECFP4'].iloc[0])
    print("Sample data type:", type(active_molecules['ECFP4'].iloc[0]))

    # Process fingerprints all at once
    binary_fps = active_molecules['ECFP4'].apply(process_fingerprint)

    # Compute Tanimoto similarity matrix
    def tanimoto_similarity(fp1, fp2):
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    fps = [DataStructs.CreateFromBitString(''.join(map(str, fp))) for fp in binary_fps]
    similarity_matrix = np.array([[tanimoto_similarity(fp1, fp2) for fp2 in fps] for fp1 in fps])
    distance_matrix = 1 - similarity_matrix  # Convert similarity to distance

    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',  # Use 'metric' instead of 'affinity'
        linkage='complete',
        distance_threshold=0.32
    )
    labels = clustering.fit_predict(distance_matrix)

    # Add clustering labels to the active molecules dataframe
    active_molecules['CLUSTER_LABEL'] = labels

    # Merge clustering labels back into the original dataset
    data = data.merge(
        active_molecules[['COMPOUND_ID', 'CLUSTER_LABEL']],
        on='COMPOUND_ID',
        how='left'
    )

    # Inactive molecules get their own cluster label (-1)
    data.loc[data['LABEL'] == 0, 'CLUSTER_LABEL'] = -1

    print(f"Number of clusters: {len(np.unique(labels))}")
    print("Cluster distribution:")
    print(data['CLUSTER_LABEL'].value_counts().sort_index())

else:
    print("No active molecules found for clustering!")
    data['CLUSTER_LABEL'] = -1

# Save the results as a CSV file
output_path = './datasets/DREAM/Val_Dataset_DREAM_combined.csv'
data.to_csv(output_path, index=False)

print(f"Clustering labels saved to {output_path}")
print(f"Final dataset shape: {data.shape}")