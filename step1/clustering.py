import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np

# Load WDR91 validation sets
file_path_0 = '../datasets/DREAM/Val_Dataset_DREAM_0.parquet'
file_path_1 = '../datasets/DREAM/Val_Dataset_DREAM_1.parquet'
data_0 = pd.read_parquet(file_path_0)
data_0 = data_0.rename(columns={'BINARY_LABEL': 'LABEL'})  # For val data
data_1 = pd.read_parquet(file_path_1)
# Print the column names of data_0 and data_1
print("Columns in data_0:", data_0.columns)
print("Columns in data_1:", data_1.columns)

# Merge data_0 and data_1 on their common columns
common_columns = data_0.columns.intersection(data_1.columns)
data = pd.concat([data_0[common_columns], data_1[common_columns]], ignore_index=True)

# Load LRRK2 validation set
# file_path = '../datasets/DREAM/Val_Dataset_DREAM_0.parquet'
# data = pd.read_parquet(file_path)

# Remove duplicate rows based on the 'SMILES' column
data = data.drop_duplicates(subset='SMILES')

# Filter active molecules
active_molecules = data[data['LABEL'] == 1].copy()

# Convert ECFP4 fingerprint strings to binary fingerprints
def process_fingerprint(fp_str):
    # Convert string representation to Python object
    fp_data = eval(fp_str)
    
    # Create binary array
    arr = np.zeros((2048,), dtype=int)
    
    # Set bits for fingerprint indices
    for idx in fp_data:
        arr[idx % 2048] = 1
    
    return arr

# Print a sample to debug
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

# Perform agglomerative clustering - compatible with older scikit-learn versions
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

# Save the results as a CSV file
output_path = './datasets/DREAM/Val_Dataset_DREAM.csv'
data.to_csv(output_path, index=False)

print(f"Clustering labels saved to {output_path}")
print(f"Number of clusters: {len(np.unique(labels))}")