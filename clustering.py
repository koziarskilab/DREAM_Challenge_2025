import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np

# Load the dataset
file_path = './datasets/DREAM/LRRK2_ASMS.parquet'
data = pd.read_parquet(file_path)

# Filter active molecules
active_molecules = data[data['LABEL'] == 1]

# Convert ECFP4 fingerprint strings to binary fingerprints
def process_fingerprint(fp_str):
    # Convert string representation to Python object
    fp_data = eval(fp_str)
    
    # Create binary array
    arr = np.zeros((2048,), dtype=int)
    
    # Handle different possible formats
    if isinstance(fp_data, dict):
        for idx in fp_data:
            arr[idx % 2048] = 1
    elif isinstance(fp_data, (list, tuple)):
        for item in fp_data:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                idx = item[0]
                arr[idx % 2048] = 1
            else:
                arr[item % 2048] = 1
    
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
output_path = './datasets/DREAM/LRRK2_ASMS_clustered.csv'
data.to_csv(output_path, index=False)

print(f"Clustering labels saved to {output_path}")
print(f"Number of clusters: {len(np.unique(labels))}")