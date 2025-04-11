# Import libraries:
import pandas as pd
import numpy as np
import gcsfs
import pandas as pd
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem
import os
from helper import Dataset,ProcessData, SimplifiedDrugFilters,ResultSubmission

df_train = Dataset("./TrainDataset_Aircheck.parquet").get_dataframe()

# Define the ratio of negative samples (N times more than positives)
N = 2  # Adjust this value as needed
# Select all rows where DELLabel == 1 (positive samples)
positive_samples = df_train[df_train["DELLabel"] == 1]


# Select N times more rows where DELLabel == 0 (negative samples)
negative_samples = df_train[df_train["DELLabel"] == 0].sample(n=len(positive_samples) * N, random_state=42)

# Combine both subsets to create a balanced dataset with the desired ratio
df_balanced = pd.concat([positive_samples, negative_samples])

df_train=df_balanced

DevData_path = './DevDataset_Aircheck.csv'

# Load the CSV file into a Pandas DataFrame
df_dev = Dataset(DevData_path).get_dataframe()
df_dev.head()

print('Number of binders:',(df_dev['Label'] == 1).sum())
print('Number of non-binders:',(df_dev['Label'] == 0).sum())
df_dev.head(3)

# Ensure that the file path is correct before running
TestData_path = './TestDataset_Aircheck.csv'

df_test = Dataset(TestData_path).get_dataframe()
print('Number of test compounds', len(df_test))
df_test.head(3)

# Get the list of column names from the DataFrame and print them
column_names_list = df_test.columns.tolist()
print(column_names_list)

fingerprint_columns = ['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'MACCS', 'RDK', 'AVALON', 'ATOMPAIR', 'TOPTOR']
selected_fps = 'ECFP4'  # Replace with desired fingerprints

# Create TrainData and TestData using the selected fingerprints

TrainData = ProcessData(df_train, selected_fps).get_data()
TestData = ProcessData(df_test, selected_fps).get_data()
DevData = ProcessData(df_dev, selected_fps).get_data()

# Creat TrainLabel from the 'DELLabel' column
TrainLabel = df_train['DELLabel']
DevLabel = df_dev['Label']

from lightgbm import LGBMClassifier

# Initialize model with detailed hyperparameters using default values
model = LGBMClassifier(
    n_estimators=100,  # Number of boosting iterations (trees)
    n_jobs=1,  # Number of parallel jobs (1 for no parallelism)
    learning_rate=0.1,  # Learning rate
    max_depth=-1,  # No limit on maximum depth of trees
    min_samples_leaf=20,  # Minimum samples at leaf node
    min_samples_split=2,  # Minimum samples to split node
    lambda_l2=0.0,  # L2 regularization (no regularization)
    lambda_l1=0.0,  # L1 regularization (no regularization)
    num_leaves=31,  # Number of leaves in each tree
    max_bin=255,  # Maximum number of bins
    subsample=1.0,  # Subsample ratio for training data (use all data)
    colsample_bytree=1.0,  # Subsample ratio for features (use all features)
    use_best_model=True,  # Use the best model based on validation performance
    random_state=None,  # Random seed for reproducibility (None for random)
    boosting_type='gbdt',  # Boosting type (Gradient Boosting Decision Tree)
    early_stopping_rounds=None,  # No early stopping
    min_split_gain=0.0,  # Minimum loss reduction required to make a further partition
    ignore_column_check=False  # Do not handle missing values automatically
)

import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score
)

# Function to train the model and compute all classification metrics
def train_model(CrossVal_data_train, CrossVal_data_test, CrossVal_label_train, CrossVal_label_test):
    """Train a LightGBM model and compute accuracy, precision, recall, F1, AUC, MCC, and Kappa."""
    model = LGBMClassifier(random_state=42)
    model.fit(CrossVal_data_train, CrossVal_label_train)

    y_pred = model.predict(CrossVal_data_test)
    y_scores = model.predict_proba(CrossVal_data_test)[:, 1]  # Probability for positive class

    metrics = {
        "Accuracy": accuracy_score(CrossVal_label_test, y_pred),
        "Precision": precision_score(CrossVal_label_test, y_pred, zero_division=0),
        "Recall": recall_score(CrossVal_label_test, y_pred),
        "F1-Score": f1_score(CrossVal_label_test, y_pred),
        "AUC-ROC": roc_auc_score(CrossVal_label_test, y_scores) if len(set(CrossVal_label_test)) > 1 else None,
        "MCC": matthews_corrcoef(CrossVal_label_test, y_pred),
        "Cohen's Kappa": cohen_kappa_score(CrossVal_label_test, y_pred),
    }

    return model, metrics

# Cross-validation
Nfold = 2
TrainData_np = np.array(TrainData)  # Ensure NumPy array
TrainLabel_np = np.array(TrainLabel)
skf = StratifiedKFold(n_splits=Nfold, shuffle=True, random_state=42)
fold_metrics = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(TrainData_np, TrainLabel_np)):
    CrossVal_data_train, CrossVal_data_test = TrainData_np[train_idx], TrainData_np[test_idx]
    CrossVal_label_train, CrossVal_label_test = TrainLabel_np[train_idx], TrainLabel_np[test_idx]

    # Now using the renamed variables for both train and test data
    _, metrics = train_model(CrossVal_data_train, CrossVal_data_test, CrossVal_label_train, CrossVal_label_test)
    fold_metrics.append(metrics)

    # Print fold metrics
    print(f"Fold {fold_idx+1} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-" * 40)

# Compute average metrics across folds
avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0]}

# Print average metrics
print("\nAverage Metrics across all folds:")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")

# Function to train the final model on the entire dataset
def train_final_model(X, y):
    final_model = LGBMClassifier(random_state=42)
    final_model.fit(X, y)
    return final_model

# Training the Final Model on the Full Dataset
final_model = train_final_model(TrainData, TrainLabel)

def evaluate_model(model, X_test):
    """Evaluate the model on the test set and return predictions and probabilities."""
    y_scores = model.predict_proba(X_test)[:, 1]  # Probability for positive class
    return np.round(y_scores, 3)

predictions = evaluate_model(final_model, DevData)
print("Predictions:\n", predictions[1:10], '\n')

Thr=0.5
df_predictions_dev = pd.DataFrame({
    'SMILES':df_dev['SMILES'],
    'PredictedScore': predictions,
    'PredictedLabel': (predictions > Thr).astype(int)  # Convert boolean to int (1 or 0)
})
df_predictions_dev.head(5)

Thr=0.9
df_predictions_dev = pd.DataFrame({
    'SMILES':df_dev['SMILES'],
    'PredictedScore': predictions,
    'PredictedLabel': (predictions > Thr).astype(int)  # Convert boolean to int (1 or 0)
})
df_predictions_dev.head(5)

# Sort by PredictedScore in descending order
df_sorted = df_predictions_dev.sort_values(by="PredictedScore", ascending=False)

# Define function to compute precision, hits, and TIR
def compute_metrics(df_sorted, df_dev, top_n):
    """Compute precision, number of hits, and true identification rate (TIR) for top N predictions."""
    top_n_df = df_sorted.head(top_n)
    merged_n = top_n_df.merge(df_dev, on="SMILES")

    # Precision: Correct predictions / total predictions
    precision = (merged_n["PredictedLabel"] == merged_n["Label"]).mean()

    # Number of Hits: Count of correctly identified positives (TPs)
    hits = ((merged_n["PredictedLabel"] == 1) & (merged_n["Label"] == 1)).sum()

    # True Identification Rate (TIR): Hits / Total Actual Positives in dev set
    total_actual_positives = (df_dev["Label"] == 1).sum()
    tir = hits / total_actual_positives if total_actual_positives > 0 else 0  # Avoid division by zero
    accuracy = (merged_n["PredictedLabel"] == merged_n["Label"]).sum() / len(merged_n)

    return precision, hits, tir, accuracy

# Compute metrics at different thresholds
precision_100, hits_100, tir_100, accuracy_100 = compute_metrics(df_sorted, df_dev, 100)
precision_50, hits_50, tir_50, accuracy_50 = compute_metrics(df_sorted, df_dev, 50)
precision_20, hits_20, tir_20, accuracy_20 = compute_metrics(df_sorted, df_dev, 20)

# Print results
print(f"Precision at Top 100: {precision_100:.4f}, Hits: {hits_100}, TIR: {tir_100:.4f}")
print(f"Accuracy at Top 20:  {accuracy_20:.4f}, Hits: {hits_20}, TIR: {tir_50:.4f}")
print(f"Precision at Top 20:  {precision_20:.4f}, Hits: {hits_20}, TIR: {tir_20:.4f}")

def evaluate_model(model, X_test):
    """Evaluate the model on the test set and return predictions and probabilities."""
    y_scores = model.predict_proba(X_test)[:, 1]  # Probability for positive class
    return np.round(y_scores, 3)

predictions = evaluate_model(final_model, TestData)

# Create a DataFrame with SMILES and prediction scores
Thr=0.6
df_predictions_test = pd.DataFrame({
    'SMILES':df_test['SMILES'],
    'PredictedScore': predictions,
    'PredictedLabel': (predictions > Thr).astype(int)  # Convert boolean to int (1 or 0)
})

df_predictions_test.head(5)

Thr=0.5
# Sort the DataFrame by prediction score in descending order
prediction_df_sorted = df_predictions_test.sort_values(by='PredictedScore', ascending=False)

# Keep only those with score > Thr as Possible Nominees
nominees = prediction_df_sorted[prediction_df_sorted['PredictedScore'] > Thr]

# Get the number of nominees
num_nominees = nominees.shape[0]
print(f"\nNumber of Possible Nominees: {num_nominees}")

# Print the top 10 highest-ranked predictions
print("Top 10 Predictions:")
print(prediction_df_sorted.head(10))