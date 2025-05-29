# Step 1: DREAM Challenge 2025 - Baseline AutoML Evaluation

## Overview

Step 1 of the DREAM Challenge 2025 focuses on establishing a robust baseline for compound selection using automated machine learning (AutoML). This step is divided into six main parts:

1. **Part 1**: Data preprocessing and clustering analysis for the DREAM validation dataset
2. **Part 2**: Identifying the optimal evaluation metric in FLAML AutoML that best correlates with the official challenge evaluation metrics
3. **Part 3**: Training baseline AutoML models using the optimal metric identified in Part 2 to provide a strong baseline for model ensembling
4. **Part 4**: Fingerprint correlation analysis to understand the impact of different molecular representations on model predictions
5. **Part 5**: Cross-Fingerprint Model Ensemble
6. **Part 6**: Final Test Set Prediction Generation

## Part 1: Data Preprocessing and Clustering

### Main Script: `clustering.py`

This script performs data preprocessing and clustering analysis on the DREAM Challenge validation dataset to generate cluster labels that align with the Challenge Evaluation Criteria. The script combines validation datasets, removes duplicates, and applies agglomerative clustering to active compounds based on molecular fingerprint similarity to establish the clustering framework used in the official challenge ranking system.

#### Key Features:
- **Data Integration**: Merges validation datasets from multiple parquet files
- **Duplicate Removal**: Eliminates duplicate compounds based on SMILES strings
- **Fingerprint Processing**: Converts ECFP4 fingerprint strings to binary arrays
- **Similarity-Based Clustering**: Uses Tanimoto similarity for molecular clustering
- **Cluster Assignment**: Assigns cluster labels to active compounds for diversity evaluation

#### Challenge Evaluation Criteria

Teams will be ranked based on the following statistics that will appear on the leaderboard for all submitted files (if two teams tie on stat #1, stat #2 is used, etc…):

1. **Number of chemical clusters [1] in the set of 200/50 selected molecules**
2. **Number of chemical clusters in the set of 500/50 selected molecules**
3. **Pseudo PR-AUC based on the fraction of clusters found in the set of selected molecules [2]**
4. **Number of hits in the set of selected molecules**

**[1]** Chemical diversity was measured using Tanimoto similarity between ECFP4 binary molecular fingerprints extracted from the provided ECFP4 count fingerprints. Clustering was performed using the agglomerative clustering algorithm with complete linkage. The clustering cut-off distance of 0.32 was selected based on an analysis of the average silhouette score and silhouette scores distributions, as well as visual analysis of the compounds forming each cluster.

**[2]** X-axis: #selected true positive clusters/# true positive clusters; Y-axis: #selected true positive clusters /# selected compounds

#### Input Datasets:
- `Val_Dataset_DREAM_0.parquet` - First validation dataset partition
- `Val_Dataset_DREAM_1.parquet` - Second validation dataset partition

or

- `LRRK2_ASMS.parquet` - LRRK2 validation set for the proxy evaluation

#### Clustering Algorithm:
- **Method**: Agglomerative Clustering with complete linkage
- **Distance Metric**: Tanimoto distance (1 - Tanimoto similarity)
- **Distance Threshold**: 0.32 (compounds with similarity > 0.68 are grouped together)
- **Fingerprint**: ECFP4 (2048-bit) molecular fingerprints

#### Processing Steps:
1. **Data Loading**: Load and merge validation datasets
2. **Column Standardization**: Rename `BINARY_LABEL` to `LABEL` for consistency
3. **Duplicate Removal**: Remove duplicate compounds based on SMILES
4. **Active Compound Filtering**: Extract compounds with `LABEL == 1`
5. **Fingerprint Conversion**: Process ECFP4 strings to binary arrays
6. **Similarity Calculation**: Compute pairwise Tanimoto similarity matrix
7. **Clustering**: Apply agglomerative clustering with distance threshold
8. **Label Assignment**: Assign cluster labels to active compounds, -1 to inactive compounds

#### Output:
- **`Val_Dataset_DREAM.csv`**: Combined validation dataset with cluster labels
  - Contains all original columns plus `CLUSTER_LABEL`
  - Active compounds: Cluster labels 0, 1, 2, ... (based on clustering results)
  - Inactive compounds: Cluster label -1

or
- **`LRRK2_ASMS_clustered.csv`**: Combined LRRK2 validation dataset with cluster labels


### Usage:
```bash
python3 clustering.py
```

## Part 2: Baseline AutoML Proxy Evaluation

### Main Script: `baseline_automl_proxy.py`

This script systematically evaluates different combinations of fingerprint types, machine learning models, and optimization metrics using FLAML AutoML to identify the best proxy metric for the official DREAM challenge evaluation.

#### Key Features:
- **Multiple Fingerprint Types**: MACCS, RDK, AVALON, ATOMPAIR
- **Various ML Models**: LightGBM, XGBoost, Random Forest, Extra Trees, Histogram-based Gradient Boosting
- **Comprehensive Optimization Metrics**: Various FLAML metrics including accuracy, log_loss, roc_auc, roc_auc_weighted, f1, micro_f1, macro_f1, and ap to find the best proxy for official evaluation
- **Cluster-Aware Evaluation**: Focus on diversity metrics that align with the DREAM challenge objectives

#### Datasets:

We use the LRRK2 dataset for this baseline evaluation because the ASMS WDR91 dataset from the AIRCHECK website contains only 30 actives, which is insufficient for proper clustering analysis. The LRRK2 dataset provides a more robust foundation with:

- **Training**: `LRRK2_DEL.parquet` - Training compounds with binary labels (20,791 actives, 300,000 inactives)
- **Validation**: `LRRK2_ASMS_clustered.csv` - Validation compounds with cluster information (149 actives, 9,987 inactives)

### Parameters

The script accepts the following command-line arguments:

#### Required Arguments:
- `--log_dir`: Directory to save experiment logs and results
- `--fps_type`: Fingerprint type to use for molecular representation
- `--model_type`: Machine learning model type
- `--metric`: FLAML optimization metric

#### Fingerprint Types (`--fps_type`):
- `MACCS` - MACCS keys fingerprints
- `RDK` - RDKit fingerprints
- `AVALON` - Avalon fingerprints
- `ATOMPAIR` - Atom pair fingerprints

#### Model Types (`--model_type`):
- `lgbm` - LightGBM
- `xgboost` - XGBoost with default settings
- `rf` - Random Forest
- `extra_tree` - Extra Trees Classifier
- `histgb` - Histogram-based Gradient Boosting

#### **FLAML Optimization Metrics (`--metric`)**:
- `accuracy` - 1 - accuracy (to minimize)
- `log_loss` - Default for multiclass classification
- `roc_auc` - 1 - roc_auc_score (Default for binary classification)
- `roc_auc_weighted` - ROC AUC with average="weighted"
- `f1` - 1 - f1_score
- `micro_f1` - 1 - f1_score with average="micro"
- `macro_f1` - 1 - f1_score with average="macro"
- `ap` - 1 - average_precision_score (PRAUC)

#### **Evaluation Metrics Calculated**:
1. **Standard Binary Classification Metrics**:
   - **PRAUC** (Precision-Recall Area Under Curve)
   - **ROC-AUC** (Receiver Operating Characteristic Area Under Curve)

2. **Cluster-Based Diversity Metrics** (Official DREAM Challenge Focus):
   - **Hits_50/200/500**: Number of positive compounds in top N selections
   - **Clusters_50/200/500**: Number of unique clusters represented in top N selections
   - **ClusterPRAUC_50/200/500**: Cluster-level precision-recall AUC for top N selections

### Usage
```bash
python3 baseline_automl_proxy.py \
    --log_dir ../runs/experiment_001 \
    --fps_type AVALON \
    --model_type lgbm \
    --metric ap
```

### Output Files

1. **`model_results.csv`**: Comprehensive results table containing all metrics for comparison
2. **`flaml_experiment.log`**: Detailed FLAML training logs for each experiment
3. **`val_predictions.csv`**: Validation set predictions with compound information

## Part 3: Baseline AutoML Model Training

### Main Script: `baseline_automl.py`

This script trains and evaluates machine learning models using FLAML AutoML for compound selection in the DREAM Challenge. It systematically tests different combinations of molecular fingerprints and ML algorithms while calculating both standard classification metrics and cluster-based diversity metrics that align with the official challenge evaluation criteria.

#### Key Features:
- **FLAML AutoML Integration**: Automated hyperparameter tuning and model selection
- **Multiple Fingerprint Support**: MACCS, RDK, AVALON, ATOMPAIR molecular representations
- **Comprehensive Model Coverage**: LightGBM, XGBoost variants, Random Forest, Extra Trees, and more
- **Cluster-Aware Evaluation**: Calculates diversity metrics for top-k compound selections
- **Results Tracking**: Automatic logging and CSV result compilation

#### Input Datasets:
- **Training**: `Train_Dataset_DREAM.parquet` - Training compounds with binary labels
- **Validation**: `Val_Dataset_DREAM.csv` - Validation compounds with cluster information (from Part 1)

#### Parameters

The script accepts the following command-line arguments:

##### Required Arguments:
- `--log_dir`: Directory to save experiment logs and results
- `--fps_type`: Fingerprint type for molecular representation
- `--model_type`: Machine learning model type

##### Fingerprint Types (`--fps_type`):
- `MACCS` - MACCS keys fingerprints (166 bits)
- `RDK` - RDKit fingerprints (2048 bits)
- `AVALON` - Avalon fingerprints (512 bits)
- `ATOMPAIR` - Atom pair fingerprints (2048 bits)

##### Model Types (`--model_type`):
- `lgbm` - LightGBM
- `xgboost` - XGBoost with default settings
- `xgb_limitdepth` - XGBoost with max_depth parameter
- `rf` - Random Forest
- `extra_tree` - Extra Trees Classifier
- `histgb` - Histogram-based Gradient Boosting
- `lrl1` - Logistic Regression with L1 regularization
- `lrl2` - Logistic Regression with L2 regularization
- `catboost` - CatBoost Classifier
- `kneighbor` - K-Nearest Neighbors

#### AutoML Configuration:
- **Time Budget**: 36,000 seconds (10 hours)
- **Optimization Metric**: ROC-AUC weighted
- **Evaluation Method**: 5-fold cross-validation
- **Task**: Binary classification
- **Seed**: 42 (for reproducibility)

#### Evaluation Metrics:

##### Standard Binary Classification Metrics:
- **PRAUC** (Precision-Recall Area Under Curve)
- **ROC-AUC** (Receiver Operating Characteristic Area Under Curve)

##### Cluster-Based Diversity Metrics (Top-k Selections):
- **Hits_Top50/200/500**: Number of positive compounds in top N selections
- **Clusters_Top50/200/500**: Number of unique clusters represented in top N selections
- **ClusterPRAUC_Top50/200/500**: Cluster-level precision-recall AUC for top N selections

#### Processing Steps:
1. **Data Loading**: Load training and validation datasets
2. **Fingerprint Processing**: Convert molecular structures to numerical features
3. **Model Training**: FLAML AutoML with specified configuration
4. **Model Evaluation**: Calculate classification and diversity metrics
5. **Top-k Analysis**: Evaluate compound selections at different thresholds
6. **Results Logging**: Save metrics to CSV and predictions to file

#### Output Files:
- **`model_results.csv`**: Comprehensive results table with all metrics
- **`flaml_experiment.log`**: Detailed FLAML training logs
- **`val_predictions.csv`**: Validation predictions with compound information
- **`best_model.pkl`**: Trained model saved as pickle file

### Usage:
```bash
python3 baseline_automl.py \
    --log_dir ../runs/experiment_001 \
    --fps_type AVALON \
    --model_type lgbm
```

## Part 4: Fingerprint Correlation Analysis

### Main Script: `evaluate_fingerprint_correlation.py`

This script analyzes the correlations between predictions generated by models trained with different molecular fingerprint types. It provides insights into how different molecular representations affect model predictions and helps identify complementary fingerprints for ensemble methods.

#### Key Features:
- **Cross-Fingerprint Analysis**: Compares predictions across MACCS, RDK, AVALON, and ATOMPAIR fingerprints
- **Model-Specific Correlations**: Analyzes how fingerprint choice affects each model type
- **Comprehensive Visualization**: Generates heatmaps and scatter plots for correlation analysis
- **Statistical Summary**: Provides correlation matrices and summary statistics
- **Ensemble Insights**: Identifies complementary fingerprint combinations for model ensembling

#### Input Data Source:
- **Prediction Files**: Loads `val_predictions.csv` from each fingerprint-model combination experiment
- **Base Directory**: `../runs/DREAM/DREAM_BASELINE_SEL_[JOB_ID]/`
- **File Pattern**: `{fps_type}_{model_type}/val_predictions.csv`

#### Parameters

The script accepts the following command-line arguments:

##### Optional Arguments:
- `--output_dir`: Directory to save correlation analysis results (default: `../prediction_correlation_analysis`)
- `--fps_types`: List of fingerprint types to analyze (default: `["MACCS", "RDK", "AVALON", "ATOMPAIR"]`)
- `--model_types`: List of model types to analyze (default: `["lgbm", "xgboost", "rf", "extra_tree", "histgb"]`)

##### Fingerprint Types Analyzed:
- `MACCS` - MACCS keys fingerprints
- `RDK` - RDKit fingerprints  
- `AVALON` - Avalon fingerprints
- `ATOMPAIR` - Atom pair fingerprints

##### Model Types Analyzed:
- `lgbm` - LightGBM
- `xgboost` - XGBoost with default settings
- `rf` - Random Forest
- `extra_tree` - Extra Trees Classifier
- `histgb` - Histogram-based Gradient Boosting
- `xgb_limitdepth` - XGBoost with max_depth parameter
- `lrl1` - Logistic Regression with L1 regularization
- `catboost` - CatBoost Classifier
- `kneighbor` - K-Nearest Neighbors

#### Analysis Components:

##### 1. **Prediction Merging**:
- Combines predictions from all fingerprint-model combinations
- Ensures consistent molecule set across all comparisons
- Creates unified prediction matrix for correlation analysis

##### 2. **Overall Correlation Analysis**:
- Calculates Pearson correlation between all prediction pairs
- Generates comprehensive correlation heatmap
- Identifies highly correlated and uncorrelated prediction pairs

##### 3. **Fingerprint-Level Analysis**:
- Computes average correlations between fingerprint types
- Creates fingerprint-specific correlation matrices
- Evaluates fingerprint complementarity for ensemble methods

##### 4. **Model-Specific Analysis**:
- Analyzes how each model responds to different fingerprints
- Generates model-specific correlation plots
- Identifies model-fingerprint interactions

##### 5. **Scatter Plot Analysis**:
- Creates scatter plots for highest and lowest correlation pairs
- Visualizes prediction relationships across fingerprint types
- Helps identify linear and non-linear correlation patterns

#### Output Files:
- **`merged_predictions.csv`**: Combined predictions from all models and fingerprints
- **`prediction_correlations.csv`**: Full correlation matrix between all prediction pairs
- **`fingerprint_prediction_correlation.csv`**: Average correlations between fingerprint types
- **`prediction_correlations_heatmap.png`**: Comprehensive correlation heatmap
- **`fingerprint_prediction_correlation.png`**: Fingerprint-level correlation heatmap
- **`{model}_fingerprint_correlation.png`**: Model-specific correlation plots
- **`{model}_fingerprint_correlation.csv`**: Model-specific correlation data
- **`scatter_plots/`**: Directory containing scatter plots for high/low correlation pairs

### Usage:
```bash
python3 evaluate_fingerprint_correlation.py \
    --output_dir ../correlation_analysis \
    --fps_types MACCS RDK AVALON ATOMPAIR \
    --model_types lgbm xgboost rf extra_tree histgb
```

## Part 5: Cross-Fingerprint Model Ensemble

### Main Script: `model_ensemble_cross_fingerprints.py`

This script creates ensemble models by combining predictions from models trained with different molecular fingerprint types. It systematically evaluates combinations of predefined low-correlation model pairs to identify optimal ensemble configurations that maximize both classification performance and chemical diversity metrics.

#### Key Features:
- **Predefined Low-Correlation Pairs**: Uses 8 carefully selected model pairs with low prediction correlations
- **Multiple Ensemble Methods**: Supports various ensemble techniques from simple averaging to advanced stacking
- **Hierarchical Evaluation**: Tests combinations from single pairs up to all available pairs
- **Cluster-Aware Weighting**: Implements cluster-based weighting that prioritizes diversity metrics
- **Comprehensive Metrics**: Evaluates both standard classification and DREAM challenge-specific metrics

#### Predefined Model Pairs:
The script uses 8 strategically selected low-correlation pairs identified from correlation analysis:

1. **AVALON_xgboost & MACCS_kneighbor**
2. **AVALON_rf & MACCS_kneighbor**  
3. **ATOMPAIR_rf & AVALON_xgb_limitdepth**
4. **AVALON_histgb & ATOMPAIR_xgboost**
5. **AVALON_extra_tree & ATOMPAIR_histgb**
6. **RDK_catboost & ATOMPAIR_lgbm**
7. **ATOMPAIR_catboost & RDK_extra_tree**
8. **ATOMPAIR_lgbm & RDK_histgb**

#### Input Data Source:
- **Pretrained Models**: Loads saved models from `../runs/DREAM/DREAM_BASELINE_SEL/{fps_type}_{model_type}/best_model.pkl`
- **Training Data**: `Train_Dataset_DREAM.parquet` - For stacking ensemble methods
- **Validation Data**: `Val_Dataset_DREAM.csv` - For ensemble evaluation

#### Parameters

The script accepts the following command-line arguments:

##### Required Arguments:
- `--log_dir`: Directory to save ensemble results and logs

##### Optional Arguments:
- `--max_pairs`: Maximum number of pairs to combine in ensembles (default: 4)
- `--ensemble_method`: Ensemble technique to use (default: `simple_average`)

##### Ensemble Methods (`--ensemble_method`):
- `simple_average` - Arithmetic mean of model predictions
- `weighted_average_clusters` - Cluster-performance weighted averaging
- `majority_vote` - Hard prediction majority voting
- `max_vote` - Maximum probability across models
- `min_vote` - Minimum probability across models
- `median_vote` - Median probability across models
- `stacking_logistic` - Logistic regression meta-learner
- `stacking_rf` - Random Forest meta-learner

#### Ensemble Process:

##### 1. **Model Loading**:
- Loads pretrained models for each fingerprint-model combination
- Verifies model availability and compatibility
- Generates predictions using respective fingerprint types

##### 2. **Combination Generation**:
- Creates all possible combinations from 1 to `max_pairs` 
- Flattens pair combinations to unique model sets
- Ensures minimum 2 models per ensemble

##### 3. **Weighting Calculation** (for weighted methods):
- **Hierarchical Cluster Scoring**: Prioritizes Top50 > Top200 > Top500 cluster metrics
- **Score Formula**: `clusters_50 × 1,000,000 + clusters_200 × 1,000 + clusters_500`
- **Weight Normalization**: Converts scores to probability weights summing to 1

##### 4. **Prediction Ensemble**:
- Combines individual model predictions using selected method
- Handles both probability and hard prediction ensemble techniques
- Supports meta-learning approaches for advanced stacking

##### 5. **Comprehensive Evaluation**:
- Calculates standard metrics (PRAUC, ROC-AUC)
- Computes diversity metrics for Top50/200/500 selections
- Generates cluster-level precision-recall curves

#### Evaluation Metrics:

##### Standard Classification Metrics:
- **PRAUC** (Precision-Recall Area Under Curve)
- **ROC-AUC** (Receiver Operating Characteristic Area Under Curve)

##### DREAM Challenge Diversity Metrics:
- **Hits_Top50/200/500**: Number of active compounds in top N selections
- **Clusters_Top50/200/500**: Number of unique clusters in top N selections  
- **ClusterPRAUC_Top50/200/500**: Cluster-level precision-recall AUC

#### Processing Steps:
1. **Model Loading**: Load all pretrained models with their fingerprint types
2. **Data Preparation**: Process validation/training data for each fingerprint type
3. **Combination Enumeration**: Generate all pair combinations up to max_pairs
4. **Ensemble Training**: Create ensemble predictions using specified method
5. **Metric Calculation**: Evaluate comprehensive performance metrics
6. **Results Logging**: Save predictions, metrics, and ensemble configurations

#### Output Files:
- **`combination_summary.csv`**: Comprehensive results for all ensemble combinations
- **`{combination_name}/val_predictions.csv`**: Ensemble predictions for each combination
- **`{combination_name}/individual_predictions.csv`**: Individual model predictions with ensemble
- **`{combination_name}/ensemble_info.pkl`**: Ensemble configuration and metadata

### Usage:
```bash
python3 model_ensemble_cross_fingerprints.py \
    --log_dir ../ensemble_results \
    --max_pairs 4 \
    --ensemble_method simple_average
```

## Part 6: Final Test Set Prediction Generation

### Main Script: `test_step1.py`

This script generates final predictions for the DREAM Challenge Step 1 test set using the optimal ensemble model identified from the previous analysis steps. It implements the best-performing ensemble configuration to create submission-ready predictions with the required format for the DREAM Challenge evaluation.

#### Key Features:
- **Best Ensemble Implementation**: Uses the optimal 6-pair ensemble (12 models total) identified from systematic evaluation
- **Median Vote Strategy**: Applies robust median ensemble method for final predictions
- **DREAM Format Compliance**: Generates predictions in the exact format required for challenge submission
- **Top-K Selection**: Automatically identifies top 200 and top 500 compounds for diversity evaluation
- **Comprehensive Logging**: Provides detailed statistics and validation of generated predictions

#### Optimal Ensemble Configuration:
The script implements the best-performing ensemble identified from correlation analysis and performance evaluation:

**6 Model Pairs (12 Individual Models)**:
1. **ATOMPAIR_rf & AVALON_xgb_limitdepth**
2. **AVALON_histgb & ATOMPAIR_xgboost**
3. **AVALON_extra_tree & ATOMPAIR_histgb**
4. **RDK_catboost & ATOMPAIR_lgbm**
5. **ATOMPAIR_catboost & RDK_extra_tree**
6. **ATOMPAIR_lgbm & RDK_histgb**

#### Input Data Source:
- **Test Dataset**: `../datasets/DREAM/Step1_TestData_Target2035.parquet`
- **Pretrained Models**: Loads from `../runs/DREAM/DREAM_BASELINE_SEL/{fps_type}_{model_type}/best_model.pkl`
- **Model Selection**: Based on optimal ensemble from Part 5 analysis

#### Output Files:

##### Primary Submission File:
- **`TeamKoziarskiLab.csv`**: Official DREAM Challenge submission file
  - **RandomID**: Unique compound identifier from test set
  - **Sel_200**: Binary indicator (1/0) for top 200 selection
  - **Sel_500**: Binary indicator (1/0) for top 500 selection  
  - **Score**: Ensemble probability score (0-1 range)

### Usage:
```bash
python3 test_step1.py
```

