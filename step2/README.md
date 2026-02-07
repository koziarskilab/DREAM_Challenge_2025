# STEP 2: Model with FINGERPRINTS+ SMILES

## Overview
In Step 2, we aim to build a more advanced and chemically meaningful compound selection strategy by integrating AutoML techniques with state-of-the-art imbalanced learning approaches. While the machine learning models in this step still rely exclusively on molecular fingerprints, we augment the overall pipeline with 3D docking analysis to enhance the biological relevance and chemical diversity of top-ranked candidates. Unlike Step 1, which focuses on baseline modeling, Step 2 systematically explores the interaction between molecular fingerprints and machine learning frameworks under class imbalance conditions. The objective is to maximize predictive performance while ensuring that selected compounds are both structurally diverse and chemically plausible. This step is organized into four interrelated parts that progressively refine the prediction and selection pipeline.

1. **Part 1**: Advanced imbalanced learning with deep neural networks for compound classification under severe class imbalance
2. **Part 2**: Fingerprint combination strategies using AutoML for enhanced molecular representation
3. **Part 3**: Ensemble methods combining multiple fingerprint representations and machine learning approaches
4. **Part 4**: 3D molecular docking analysis for biological validation of top-ranked compounds
5. **Part 5**: Test set prediction generation 

## Part 1: Advanced Imbalanced Learning with Multi-Layer Perceptrons

### Main Script: `baseline_imbalanced_learning.py`

Part 1 investigates the effectiveness of advanced imbalanced learning strategies when applied to multi-layer perceptrons (MLPs) for compound classification. We include these imbalanced learning methods because the DEL dataset is naturally and severely imbalanced, and we aim to systematically evaluate how classical imbalance-handling techniques affect performance under such conditions. Rather than relying solely on basic resampling techniques, we adopt a suite of algorithm-level imbalance mitigation methods introduced in the ImDrug framework.

#### Key Features:
- **Advanced Loss Functions**: Class-Balanced Focal Loss, Balanced Softmax, Influence-Balanced Loss, Class-Dependent Temperatures
- **Data Augmentation**: Mixup and Remix techniques for minority class enhancement
- **Specialized Architectures**: Bilateral-Branch Network (BBN) for imbalanced learning
- **Two-Stage Training**: Decoupling representation learning and classifier training
- **Adaptive Network Architecture**: Dynamic hidden layer sizing based on input dimensions
- **Comprehensive Evaluation**: Both standard metrics and cluster-based diversity metrics

#### Imbalanced Learning Methods (`--imbalanced`):

The evaluated methods include:

##### Advanced Loss Functions:
- `CE` - **Standard Cross-Entropy**: Traditional loss function for comparison
- `CB_F` - **Class-Balanced Focal Loss**: Combines focal loss with class-balanced weighting using effective number of samples
- `BS` - **Balanced Softmax**: Adjusts logits based on class frequencies during inference
- `CB_CE` - **Class-Balanced Cross-Entropy**: Reweights cross-entropy loss using effective number of samples
- `CS` - **Cost-Sensitive Cross-Entropy**: Uses inverse frequency weighting with adjustable gamma parameter
- `IB` - **Influence-Balanced Loss**: Weights samples based on gradient and feature influence
- `CDT` - **Class-Dependent Temperatures**: Applies different temperature scaling per class

##### Data Augmentation Methods:
- `MIXUP` - **Mixup Augmentation**: Linear interpolation between samples and labels
- `REMIX` - **Remix Augmentation**: Adaptive mixup with minority class protection

##### Advanced Training Strategies:
- `DECOUPLING` - **Decoupling Representation and Classifier**: Two-stage training separating feature learning and classification
- `BBN` - **Bilateral-Branch Network**: Dual-stream architecture for balanced and imbalanced data processing

#### Performance Analysis:
These techniques aim to address data imbalance either through reweighting the loss function, augmenting data representations, or decoupling feature and label distributions. Each method is trained with a single type of fingerprint, with performance measured using cluster-based metrics. Our findings highlight the sensitivity of MLPs to imbalance and underscore the importance of imbalanced learning, particularly in low-hit-rate datasets like WDR91. However, since SMILES strings are not available for the training set, we are restricted to using MLPs with molecular fingerprints, which limits the model's representational capacity. Although MLPs with imbalanced learning show improvement over their non-imbalanced counterparts, their performance still falls short compared to traditional machine learning models. As a result, we do not incorporate these MLP-based models into the subsequent ensemble stage, which prioritizes higher-performing models trained on diverse fingerprint combinations using more robust algorithms.

#### Input Datasets:
- **Training**: `Train_Dataset_DREAM.parquet` - Training compounds with binary labels
- **Validation**: `Val_Dataset_DREAM_Step2.csv` - Validation compounds with cluster information

#### Example Usage:
```bash
# Single fingerprint with Class-Balanced Focal Loss
python3 baseline_imbalanced_learning.py \
    --log_dir ../runs/DREAM/experiment_cb_focal \
    --fps_type AVALON \
    --imbalanced CB_F

# Bilateral-Branch Network with multiple fingerprints
python3 baseline_imbalanced_learning.py \
    --log_dir ../runs/DREAM/experiment_bbn \
    --fps_type MACCS,RDK,AVALON \
    --imbalanced BBN
```

## Part 2: Fingerprint Combination Strategies with AutoML

### Main Script: `baseline_fingerprint_concatenate.py`

Part 2 explores the effects of different fingerprint combinations on model performance. We concatenate various molecular fingerprints, including MACCS, Avalon, Atom Pair, and RDK, to form hybrid descriptors that capture both structural and topological features. By systematically evaluating different fingerprint combinations (15 possible combinations in total), we identify combinations that offer complementary information, resulting in more expressive molecular representations.

#### Key Features:
- **Fingerprint Concatenation**: Combines multiple molecular fingerprint types into unified representations
- **AutoML Integration**: FLAML AutoML for automated hyperparameter optimization
- **Systematic Evaluation**: Tests all possible fingerprint combinations (15 total combinations)
- **Comprehensive Model Support**: LightGBM, XGBoost variants, Random Forest, and more
- **Performance Tracking**: Detailed logging and CSV result compilation

#### Fingerprint Combinations:
- **Single**: MACCS, RDK, AVALON, ATOMPAIR (4 combinations)
- **Pairs**: All 6 possible pairs (e.g., MACCS+RDK, AVALON+ATOMPAIR)
- **Triplets**: All 4 possible triplets (e.g., MACCS+RDK+AVALON)
- **Quadruplet**: All four fingerprints combined (MACCS+RDK+AVALON+ATOMPAIR)

#### Model Types (`--model_type`):
- `lgbm` - LightGBM
- `xgboost` - XGBoost with default settings
- `xgb_limitdepth` - XGBoost with max_depth parameter
- `rf` - Random Forest
- `extra_tree` - Extra Trees Classifier
- `histgb` - Histogram-based Gradient Boosting
- `lrl1` - Logistic Regression with L1 regularization
- `catboost` - CatBoost Classifier
- `kneighbor` - K-Nearest Neighbors

#### Input Datasets:
- **Training**: `Train_Dataset_DREAM.parquet` - Training compounds with binary labels
- **Validation**: `Val_Dataset_DREAM_Step2.csv` - Validation compounds with cluster information

#### Example Usage:
```bash
# Single fingerprint
python3 baseline_fingerprint_concatenate.py \
    --log_dir ../runs/experiment_single \
    --fps_type AVALON \
    --model_type lgbm \
    --time_budget 3600

# Multiple fingerprint combination
python3 baseline_fingerprint_concatenate.py \
    --log_dir ../runs/experiment_combo \
    --fps_type MACCS,RDK,AVALON,ATOMPAIR \
    --model_type xgb_limitdepth \
    --time_budget 18000
```


## Part 3: Ensemble Methods for Fingerprint Combinations

### Main Script: `model_ensemble.py`

Similar to Step 1, Part 3 implements ensemble learning using machine learning models pretrained on different fingerprint combinations. We construct model ensembles by aggregating predictions from diverse base learners, leveraging their unique perspectives on molecular similarity. Voting and averaging strategies are employed to combine model outputs, with ensemble performance measured against individual model baselines.

#### Key Features:
- **Pre-trained Model Integration**: Uses optimal fingerprint-model combinations
- **Multiple Ensemble Methods**: Simple averaging, weighted voting, stacking approaches
- **Meta-Learning Support**: Logistic regression and Random Forest meta-learners
- **Cluster-Aware Weighting**: Prioritizes diversity metrics in ensemble weighting
- **Comprehensive Evaluation**: Both classification and diversity metrics

#### Selected Model Combinations:
The script uses strategically chosen models based on performance analysis from Part 2:

1. **histgb with RDK_AVALON_ATOMPAIR** - Histogram Gradient Boosting with 3-fingerprint combination
2. **lgbm with MACCS_RDK_AVALON** - LightGBM with complementary 3-fingerprint set
3. **xgb_limitdepth with MACCS_ATOMPAIR** - XGBoost with focused 2-fingerprint combination
4. **xgboost with MACCS_RDK_AVALON_ATOMPAIR** - XGBoost with all fingerprints

#### Ensemble Methods (`--ensemble_method`):

##### Simple Voting Methods:
- `simple_average` - Arithmetic mean of model predictions
- `majority_vote` - Hard prediction majority voting
- `max_vote` - Maximum probability across models
- `min_vote` - Minimum probability across models
- `median_vote` - Median probability across models

##### Weighted Methods:
- `weighted_average_clusters` - Cluster-performance weighted averaging

##### Advanced Meta-Learning:
- `stacking_logistic` - Logistic regression meta-learner trained on base model predictions
- `stacking_rf` - Random Forest meta-learner for non-linear combination

#### Example Usage:
```bash
# Simple averaging ensemble
python3 model_ensemble.py \
    --log_dir ../runs/ensemble_simple \
    --ensemble_method simple_average

# Stacking with logistic regression meta-learner
python3 model_ensemble.py \
    --log_dir ../runs/ensemble_stacking \
    --ensemble_method stacking_logistic
```

## Part 4: 3D Molecular Docking Analysis

Part 4 introduces a structural biology component by performing 3D molecular docking on the top 5000 compounds ranked by the best-performing ensemble model. Docking scores are computed to assess potential binding affinity to the WDR91 protein, providing an orthogonal validation signal to the ligand-based prediction models.

### Docking Protocol:

#### Protein Structures:
We downloaded seven publicly available protein–ligand crystal complexes from the Protein Data Bank (PDB): 8SHJ, 9DTA, 9DTB, 9EJO, 9EJP, 9MK7, and 8T55. Among them, the ligand in 8T55 binds covalently, while the ligands in the other complexes bind non-covalently; thus, 8T55 was excluded from the study.

#### Docking Software:
- **Program**: AutoDock Vina with both default and customized scoring functions
- **Target Selection**: Cross-docking performed to select generalized protein structure

#### Cross-Docking Protocol:
We performed cross-docking to select a generalized protein (i.e., a protein capable of accommodating diverse chemical ligands) for the main docking pipeline. During cross-docking, we tested six docking protocols differing in their scoring strategies:

1. **Default Vina Score**: Standard AutoDock Vina scoring
2. **Shape-Based Scoring**: `shape_score` - geometric similarity to crystal ligands
3. **Electrostatics-Based Scoring**: `esp_score` - electrostatic similarity to crystal ligands
4. **Weighted Combinations**: 
   - `0.4 × shape_score + 0.6 × esp_score`
   - `0.5 × shape_score + 0.5 × esp_score`
   - `0.6 × shape_score + 0.4 × esp_score`

#### Selected Docking Configuration:
- **Protein Structure**: 8SHJ
- **Scoring Function**: `0.5 × shape_score + 0.5 × esp_score`
- **Target Compounds**: Top 5000 molecules predicted by ML pipeline (actual number: 4999, ID_46910 failed to dock)

#### Interaction Analysis:
After docking, we manually analyzed the interactions of the top 500 docked molecules and selected ligands that showed interaction patterns similar to those formed by the six crystal ligands. Key interactions include:

- **Hydrogen Bonds**: THR460, LYS461, ARG462, and THR532
- **Hydrophobic Interactions**: Fitting into the hydrophobic pocket composed of LEU465, LEU467, LEU477, and ALA459

### Docking Results:
The 3D docking results provide structural validation for the machine learning predictions and help identify compounds with favorable binding poses and interaction profiles similar to known WDR91 ligands.

**Note**: Implementation details and code for the docking pipeline will be provided in future update.

## Part 5: Test Set Prediction Generation

**Note**: Part 5 should be easy to be implemented by users themselves.