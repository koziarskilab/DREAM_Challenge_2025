# STEP 2: Machine Learning + 3D Docking

## Background

Step 2 builds a more advanced and chemically meaningful compound selection strategy by integrating AutoML techniques with state-of-the-art imbalanced learning approaches. While the machine learning models in this step still rely exclusively on molecular fingerprints, the overall pipeline is augmented with **3D docking analysis** to enhance the biological relevance and chemical diversity of top-ranked candidates.

Unlike Step 1, which focuses on baseline modeling, Step 2 systematically explores the interaction between molecular fingerprints and machine learning frameworks under class imbalance conditions. The objective is to maximize predictive performance while ensuring that selected compounds are both structurally diverse and chemically plausible.

All data is sourced from [AIRCHECK](https://aircheck.ai/datasets). Beyond the challenge-provided training and test sets, we use an external ASMS validation set of 14,764 compounds (4,704 non-binders and 60 binders). Of the 60 actives, 30 are publicly available ligands — these public ligands are used **only in Step 2 and Step 3**.

### The two halves of Step 2

**Machine learning** (`ML/`) — four interrelated parts that progressively refine the prediction pipeline:

1. Advanced imbalanced learning with MLPs (loss reweighting, Mixup/Remix augmentation, Decoupling, BBN). The DEL dataset is naturally and severely imbalanced, so we adopt algorithm-level mitigation methods from the ImDrug framework. These MLPs improve over their non-imbalanced counterparts but still fall short of traditional ML models, and are therefore **excluded from the final ensemble**.
2. Fingerprint concatenation strategies with AutoML (MACCS, RDK, Avalon, Atom Pair — 15 possible combinations).
3. Ensemble methods over models pretrained on different fingerprint combinations.
4. Test set prediction generation.

**3D docking** (`docking/`) — orthogonal structural validation of the top ML-ranked compounds. Seven WDR91 protein–ligand crystal complexes were downloaded from the PDB (8SHJ, 9DTA, 9DTB, 9EJO, 9EJP, 9MK7, 8T55); 8T55 was excluded because its ligand binds covalently. Cross-docking across six scoring strategies was used to select a generalized protein structure. The final configuration is **8SHJ** with the custom scorer **0.5 × shape_score + 0.5 × esp_score**, applied to the top 5000 ML-predicted molecules (4999 in practice — ID_46910 failed to dock).
