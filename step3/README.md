# STEP 3: Test-Set Screening + Prediction Generation (Enamine)

## Goal
Step 3 produces the **DREAM Step 3 submission file** by:
1) preparing a filtered screening dataset from the **Enamine SDF**,  
2) scoring the filtered compounds using **pretrained Step 2 fingerprint models** with a **max-vote ensemble**, and  
3) re-ranking the top 5000 molecules using **Boltz-2 structure prediction** and **molecular docking**.

**Final output (Top 5000):** `Catalog_ID, SMILES, Molecular_Weight, aLogP, Score`  
Saved as: `../runs/DREAM/step3/TeamKoziarskiLab_Step3.csv`

**Re-ranking formula (applied manually):**  
`final_score = 0.7 × boltz_confidence_score + 0.3 × docking_score`

> Step 3 does **not** train new models. It reuses the best Step 2 models and refines predictions with structure-based methods.

---

## What's in this folder
- `test_set_prep.py` — **SDF → filtered CSV**, removes known actives, applies MW/aLogP filters
- `test_ml_step3.py` — loads pretrained models, generates predictions in chunks, **max-vote ensemble**, writes Top 5000
- **`boltz/`** — Boltz-2 structure prediction for re-ranking
  - `batch_predict.py` — Boltz-2 batch prediction for top 5000 compounds
  - `config/wdr91_8hsj_pocket_template.yaml` — WDR91 pocket template (skips co-folding)
- **`docking/`** — Molecular docking for re-ranking

---

## Part A — Prepare the Step 3 screening dataset

### Inputs
- Enamine SDF: `../datasets/DREAM/Enamine_screening_collection_202506.sdf`
- Known actives (excluded): `../datasets/DREAM/known_active_molecules.csv` (must contain a `SMILES` column)

### Filters applied
- `Molecular_Weight < 500`
- `aLogP < 4.0`  (RDKit `Descriptors.MolLogP`)

### Output
- `../datasets/DREAM/Test_Step3_Dataset_DREAM.csv` with columns:
  - `Catalog_ID`
  - `SMILES`
  - `Molecular_Weight`
  - `aLogP`

### Run
```bash
python3 test_set_prep.py
```

---

## Part B — Generate Step 3 predictions (pretrained Step 2 models)

### Required input dataset
This script expects a fingerprint-ready CSV at:
- `../datasets/DREAM/Test_Step3_Dataset_DREAM_w_FP.csv`

At minimum it uses these columns:
- `Catalog_ID`, `SMILES`, `Molecular_Weight`, `aLogP`

> If you only have `Test_Step3_Dataset_DREAM.csv`, you must generate the `_w_FP.csv` file using the same preprocessing/fingerprint pipeline as in Step 2 (via `helper.ProcessData`).

### Models used (must exist)
The script loads these pretrained Step 2 models:
- `../runs/DREAM/step2/histgb/RDK_AVALON_ATOMPAIR/best_model.pkl`
- `../runs/DREAM/step2/lgbm/MACCS_RDK_AVALON/best_model.pkl`
- `../runs/DREAM/step2/xgb_limitdepth/MACCS_ATOMPAIR/best_model.pkl`
- `../runs/DREAM/step2/xgboost/MACCS_RDK_AVALON_ATOMPAIR/best_model.pkl`

### Ensemble method
- **max-vote** over probabilities:  
  `Score = max(p_model_1, p_model_2, p_model_3, p_model_4)`

### Outputs
Written to `../runs/DREAM/step3/`:
- **Initial Top 5000:** `top5000_ml_predictions.csv`
- Per-model scores (Top 5000): `test_predictions_individual_models_step3.csv`
- Run metadata: `ensemble_info.pkl`

### Run
```bash
python3 test_ml_step3.py
```

---

## Part C — Re-rank top 5000 with Boltz-2 structure prediction

### Script
`boltz/batch_predict.py`

### Input
- Top 5000 from Part B: `../runs/DREAM/step3/top5000_ml_predictions.csv`

### Template configuration
Uses pre-configured template:
- `boltz/config/wdr91_8hsj_pocket_template.yaml`

**Key features:**
- **Skip co-folding:** Template provides pre-defined WDR91 structure (PDB: 8HSJ)
- **Pocket specification:** Binding pocket coordinates are pre-defined
- **Confidence scoring:** Boltz-2 `confidence_score` correlates with binding affinity

### Output
- `../runs/DREAM/step3/boltz_scores.csv` containing:
  - `Catalog_ID`, `boltz_confidence_score`

### Run
```bash
cd boltz
python batch_predict.py \
  --csv_file ../runs/DREAM/step3/top5000_ml_predictions.csv \
  --template_yaml config/wdr91_8hsj_pocket_template.yaml
```

---

## Part D — Re-rank top 5000 with molecular docking


## Re-ranking formula
```python
final_score = 0.7 × boltz_confidence_score + 0.3 × docking_score
```

