# STEP 3: Test-Set Screening + Prediction Generation (Enamine)

## Goal
Step 3 produces the **DREAM Step 3 submission file** by:
1) preparing a filtered screening dataset from the **Enamine SDF**, and  
2) scoring the filtered compounds using **pretrained Step 2 fingerprint models** with a **max-vote ensemble**.

**Final output (Top 5000):** `Catalog_ID, SMILES, Molecular_Weight, aLogP, Score`  
Saved as: `../runs/DREAM/step3/TeamKoziarskiLab_Step3.csv`

> Step 3 does **not** train new models. It reuses the best Step 2 models (`../runs/DREAM/step2/.../best_model.pkl`).

---

## What’s in this folder
- `test_set_prep.py` — **SDF → filtered CSV**, removes known actives, applies MW/aLogP filters
- `test_ml_step3.py` — loads pretrained models, generates predictions in chunks, **max-vote ensemble**, writes Top 5000

---

## Part A — Prepare the Step 3 screening dataset

### Script
`test_set_prep.py`

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

### Script
`test_ml_step3.py`

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
- **Submission (Top 5000):** `TeamKoziarskiLab_Step3.csv`
- Per-model scores (Top 5000): `test_predictions_individual_models_step3.csv`
- Run metadata: `ensemble_info.pkl`

### Run
```bash
python3 test_ml_step3.py
```

---

## Notes / troubleshooting
- **Missing model file:** verify Step 2 has been run and the path matches the expected layout under `../runs/DREAM/step2/`.
- **Memory:** predictions run in chunks. If needed, reduce `chunk_size` in `process_models_in_chunks(..., chunk_size=...)`.
- **RDKit required:** `test_set_prep.py` needs RDKit. `test_ml_step3.py` also needs whatever dependencies `helper.ProcessData` uses to compute