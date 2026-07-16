# STEP 3: Boltz-2 part

### Script
`batch_predict.py`

### Required input dataset
- `../datasets/DREAM/ML_top5K.csv` containing:
  - `SMILES` column
  - `LABEL` column to calculate the metric-label correlation

### Template configuration
The script uses a pre-configured template at:
- `config/wdr91_8hsj_pocket_template.yaml`

**Key features:**
- **Skip co-folding:** Template provides pre-defined WDR91 structure (PDB: 8HSJ)
- **Pocket specification:** Binding pocket coordinates are pre-defined in template
- **Best metric:** Boltz-2 outputs a `confidence_score` that correlates with binding affinity the best

### Run
```bash
python batch_predict.py \
  --csv_file datasets/DREAM/boltz_samples.csv \
  --template_yaml config/wdr91_8hsj_pocket_template.yaml
```