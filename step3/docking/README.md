# STEP 3: Docking Part

# WDR91 Docking with AutoDock Vina (ADVina)
This docking code is customized for WDR91 docking using the protein structure **8SHJ** and a box of **20A around the crystal ligand** to identify potential WDR91 hits. Importantly, instead of using the default ADVina docking score, a **custom scorer** is implemented to select the final docked poses.

## Custom Scorer
The customized scorer (**BEST_SCORE**) is based on the electrostatic and shape similarity of docked ligands with the crystal ligands. For this, the **ESP_SIM** package (https://github.com/hesther/espsim) is integrated into the docking pipeline.

First, docking generates **20 docked poses per ligand**, followed by the comparison of these docking poses with experimental binding poses of **6 crystal ligands** stored in

```data/crystal_ligands```

Each docked pose of molecule is compared against all crystal ligands using shape and electrostatic similarity.

```20 docked poses/ligand * 6 crystal ligands = 120 comparisons```

The custom scorer allows different contributions of shape and electrostatic terms. In the current implementation, we employed equal contributions.

```BEST_SCORE = (0.5 * ESP_SCORE) +  (0.5 * SHAPE_SCORE)```

The final poses are saved based on the highest **BEST_SCORE** of docked molecules.


## Installation
Create and activate the conda environment:
First enter `step3/docking` directory

```bash
conda env create -f environment.yaml
conda activate wdr91_docking
```


## Instructions

### Configuration File
- All docking parameters are defined in 

```config.yaml``` 

The parameters are currently set for WDR91 docking protocol. A brief description of each parameter is described in this file.

The configuration file also contains settings for **SLURM** job scheduler. Currently,

```slurm_use: False```  

To activate SLURM-based parallel processing across multiple cluster nodes, change it to:

```slurm_use: True```


### Protein Preparation
For docking, an appropriately prepared protein structure is required, which includes:
- Correct protonation states
- Solvent treatment
- Addition of hydrogen atoms
- No missing side chains in the binding site

The protonated states were corrected externally using PROPKA, and that protein (with no hydrogens) is stored in:

```data/receptors``` 

Additional protein preparation was done using **PDBFixer**, and the docking-ready file was stored as a **.pdbqt** file required for ADVina docking using:

```bash
python src/code/prep_receptor.py
```

Based on the `job name` in `configuration` file, a docking-ready protein will be saved under:

``` vina_results/WDR91_crystal_ligands_docking/prepared_receptors_pdbqt```

This receptor file is ready for WDR91 docking. 

### Ligand Preparation
All molecules to be docked are provided in **.csv file** 
 
```data/wdr91_crystal_ligands.csv``` 

The file **MUST** contain required column names (see the .csv file please).

First SMILES are corrected at desired pH and converted into SDF format using 
```bash
python src/code/csv_to_sdf.py
```

The generated SDF ligands are saved in:
```data/ligands```

Then, for each molecule, 100 conformations are generated, and the lowest energy conformation is saved for docking in `.pdbqt` format required for ADVina docking using:

```bash
python src/code/prep_ligand.py
```

The docking-ready `.pdbqt` format molecules are saved under:
```vina_results/WDR91_crystal_ligands_docking/prepared_ligands_pdbqt```


### Docking
Docking process can be started using :
```bash
python src/code/docking.py
```

This will generate and save the docking poses in `.sdf` format under:

```vina_results/WDR91_crystal_ligands_docking/docked_ligand_poses/{receptor_file_name}```


### Results Collection
The docking scores of the molecules can be collected using:
```bash
python src/code/get_scores_in_csv.py
```

This will save the results as:
```vina_results/WDR91_crystal_ligands_docking/docked_ligand_poses/{receptor_file_name}/results_{receptor_file_name}.csv```

This CSV file should have fetched scores from docked molecules, i.e., VINA_SCORE, SHAPE_SCORE, ESP_SCORE and BEST_SCORE.

- VINA_SCORE: default ADVina score
- SHAPE_SCORE: shape similarity score between the docked molecule and the best matching crystal ligand.
- ESP_SCORE: electrostatic similarity score between the docked molecule and the best matching crystal ligand.
- BEST_SCORE: A weighted score between the docked molecule and the best matching crystal ligands.

High BEST_SCORE is the best; Have fun!

### Try This Simple Workflow for WDR91 Docking
First clear previous results:

```bash
rm data/ligands/* 
rm -r vina_results/WDR91_crystal_ligands_docking
```

Then, run following commands:

```
          SMILES (.csv)
                |
                v
    python src/code/csv_to_sdf.py
                |
                v
  python src/code/prep_ligand.py
                |
                v
 python src/code/prep_receptor.py
                |
                v
    python src/code/docking.py
                |
                v
python src/code/get_scores_in_csv.py
```

Finally, if you face any challenges running this code, please shoot an email to msbahia17@gmail.com





          
