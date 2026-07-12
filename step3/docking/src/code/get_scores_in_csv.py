
import pandas as pd
from utils.utils import load_config
from pathlib import Path
from utils import path_setups
import sys
import re
import multiprocessing as mp
import os



def get_scores(docked_sdf_path):
    scores = {'MOL_ID': None, 'BEST_SCORE': None, 'VINA_SCORE': None, 'SHAPE_SCORE': None, 'ESP_SCORE': None}

    mol_id = (docked_sdf_path.stem).split('_')[0]
    scores['MOL_ID'] = mol_id
    print(f"MOL_ID: {mol_id}", flush=True)

    with open(docked_sdf_path, 'r') as file:
        all_rows = file.readlines()

        for i, row in enumerate(all_rows):
            if "<meeko>" in row:
                vina_score = all_rows[i+1].strip().split(',')[1]
                v_score = re.split('[:}]', vina_score)[1]
                scores['VINA_SCORE'] = float(v_score)
                print(f"VINA_SCORE: {v_score}", flush = True)
            if '<Best Score>' in row:
                best_score = (all_rows[i+1]).strip()
                scores['BEST_SCORE'] = float(best_score)
                print(f"BEST_SCORE: {best_score}", flush=True)
            if '<Shape Score>' in row:
                shape_score = (all_rows[i+1]).strip()
                scores['SHAPE_SCORE'] = float(shape_score)
                print(f"SHAPE_SCORE: {shape_score}", flush=True)
            if '<Esp Score>' in row:
                esp_score = (all_rows[i+1]).strip()
                scores['ESP_SCORE'] = float(esp_score)
                print(f"ESP_SCORE: {esp_score}", flush=True)
    return scores

def main():
    for dock_res_pro_dir in docking_res_protein_list:
        print(f"Reading docked ligand files: {dock_res_pro_dir}")
        
        protein_name = dock_res_pro_dir.parts[-1]
        csv_save_path = f"{dock_res_pro_dir}/results_{protein_name}.csv"
        
        docked_ligands_list = [dock_ligand for dock_ligand in dock_res_pro_dir.iterdir() if dock_ligand.suffix == '.sdf']

        with mp.Pool(n_process) as pool:
            results_list = pool.map(get_scores, docked_ligands_list)

        # convert list to dict with mol_id as keys. super fast compared to list
        print("Creating a dictionary of docking scores.")
        results_dict = {result['MOL_ID']:result for result in results_list}
        print(f"Dictionary of docking scores created.")

        print("Reading input ligand csv file.")
        df = pd.read_csv(input_csv, sep=',')

        required_columns = {"SMILES", "MOL_ID", "BEST_SCORE", "SHAPE_SCORE", "ESP_SCORE", "BEST_SCORE"}
        if not required_columns.issubset(df.columns):
            print(f"[ERROR] Please check CSV file. It lacks following required column(s): {required_columns - set(df.columns)}")
            sys.exit(1)

        print("Writing docking scores of ligands in csv file.")

        for index, row in df.iterrows():
            if pd.isna(row['BEST_SCORE']) and pd.isna(row['VINA_SCORE']) and pd.isna(row['SHAPE_SCORE']) and  pd.isna(row['ESP_SCORE']):
                mol_name = row['MOL_ID']

                if mol_name in results_dict:
                    df.at[index, 'VINA_SCORE'] = results_dict[mol_name]['VINA_SCORE']
                    df.at[index, 'BEST_SCORE'] = results_dict.get(mol_name)['BEST_SCORE']
                    df.at[index, 'SHAPE_SCORE'] = results_dict.get(mol_name)['SHAPE_SCORE']
                    df.at[index, 'ESP_SCORE'] = results_dict[mol_name]['ESP_SCORE']
            
        df.to_csv(csv_save_path, sep=',', header=True, index=False)
        print(f"Docking score of the ligands saved: {csv_save_path}")

if __name__ == "__main__":
    n_process = os.cpu_count()
    input_csv = load_config("Paths", "ligand_file_csv")
    job_path, *_ = path_setups.project_path_setups()
    docking_results = job_path / "docked_ligand_poses"
    
    docking_res_protein_list= [protein_dir for protein_dir in docking_results.iterdir() if protein_dir.is_dir()]
    print(docking_res_protein_list)
    main()