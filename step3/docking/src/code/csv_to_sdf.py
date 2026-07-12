
from openbabel import openbabel as ob
import pandas as pd
import os
from utils import utils
import multiprocessing as mp
from rdkit import Chem
from pathlib import Path
from rdkit.Chem import AllChem


CSV_FILE = utils.load_config("Paths", "ligand_file_csv")
UNPREP_LIGANDS_DIR = utils.load_config("Paths", "unprepared_ligands_dir")
ph_val = utils.load_config("Ligand_Embeddings", "ph_value")
N_PROCESS = os.cpu_count()


def csv_to_sdf():
    chunks = pd.read_csv(CSV_FILE, delimiter=",", chunksize=100)

    with mp.Pool(N_PROCESS) as pool:
        pool.map(process_df, chunks)


def process_df(df):

    for _, row in df.iterrows():

        if row is not None:
            smiles = row["SMILES"]
            mol_id = row["MOL_ID"]

            try:
                obc = ob.OBConversion()
                obmol = ob.OBMol()
                obc.SetInAndOutFormats('smi', 'smi')
                obc.ReadString(obmol, smiles)
                obmol.CorrectForPH(ph_val)
                ph_corrected_smiles = obc.WriteString(obmol).strip()
            
                mol = Chem.MolFromSmiles(ph_corrected_smiles, sanitize=True)

                if mol is None:
                    print(f"{mol_id} failed: MolFromSmiles returned None")
                    continue

                molh = Chem.AddHs(mol)

                if AllChem.EmbedMolecule(molh, randomSeed=42, enforceChirality=True) != 0:
                    print(f"Embedding failed for {mol_id}")
                    continue


                writer = Chem.SDWriter(
                    Path(UNPREP_LIGANDS_DIR).joinpath(f"{mol_id}.sdf")
                    )
                    
                writer.write(molh)
                writer.close()
                # print(f"{mol_id} Smiles converted to sdf file.")

            except Exception as e:
                print(f"Error processing {mol_id}: {e}")

if __name__ == '__main__':
    csv_to_sdf()