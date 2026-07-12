import re
import yaml
from rdkit import Chem

import pandas as pd
import multiprocessing as mp
from pathlib import Path
import shutil
from openbabel import openbabel as ob
import subprocess
from rdkit.Chem import AllChem


def load_config(section, key):
    with open("config.yaml", "r") as config_file:
        configs = yaml.safe_load(config_file)
    return configs[section][key]

SEED = load_config("Ligand_Embeddings", "random_seed")
ph_val = load_config("Ligand_Embeddings", "ph_value")
CSV_FILE = load_config("Paths", "ligand_file_csv")
UNPREP_LIGANDS_DIR = load_config("Paths", "unprepared_ligands_dir")
N_PROCESS = load_config("General_Setup", "num_jobs_per_node")
DOCKING_TYPE = load_config("Vina_Docking", "docking_type")
VALID_DOCK_TYPES = {"flexible", "regular"}

LIGAND_TYPE = load_config("General_Setup", "ligand_file_type")
VALID_LIGAND_TYPE = {"sd", "sdf", "pdbqt", "csv"}


def verify_input_docking_type():
    if DOCKING_TYPE not in VALID_DOCK_TYPES:
        raise ValueError(
            f"[ERROR] Incorrect docking type. Provided: {DOCKING_TYPE}. Allowed from {VALID_DOCK_TYPES}."
        )
    return DOCKING_TYPE


def verify_input_ligand_type():
    if LIGAND_TYPE not in VALID_LIGAND_TYPE:
        raise ValueError(
            f"[ERROR] Incorrect ligand type: {LIGAND_TYPE}. Provide one of {VALID_LIGAND_TYPE}"
        )
    return LIGAND_TYPE


def read_receptors_dir(dir, ext):
    if ext == "pdb":

        all_receptors = [
            re.split(r"[._]", r.stem)[0]
            for r in dir.iterdir()
            if str(r).endswith(".pdb")
        ]

    else:
        all_receptors = [
            re.split(r"[._]", r.stem)[0]
            for r in dir.iterdir()
            if str(r).endswith(".pdbqt")
        ]

    if not all_receptors:
        raise FileNotFoundError(
            f"[ERROR] No receptor(s) (pdb/pdbqt) found in dir: {dir}!"
        )

    return all_receptors


def read_ligands_dir(dir, ext):
    if ext == "sdf" or ext == "csv":
        all_ligands = [l.name for l in dir.iterdir() if str(l).endswith(".sdf")]

    elif ext == "sd":
        all_ligands = [l.name for l in dir.iterdir() if str(l).endswith(".sd")]
    elif ext == "pdbqt":
        all_ligands = [l.name for l in dir.iterdir() if str(l).endswith(".pdbqt")]
    else:
        all_ligands = []
        print("TODO: implement code for this type")

    if not all_ligands:
        raise FileNotFoundError(f"[ERROR] No ligand(s) (sd/sdf) found in dir: {dir}.")

    return all_ligands


def print_time_elapsed(start_time, end_time):
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"Time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")


SAVE_CONFIG_PATH = Path(load_config("Paths", "results_base_dir")).joinpath(
    load_config("General_Setup", "job_name")
)


def save_config_file():
    config_path = "./config.yaml"
    shutil.copy(config_path, SAVE_CONFIG_PATH)


def pdbqt_to_sdf(pdbqt_path, sdf_path):
    try:
        cmd = [
            f"mk_export.py",
            str(pdbqt_path),
            "-s",
            str(sdf_path),
            ]
        subprocess.run(cmd, shell=False, check=True)
    except Exception as e:
        print(f"mk_export commandline failed: {e}")