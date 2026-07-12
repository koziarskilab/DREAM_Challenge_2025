
import os
import re
import subprocess
from vina import Vina
import numpy as np

from .path_setups import project_path_setups
from .utils import load_config
import glob
import sys
import tempfile
from pathlib import Path
from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit.Chem.rdShapeHelpers import ShapeTanimotoDist
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from espsim import GetEspSim, GetShapeSim
import uuid

# DOCKER = load_config("Vina_Docking", "docker")
SEED = load_config("Vina_Docking", "random_seed")
BOX_CENTER = load_config("Vina_Docking", "center")
BOX_SIZE = load_config("Vina_Docking", "box_size")
EXHAUSTIVENESS = load_config("Vina_Docking", "exhaustiveness")
DOCKED_POSES_TYPE = load_config("Vina_Docking", "docked_poses_file_type")
SAVE_POSES = load_config("Vina_Docking", "generate_poses")
*_, VINA_DOCKING_RESULTS_PATH, _, _ = project_path_setups()
VINA_CPU = load_config("Vina_Docking", "cpu_per_ligand_exhaustiveness")
POSE_FILTERING = load_config("General_Setup", "pose_filtering")
SHAPE_WT = load_config("General_Setup", "shape_weight")
ESP_WT = load_config("General_Setup", "esp_weight")
*_, CRYSTAL_LIGANDS_DIR, _ = project_path_setups()

def vina_docking(rigid_receptor, flex_receptor, lig_path):

    try:
        lig_bname = Path(lig_path).stem
        recep_bname = re.split(r"[._]", os.path.basename(rigid_receptor))[0]
        best_pdbqt_path = VINA_DOCKING_RESULTS_PATH / f"{recep_bname}/{lig_bname}_out.pdbqt"
        best_sdf_path = VINA_DOCKING_RESULTS_PATH / f"{recep_bname}/{lig_bname}_out.sdf"

        # if DOCKER == 'qvina':

        #     cmd = build_qvina_cmd(rigid_receptor, lig_path, flex_receptor)

        #     if POSE_FILTERING:
        #         with tempfile.TemporaryDirectory() as temp_dir:
        #             temp_path = Path(temp_dir)
            
        #             pdbqt_path = temp_path / f"{lig_bname}.pdbqt"
        #             sdf_path = temp_path / f"{lig_bname}.sdf"

        #             args = ["--num_modes", str(SAVE_POSES), "--out", str(pdbqt_path)]
        #             cmd.extend(args)
        #             run_cmd(cmd, lig_bname)
        #             pdbqt_to_sdf(pdbqt_path, sdf_path)

        #             all_cls = [f for f in CRYSTAL_LIGANDS_DIR.iterdir() if f.suffix == ".sdf"]

        #             get_best_mol(sdf_path, all_cls, best_pdbqt_path, best_sdf_path)

        #     else:
        #         args = ["--num_modes", str(SAVE_POSES), "--out", str(best_pdbqt_path)]
        #         cmd.extend(args)
        #         run_cmd(cmd, lig_bname)

        #         if DOCKED_POSES_TYPE in ('sdf', 'both'):
        #             pdbqt_to_sdf(best_pdbqt_path, best_sdf_path)

        #     if (best_sdf_path).exists() or (best_pdbqt_path).exists():
        #         print(
        #             f"Docked ligand {lig_bname} with protein {recep_bname} successfully!"
        #         )
                
        # elif DOCKER == 'advina':
        vina = Vina(sf_name="vina", seed=SEED, cpu=VINA_CPU, verbosity=1)

        if flex_receptor:
            vina.set_receptor(
                    rigid_pdbqt_filename=str(rigid_receptor),
                    flex_pdbqt_filename=str(flex_receptor),
                )
        else:
            vina.set_receptor(
                    rigid_pdbqt_filename=str(rigid_receptor),
                )

        vina.set_ligand_from_file(str(lig_path))
        vina.compute_vina_maps(center=BOX_CENTER, box_size=BOX_SIZE)
        vina.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=20)

        if POSE_FILTERING:
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir)
                unique_name = uuid.uuid4().hex[:8]
                pdbqt_path = path / f"{lig_bname}_{unique_name}.pdbqt"
                sdf_path = path / f"{lig_bname}_{unique_name}.sdf"

                vina.write_poses(
                        str(pdbqt_path),
                        n_poses=SAVE_POSES,
                        energy_range=3,
                        overwrite=True,
                    )
                pdbqt_to_sdf(pdbqt_path, sdf_path)

                all_cls = [f for f in CRYSTAL_LIGANDS_DIR.iterdir() if f.suffix == ".sdf"]

                get_best_mol(sdf_path, all_cls, best_pdbqt_path, best_sdf_path)

                if (best_sdf_path).exists() or best_pdbqt_path.exists():
                        print(
                            f"Docked ligand {lig_bname} with protein {recep_bname} successfully!"
                        )

        else:
            vina.write_poses(
                    str(best_pdbqt_path),
                    n_poses=SAVE_POSES,
                    energy_range=3,
                    overwrite=True,
                )

            if DOCKED_POSES_TYPE == 'sdf':
                pdbqt_to_sdf(best_pdbqt_path, best_sdf_path)

        if (best_sdf_path).exists():
            print(
                    f"Docked ligand {lig_bname} with protein {recep_bname} successfully!"
                    )
                
    except Exception as e:
        print(
            f"Docking failed for ligand {lig_path} with receptor {rigid_receptor}: {e}"
        )



def get_best_mol(docked_pool_path, cligs, best_pdbqt_path, best_sdf_path):
    best_score = float("-inf")
    best_shape_score = float("-inf")
    best_esp_score = float("-inf")
    best_pose = None
    best_esp_mol = None

    all_cl_mols = []
    for cl in cligs:
        cl_supplier = Chem.SDMolSupplier(str(cl), removeHs=False)
        cl_mol = cl_supplier[0]
        all_cl_mols.append(cl_mol)

    dock_supplier = Chem.SDMolSupplier(str(docked_pool_path), removeHs=False)

    for mol1 in dock_supplier:  
        if mol1 is None:
            continue
            
        for index, cl_mol1 in enumerate(all_cl_mols):
            mol = mol1
            cl_mol = cl_mol1
            
            esp_score = GetEspSim(mol, cl_mol, renormalize=True)
            tani_shape = GetShapeSim(mol, cl_mol)

            score = ESP_WT * esp_score + SHAPE_WT * tani_shape
          
            if score > best_score:
                best_score = score
                best_pose = mol
                best_esp_score = esp_score
                best_shape_score = tani_shape
                best_esp_mol = index

    # print("ESP mol for sim: ", cligs[best_esp_mol].stem)
    best_pose.SetProp("Best Score", str(best_score))
    print(f"Best score:  {best_score}")
    best_pose.SetProp("Shape Score", str(best_shape_score))
    best_pose.SetProp("Esp Score", str(best_esp_score))
    best_pose.SetProp("Reference Crystal Ligand", str(cligs[best_esp_mol].stem))

    if DOCKED_POSES_TYPE in ["sdf", 'both']:
        with Chem.SDWriter(best_sdf_path) as w:
            w.write(best_pose)


    if DOCKED_POSES_TYPE == 'both':
        mk_prep = MoleculePreparation()
        molsetup_list = mk_prep(best_pose)
        mol_setup = molsetup_list[0]
        pdbqt_string, *_ = PDBQTWriterLegacy.write_string(mol_setup)

        with open(best_pdbqt_path, "w") as f:
            f.write(pdbqt_string)

# def build_qvina_cmd(rigid_recep, lig_path, flex_res):
#     cmd = [
#             "qvina2",
#             "--receptor",
#             rigid_recep,
#             "--ligand",
#             lig_path,
#             "--center_x",
#             str(BOX_CENTER[0]),
#             "--center_y",
#             str(BOX_CENTER[1]),
#             "--center_z",
#             str(BOX_CENTER[2]),
#             "--size_x",
#             str(BOX_SIZE[0]),
#             "--size_y",
#             str(BOX_SIZE[1]),
#             "--size_z",
#             str(BOX_SIZE[2]),
#             "--seed",
#             str(SEED),
#             "--cpu",
#             str(VINA_CPU),
#             "--exhaustiveness",
#             str(EXHAUSTIVENESS),
#         ]

#     if flex_res:
#         cmd.insert(cmd.index('--ligand'), '--flex')
#         cmd.insert(cmd.index('--ligand'), str(flex_res))

#     return cmd

# def run_cmd(cmd, lig_bname):
#     try:
#         subprocess.run(cmd, shell=False, check=True)
#         print(f"Qvina docked {lig_bname} successfully.")
#     except Exception as e:
#         print(f"Qvina {lig_bname} docking failed.")
#         print(e)

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