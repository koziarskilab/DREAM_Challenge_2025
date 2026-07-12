import subprocess
import tempfile
from pathlib import Path

from openmm.app import PDBFile
from pdbfixer import PDBFixer

from .path_setups import project_path_setups
from .utils import (
    load_config,
    verify_input_docking_type,
)

DOCKING_TYPE = verify_input_docking_type()
SEED = load_config("PDB_fixer", "random_seed")
FLEX_RES = load_config("Vina_Docking", "flex_residues")
DEL_RES = load_config("Vina_Docking", "del_residues")
_, _, _, PDBQT_RECEP_DIR, *_ = project_path_setups()
PDBFixer_pdb_out = Path("./PDBFixer_pdb_out")

def fix_pdb_save_pdbqt(pdb_file):
    pdb_name = pdb_file.stem
    fixer = PDBFixer(filename=str(pdb_file))
    fixer.findMissingResidues()
    print(f"Missing residues: {fixer.missingResidues}")

    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=SEED)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    # false remove even waters; "True" keep waters
    fixer.removeHeterogens(False)
    print(f"PDBFixer job done.")


    with tempfile.TemporaryDirectory() as temp_dir:
        rec_temp_path = Path(temp_dir) / f"{pdb_name}.pdb"

        PDBFile.writeFile(fixer.topology, fixer.positions, str(rec_temp_path))
        PDBFile.writeFile(fixer.topology, fixer.positions, str(f"{PDBFixer_pdb_out}/{pdb_name}_PDBFixerOutput.pdb"))
        print(f"Corrected PDB saved at {PDBFixer_pdb_out}")

        if not rec_temp_path.exists():
            raise FileNotFoundError(
                "[ERROR:] Fixed receptor not found in temp dir for pdbqt processing."
            )

        print(
            f"[Meeko job]---x---Preparing receptor {pdb_name} for {DOCKING_TYPE} docking: ---x---"
        )

        if DOCKING_TYPE == "regular":
            cmd = [
                "mk_prepare_receptor.py",
                "-i",
                rec_temp_path,
                "--delete_residues",
                DEL_RES,
                "-p",
                PDBQT_RECEP_DIR / f"{pdb_name}_regular.pdbqt",
                "--keep_altloc A"
            ]
            subprocess.run(cmd, shell=False, check=True)

            return (PDBQT_RECEP_DIR / f"{pdb_name}_regular.pdbqt").exists()

        elif DOCKING_TYPE == "flexible":
            cmd = [
                "mk_prepare_receptor.py",
                "-i",
                rec_temp_path,
                "--delete_residues",
                DEL_RES,
                "-p",
                (PDBQT_RECEP_DIR / f"{pdb_name}.pdbqt"),
                "--keep_altloc A",
                "-f",
                FLEX_RES,
            ]

            subprocess.run(cmd, shell=False, check=True)

            return (PDBQT_RECEP_DIR / f"{pdb_name}_rigid.pdbqt").exists and (
                PDBQT_RECEP_DIR / f"{pdb_name}_flex.pdbqt"
            ).exists()
        else:
            print("Plase select docking type: Regular or Flexible.")
