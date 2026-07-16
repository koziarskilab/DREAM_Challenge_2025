import multiprocessing as mp
import os
import time

from more_itertools import chunked

from utils import prep_ligand_helper, utils
from utils.path_setups import project_path_setups

is_slurm_used = utils.load_config("General_Setup", "slurm_use")
idx = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
_, _, UNPREP_LIGANDS_DIR, *_ = project_path_setups()
N_PROCESS = utils.load_config("General_Setup", "num_jobs_per_node")
BATCH_SIZE = utils.load_config("General_Setup", "batch_size")


def write_pdbqt_file(pdbqt_string, lig_name):

    *_, PDBQT_LIGANDS_DIR, _, _, _ = project_path_setups()

    pbbqt_f_save = PDBQT_LIGANDS_DIR / f"{lig_name}.pdbqt"

    with open(pbbqt_f_save, "w") as file:
        file.write(pdbqt_string)
    return pbbqt_f_save.exists()


def process_ligands(lig_name):

    lig_bname = os.path.basename(lig_name).split(".")[0]

    try:

        pdbqt_string, *_ = prep_ligand_helper.prepare_pdbqt_ligand(
            (UNPREP_LIGANDS_DIR / lig_name)
        )

        if not write_pdbqt_file(pdbqt_string, lig_bname):
            raise FileNotFoundError(f"[ERROR:] {lig_bname}.pdbqt file not saved!")

    except Exception as e:
        print(f"[SKIPPED] ligand {lig_name} failed to prepare: {e}")
        return


def prepare_ligands():

    try:
        LIGAND_TYPE = utils.verify_input_ligand_type()

        all_ligands = utils.read_ligands_dir(UNPREP_LIGANDS_DIR, LIGAND_TYPE)

        if is_slurm_used:
            all_batches = list(chunked(all_ligands, BATCH_SIZE))
            all_ligands = all_batches[idx]
            print(
                f"Total ligand batches: {len(all_batches)} | batch size: {len(all_ligands)}"
            )

        with mp.Pool(N_PROCESS) as pool:
            pool.map(process_ligands, all_ligands)

    except Exception as e:
        print(f"[ERROR:] Molecule preperation failed!")
        print(e)


if __name__ == "__main__":
    start_time = time.time()
    prepare_ligands()
    end_time = time.time()
    utils.print_time_elapsed(start_time, end_time)
