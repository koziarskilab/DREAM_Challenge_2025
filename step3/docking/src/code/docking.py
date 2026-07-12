import multiprocessing as mp
import os
import time
from more_itertools import chunked


from utils import docking_helper, path_setups, utils

is_slurm_used = utils.load_config("General_Setup", "slurm_use")
idx = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
N_PROCESS = utils.load_config("General_Setup", "num_jobs_per_node")
BATCH_SIZE = utils.load_config("General_Setup", "batch_size")
DOCKING_TYPE = utils.load_config("Vina_Docking", "docking_type")
*_, VINA_DOCKING_RESULTS_PATH,_,_ = path_setups.project_path_setups()
*_, PDBQT_LIGANDS_PATH, _, _,_ = path_setups.project_path_setups()
*_, PDBQT_RECEPTORS_PATH, _, _, _,_ = path_setups.project_path_setups()


def sel_req_receptors(r_name):
    if DOCKING_TYPE == "regular":

        rigid_recep = PDBQT_RECEPTORS_PATH / f"{r_name}_regular.pdbqt"
        flex_recep = None
        return rigid_recep, flex_recep
    else:

        rigid_recep = PDBQT_RECEPTORS_PATH / f"{r_name}_rigid.pdbqt"
        flex_recep = PDBQT_RECEPTORS_PATH / f"{r_name}_flex.pdbqt"
        return rigid_recep, flex_recep


def dock_molecules(r_name, ligands):
    rigid_recep, flex_recep = sel_req_receptors(r_name)

    with mp.Pool(N_PROCESS) as pool:

        pool.starmap(
            docking_helper.vina_docking,
            [
                (
                    rigid_recep,
                    flex_recep,
                    (PDBQT_LIGANDS_PATH / lig),
                )
                for lig in ligands
            ],
        )


def run_vina_docking():
    utils.save_config_file()

    utils.verify_input_docking_type()

    try:
        receptors = utils.read_receptors_dir(PDBQT_RECEPTORS_PATH, "pdbqt")
        all_ligands = utils.read_ligands_dir(PDBQT_LIGANDS_PATH, "pdbqt")

        if is_slurm_used:
            all_batches = list(chunked(all_ligands, BATCH_SIZE))
            ligands = all_batches[idx]
        else:
            ligands = all_ligands

        for rec_name in sorted(set(receptors)):
            if not (VINA_DOCKING_RESULTS_PATH / rec_name).exists():
                (VINA_DOCKING_RESULTS_PATH / rec_name).mkdir(parents=True, exist_ok=True)

            try:
                dock_molecules(rec_name, ligands)

            except Exception as e:
                print(f"[SKIP] Docking failed with receptor {e}. ")
                print(e)
                continue

    except Exception as e:
        print(f"Vina docking failed.")
        print(e)


if __name__ == "__main__":
    start_time = time.time()
    run_vina_docking()
    end_time = time.time()
    utils.print_time_elapsed(start_time, end_time)
