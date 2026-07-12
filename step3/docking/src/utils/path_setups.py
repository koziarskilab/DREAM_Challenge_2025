from pathlib import Path

from .utils import load_config

job_name = load_config("General_Setup", "job_name")
results_base_dir = Path(load_config("Paths", "results_base_dir"))


def project_path_setups():

    job_path = results_base_dir / job_name

    unprep_recep_dir = Path(load_config("Paths", "unprepared_receptors_dir"))
    unprep_ligands_dir = Path(load_config("Paths", "unprepared_ligands_dir"))
    crystal_ligands_dir = Path(load_config("Paths", "crystal_ligands_dir"))
    crystal_ligands_dir_cv = Path(load_config("Paths", "crystal_ligands_dir_cv"))

    pdbqt_recp_dir = job_path / "prepared_receptors_pdbqt"
    pdbqt_lig_dir = job_path / "prepared_ligands_pdbqt"
    docked_lig_poses = job_path / "docked_ligand_poses"
    

    for path in [job_path, pdbqt_recp_dir, pdbqt_lig_dir, docked_lig_poses]:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory for {job_name}")

    return (
        job_path,
        unprep_recep_dir,
        unprep_ligands_dir,
        pdbqt_recp_dir,
        pdbqt_lig_dir,
        docked_lig_poses,
        crystal_ligands_dir,
        crystal_ligands_dir_cv

    )
