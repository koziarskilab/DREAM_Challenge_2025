from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem

from .utils import load_config
from openbabel import openbabel as ob

NUM_CONFS = load_config("Ligand_Embeddings", "num_confs")
SEED = load_config("Ligand_Embeddings", "random_seed")
NUM_THREADS = load_config("Ligand_Embeddings", "num_threads")
MAX_ITERS = load_config("Ligand_Embeddings", "max_iters")


def get_min_energy_conf(pool_of_confs):

    energies = []
    for i in range(pool_of_confs.GetNumConformers()):

        AllChem.UFFOptimizeMolecule(pool_of_confs, confId=i, maxIters=MAX_ITERS)
        ff = AllChem.UFFGetMoleculeForceField(pool_of_confs, confId=i)
        energy = ff.CalcEnergy()
        energies.append({"id": i, "energy_values": energy})

    if not energies:
        raise ValueError("All conformers failed to optimize.")

    best_conf = min(energies, key=lambda x: x["energy_values"])
    best_id = best_conf["id"]
    min_energy_conf = pool_of_confs.GetConformer(best_id)

    copy_molh = Chem.Mol(pool_of_confs)
    copy_molh.RemoveAllConformers()
    copy_molh.AddConformer(min_energy_conf)
    return copy_molh


def prepare_pdbqt_ligand(lig_path):

    supplier = Chem.SDMolSupplier(lig_path, removeHs=False)
    for mol in supplier:
        if mol is None:
            raise ValueError("SD file contains no valid molecule.")

        mol.RemoveAllConformers()

        if mol.GetNumConformers() != 0:
            raise ValueError(f"Failed to remove orignal conformer.")

        molh = Chem.AddHs(mol)

        AllChem.EmbedMultipleConfs(
            molh,
            numConfs=NUM_CONFS,
            enforceChirality=True,
            randomSeed=SEED,
            numThreads=NUM_THREADS,
        )

        if molh.GetNumConformers() == 0:
            raise ValueError(f"3D embedding failed: No conformer generated.")

        min_energy_conf = get_min_energy_conf(molh)
        mk_prep = MoleculePreparation(rigid_macrocycles=False)
        molsetup_list = mk_prep(min_energy_conf)
        molsetup = molsetup_list[0]
        pdbqt_string = PDBQTWriterLegacy.write_string(molsetup)

        return pdbqt_string
