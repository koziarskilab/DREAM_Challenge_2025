from utils import prep_receptor_helper, utils
from utils.path_setups import project_path_setups


def prepare_receptors():
    try:
        DOCKING_TYPE = utils.verify_input_docking_type()

        _, UNPREP_RECEP_DIR, *_ = project_path_setups()

        all_receptors = utils.read_receptors_dir(UNPREP_RECEP_DIR, "pdb")

        for rec_name in all_receptors:
            print(rec_name)
            rec_path = UNPREP_RECEP_DIR / f"{rec_name}.pdb"

            try:
                success = prep_receptor_helper.fix_pdb_save_pdbqt(rec_path)

                if success:
                    print(
                        f"[SUCCESS:] Receptor {rec_name} processed for {DOCKING_TYPE} docking!"
                    )
                else:
                    print(f"[ERROR] Receptor {rec_name} preparation failed.")

            except Exception as e:
                print(f"[SKIPPED] Receptor {rec_name}: Failed to prepare.")
                print(e)
                continue

    except Exception as e:
        print(f"Something went wrong with receptor preparation.")
        print(e)
        return


if __name__ == "__main__":
    prepare_receptors()
