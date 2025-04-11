import requests
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage
from IPython.display import Image, display
import gcsfs
import datetime
import enum
from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem

from dataclasses import dataclass

import numpy as np
import pandas as pd

FINGERPRINT_TYPES = ['ATOMPAIR', 'MACCS', 'ECFP6',
                     'ECFP4', 'FCFP4', 'FCFP6', 'TOPTOR', 'RDK', 'AVALON']


def display_google_drive_image(file_id):
    """
    Display an image from Google Drive in a local Jupyter Notebook

    Parameters:
    -----------
    file_id : str
        The unique file ID from the Google Drive share link
    """
    # Direct download URL for Google Drive
    direct_url = f'https://drive.google.com/uc?id={file_id}'

    try:
        # Fetch the image
        response = requests.get(direct_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Open and display image using PIL and matplotlib
        img = PILImage.open(io.BytesIO(response.content))
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    

    except Exception as e:
        print(f"Error displaying image: {e}")
        print("Troubleshooting tips:")
        print("1. Ensure the image is publicly accessible")
        print("2. Check the file ID is correct")
        print("3. Verify your internet connection")


"""Module for handling datasets."""


@dataclass
class Dataset:
    """Class for loading dataset from Parquet or CSV file."""

    def __init__(self, filename: str):
        self.filename = filename
        self.df = self._load_file()

    def _load_file(self):
        """Loads the dataset based on the file extension."""
        if self.filename.endswith(".parquet"):
            return pd.read_parquet(self.filename)
        elif self.filename.endswith(".csv"):
            return pd.read_csv(self.filename)
        else:
            raise ValueError("Unsupported file format. Use .parquet or .csv")

    def get_dataframe(self):
        """Returns the loaded DataFrame."""
        return self.df


class ProcessData:
    """Class to process a DataFrame column into a NumPy array."""

    def __init__(self, data_frame: pd.DataFrame, column_name: str):
        if column_name not in data_frame.columns:
            raise ValueError(
                f"Column '{column_name}' not found in the DataFrame.")

        self.data_frame = data_frame
        self.column_name = column_name
        self.processed_data = self._process_column()

    def _process_column(self) -> np.ndarray:
        """Converts a column of comma-separated strings into a NumPy array."""
        return np.stack(
            self.data_frame[self.column_name].apply(
                lambda x: np.fromstring(str(x), sep=',', dtype=np.float32))
        )

    def get_data(self) -> np.ndarray:
        """Returns the processed NumPy array."""
        return self.processed_data


class SimplifiedDrugFilters:
    @staticmethod
    def fetch_attributes(molecule):
        return {
            "molecular_weight": Descriptors.ExactMolWt(molecule),
            "logp": Descriptors.MolLogP(molecule),
            "h_bond_donor": Descriptors.NumHDonors(molecule),
            "h_bond_acceptors": Descriptors.NumHAcceptors(molecule),
            "rotatable_bonds": Descriptors.NumRotatableBonds(molecule),
            "num_atoms": Chem.rdchem.Mol.GetNumAtoms(molecule),
            "molar_refractivity": Chem.Crippen.MolMR(molecule),
            "topo_surface_area": Chem.QED.properties(molecule).PSA
        }

    def filter(self, smiles):
        results = {"lipinski": [], "ghose": [],
                   "veber": [], "pass_all_filters": []}
        molecules = [Chem.MolFromSmiles(i) for i in smiles]

        for i, mol in enumerate(molecules):
            props = self.fetch_attributes(mol)

            # Lipinski Rule of 5
            lipinski = (props["molecular_weight"] <= 500 and props["logp"] <= 5 and
                        props["h_bond_donor"] <= 5 and props["h_bond_acceptors"] <= 10 and
                        props["rotatable_bonds"] <= 5)

            # Ghose Filter
            ghose = (160 <= props["molecular_weight"] <= 480 and -0.4 <= props["logp"] <= 5.6 and
                     20 <= props["num_atoms"] <= 70 and 40 <= props["molar_refractivity"] <= 130)

            # Veber Rule
            veber = (props["rotatable_bonds"] <=
                     10 and props["topo_surface_area"] <= 140)

            results["lipinski"].append(lipinski)
            results["ghose"].append(ghose)
            results["veber"].append(veber)
            results["pass_all_filters"].append(all([lipinski, ghose, veber]))

        return results

    def assignPointsToClusters(self, picks, fps):
        clusters = defaultdict(list)
        for i, idx in enumerate(picks):
            clusters[i].append(idx)
        sims = np.zeros((len(picks), len(fps)))
        for i in tqdm(range(len(picks))):
            pick = picks[i]
            sims[i, :] = DataStructs.BulkTanimotoSimilarity(fps[pick], fps)
            sims[i, i] = 0  # Don't compare the molecule with itself
        best = np.argmax(sims, axis=0)
        for i, idx in enumerate(best):
            if i not in picks:
                clusters[idx].append(i)
        return clusters


class ResultSubmission:
    @staticmethod
    def submit_result(team_name, df_predictions_test):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{team_name}_{timestamp}.csv"
        try:

            gcs_path = f"gs://aircheck-workshop-writeonly/results/{file_name}"
            fs = gcsfs.GCSFileSystem(anonymous=True)
            with fs.open(gcs_path, 'w') as f:
                df_predictions_test.to_csv(f, index=False)
            print(f"Successfully wrote best_nominees for {team_name}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    data_frame = Dataset("./TrainDataset_Aircheck_class0_1x.parquet")
    df_train = data_frame.get_dataframe()
    print(df_train.head())
    N = 1

    # Select all rows where DELLabel == 1 (positive samples)
    positive_samples = df_train[df_train["DELLabel"] == 1]
    print("length of positive sample", len(positive_samples))

    # Select N times more rows where DELLabel == 0 (negative samples)
    negative_samples = df_train[df_train["DELLabel"] == 0].sample(
        n=len(positive_samples) * N, random_state=42)

    # Combine both subsets to create a balanced dataset with the desired ratio
    df_balanced = pd.concat([positive_samples, negative_samples])

    df_train = df_balanced
    print(len(df_balanced))
    # df_test =
