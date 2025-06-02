import requests
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage
from IPython.display import Image, display

# import gcsfs
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

import pyarrow.parquet as pq


FINGERPRINT_TYPES = [
    "ATOMPAIR",
    "MACCS",
    "ECFP6",
    "ECFP4",
    "FCFP4",
    "FCFP6",
    "TOPTOR",
    "RDK",
    "AVALON",
]

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
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

        self.data_frame = data_frame
        self.column_name = column_name
        self.processed_data = self._process_column()

    def _process_column(self) -> np.ndarray:
        """Converts a column of lists, arrays, or string representations into a NumPy array."""
        column_data = self.data_frame[self.column_name]
        
        if isinstance(column_data.iloc[0], (list, np.ndarray)):
            # If the column contains lists or arrays, stack them directly
            return np.stack(column_data)
        elif isinstance(column_data.iloc[0], str):
            # If the column contains string representations of arrays, parse them
            if column_data.iloc[0].startswith("["):
                # If the string starts with '[', it's likely a list representation
                import pdb
                breakpoint()
                return np.stack(
                    column_data.apply(
                        lambda x: np.fromstring(x.strip("[]").replace("...", ""), sep=" ", dtype=np.float32)
                    )
                )
            else:
                # If the string does not start with '[', it's likely a comma-separated string
                return np.stack(
                    column_data.apply(lambda x: np.fromstring(str(x), sep=",", dtype=np.float32))
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
            "topo_surface_area": Chem.QED.properties(molecule).PSA,
        }

    def filter(self, smiles):
        results = {"lipinski": [], "ghose": [], "veber": [], "pass_all_filters": []}
        molecules = [Chem.MolFromSmiles(i) for i in smiles]

        for i, mol in enumerate(molecules):
            props = self.fetch_attributes(mol)

            # Lipinski Rule of 5
            lipinski = (
                props["molecular_weight"] <= 500
                and props["logp"] <= 5
                and props["h_bond_donor"] <= 5
                and props["h_bond_acceptors"] <= 10
                and props["rotatable_bonds"] <= 5
            )

            # Ghose Filter
            ghose = (
                160 <= props["molecular_weight"] <= 480
                and -0.4 <= props["logp"] <= 5.6
                and 20 <= props["num_atoms"] <= 70
                and 40 <= props["molar_refractivity"] <= 130
            )

            # Veber Rule
            veber = props["rotatable_bonds"] <= 10 and props["topo_surface_area"] <= 140

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

def read_parquet_file(file_path, columns=None, nrows=None):
    df = pd.read_parquet(file_path, columns=columns, engine='pyarrow')
    if nrows is not None:
        df = df.head(nrows)
    return df

def process_column_to_array(df, column_name):
    if isinstance(df[column_name].iloc[0], str):
        # Column is string, needs conversion
        return np.stack(df[column_name].apply(lambda x: np.fromstring(x, sep=',', dtype=np.float32)))
    else:
        # Column is already array-like
        return np.stack(df[column_name])

