import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
from typing import List, Tuple, Dict, Union

def smiles_to_dgl_graph(smiles: str) -> dgl.DGLGraph:
    """Convert SMILES to DGL graph (same as before)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    
    node_features = []
    for atom in atoms:
        features = get_atom_features(atom)
        node_features.append(features)
    
    edges_src = []
    edges_dst = []
    edge_features = []
    
    for bond in bonds:
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        
        edges_src.extend([src, dst])
        edges_dst.extend([dst, src])
        
        bond_feat = get_bond_features(bond)
        edge_features.extend([bond_feat, bond_feat])
    
    if len(edges_src) == 0:
        g = dgl.graph(([], []), num_nodes=len(atoms))
    else:
        g = dgl.graph((edges_src, edges_dst), num_nodes=len(atoms))
    
    g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
    if len(edge_features) > 0:
        g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float32)
    
    return g

def get_atom_features(atom) -> List[float]:
    """Get atom features (same as before)"""
    features = []
    
    atomic_num = atom.GetAtomicNum()
    common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    atom_encoding = [1 if atomic_num == x else 0 for x in common_atoms]
    atom_encoding.append(1 if atomic_num not in common_atoms else 0)
    features.extend(atom_encoding)
    
    degree = atom.GetDegree()
    degree_encoding = [1 if degree == x else 0 for x in range(6)]
    features.extend(degree_encoding)
    
    features.append(atom.GetFormalCharge())
    
    hybridization = atom.GetHybridization()
    hybrid_encoding = [1 if hybridization == x else 0 for x in [
        Chem.HybridizationType.SP, Chem.HybridizationType.SP2, 
        Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, 
        Chem.HybridizationType.SP3D2
    ]]
    features.extend(hybrid_encoding)
    
    features.append(1 if atom.GetIsAromatic() else 0)
    
    return features

def get_bond_features(bond) -> List[float]:
    """Get bond features (same as before)"""
    features = []
    
    bond_type = bond.GetBondType()
    bond_encoding = [1 if bond_type == x else 0 for x in [
        Chem.BondType.SINGLE, Chem.BondType.DOUBLE, 
        Chem.BondType.TRIPLE, Chem.BondType.AROMATIC
    ]]
    features.extend(bond_encoding)
    
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    
    return features

def parse_compound_smiles(smiles: str) -> List[str]:
    """Parse compound SMILES separated by semicolons"""
    if ';' in smiles:
        # Split by semicolon and clean up
        smiles_list = [s.strip() for s in smiles.split(';') if s.strip()]
        # Validate each SMILES
        valid_smiles = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                valid_smiles.append(s)
        return valid_smiles
    else:
        # Single SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return [smiles]
        else:
            return []

class SMILESDataset(Dataset):
    def __init__(self, csv_file: str, smiles_col: str = 'SMILES', label_col: str = 'LABEL', 
                 model_type: str = 'gcn', max_length: int = 100, 
                 fp_type: str = 'MACCS', fp_size: int = 2048, radius: int = 2,
                 is_validation: bool = False):
        """
        Dataset for SMILES and labels
        Args:
            csv_file: Path to CSV file
            smiles_col: Column name for SMILES
            label_col: Column name for labels
            model_type: 'gcn', 'transformer', or 'mlp'
            max_length: Maximum sequence length for transformer
            fp_type: Fingerprint type for MLP (not used, kept for compatibility)
            fp_size: Fingerprint size (not used, kept for compatibility)
            radius: Radius for Morgan fingerprint (not used, kept for compatibility)
            is_validation: Whether this is a validation dataset (affects compound SMILES handling)
        """
        # Handle different file formats
        if csv_file.endswith('.parquet'):
            self.df = pd.read_parquet(csv_file)
        else:
            self.df = pd.read_csv(csv_file)
            
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.model_type = model_type
        self.max_length = max_length
        self.fp_type = fp_type
        self.fp_size = fp_size
        self.radius = radius
        self.is_validation = is_validation
        
        # Store original data for compound SMILES handling
        self.original_data = []
        
        # Process compound SMILES and create expanded dataset
        expanded_data = []
        for idx, row in self.df.iterrows():
            smiles = row[smiles_col]
            label = row[label_col]
            
            # Parse compound SMILES
            parsed_smiles = parse_compound_smiles(smiles)
            
            if len(parsed_smiles) == 0:
                continue  # Skip invalid SMILES
            
            if self.is_validation and len(parsed_smiles) > 1:
                # For validation with compound SMILES, store all variants
                # We'll handle prediction aggregation in the collate function
                for i, single_smiles in enumerate(parsed_smiles):
                    new_row = row.copy()
                    new_row[smiles_col] = single_smiles
                    new_row['original_idx'] = idx
                    new_row['compound_variant'] = i
                    new_row['total_variants'] = len(parsed_smiles)
                    expanded_data.append(new_row)
                    
                # Store original data for cluster metrics
                self.original_data.append({
                    'idx': idx,
                    'smiles': smiles,
                    'label': label,
                    'row': row,
                    'parsed_smiles': parsed_smiles
                })
            else:
                # For training or single SMILES, just use the first valid one
                new_row = row.copy()
                new_row[smiles_col] = parsed_smiles[0]
                new_row['original_idx'] = idx
                new_row['compound_variant'] = 0
                new_row['total_variants'] = 1
                expanded_data.append(new_row)

        print(f"Loaded {len(self.df) if self.is_validation else len(self.df)} original samples")
        
        self.df = pd.DataFrame(expanded_data).reset_index(drop=True)
        
        print(f"Expanded to {len(self.df)} processed samples")
        
        # For transformer, build vocabulary
        if model_type == 'transformer':
            self.vocab = self._build_vocab()
    
    def _build_vocab(self):
        """Build vocabulary from SMILES strings"""
        vocab = set()
        for smiles in self.df[self.smiles_col]:
            vocab.update(list(smiles))
        
        # Add special tokens
        vocab_list = ['<PAD>', '<UNK>', '<CLS>', '<SEP>'] + sorted(list(vocab))
        vocab_dict = {token: idx for idx, token in enumerate(vocab_list)}
        
        print(f"Built vocabulary with {len(vocab_dict)} tokens")
        return vocab_dict
    
    def _smiles_to_tokens(self, smiles: str):
        """Convert SMILES to token IDs"""
        # Add CLS token at the beginning
        tokens = ['<CLS>'] + list(smiles)
        
        # Convert to IDs
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([self.vocab['<PAD>']] * (self.max_length - len(token_ids)))
        
        # Create attention mask
        attention_mask = [1 if token_id != self.vocab['<PAD>'] else 0 for token_id in token_ids]
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row[self.smiles_col]
        label = row[self.label_col]
        
        # Include metadata for compound SMILES handling
        metadata = {
            'original_idx': row['original_idx'],
            'compound_variant': row['compound_variant'],
            'total_variants': row['total_variants']
        }
        
        if self.model_type == 'gcn':
            return smiles, int(label), metadata
        elif self.model_type == 'transformer':
            token_ids, attention_mask = self._smiles_to_tokens(smiles)
            return token_ids, attention_mask, int(label), metadata
        else:  # mlp - this should not be used since MLP uses baseline approach
            # Return dummy data - MLP should use baseline Dataset/ProcessData approach
            raise NotImplementedError("MLP model should use baseline Dataset/ProcessData approach")

def collate_fn_gcn(batch):
    """Collate function for GCN with compound SMILES handling"""
    if len(batch[0]) == 3:  # New format with metadata
        smiles_list, labels, metadata_list = zip(*batch)
    else:  # Old format for backward compatibility
        smiles_list, labels = zip(*batch)
        metadata_list = [{'original_idx': i, 'compound_variant': 0, 'total_variants': 1} 
                        for i in range(len(batch))]
    
    graphs = []
    valid_labels = []
    valid_metadata = []
    
    for smiles, label, metadata in zip(smiles_list, labels, metadata_list):
        try:
            graph = smiles_to_dgl_graph(smiles)
            graphs.append(graph)
            valid_labels.append(label)
            valid_metadata.append(metadata)
        except ValueError:
            continue
    
    if len(graphs) == 0:
        return None, None, None
    
    batched_graph = dgl.batch(graphs)
    labels_tensor = torch.tensor(valid_labels, dtype=torch.long)
    
    return batched_graph, labels_tensor, valid_metadata

def collate_fn_transformer(batch):
    """Collate function for Transformer with compound SMILES handling"""
    if len(batch[0]) == 4:  # New format with metadata
        token_ids_list, attention_mask_list, labels, metadata_list = zip(*batch)
    else:  # Old format for backward compatibility
        token_ids_list, attention_mask_list, labels = zip(*batch)
        metadata_list = [{'original_idx': i, 'compound_variant': 0, 'total_variants': 1} 
                        for i in range(len(batch))]
    
    token_ids = torch.stack(token_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return token_ids, attention_mask, labels_tensor, metadata_list

def collate_fn_mlp(batch):
    """Collate function for MLP - not used since MLP uses baseline approach"""
    fingerprints, labels = zip(*batch)
    
    fingerprints_tensor = torch.stack(fingerprints)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return fingerprints_tensor, labels_tensor