import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from tqdm import tqdm

from rdkit.Geometry import Point3D
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import BondType as BT

# Simple bond type encoding:
# 0: no bond, 1: single, 2: double, 3: triple, 4: aromatic, 5-8: stereo bonds
BOND_TYPES = {
    BT.SINGLE: 1,
    BT.DOUBLE: 2,
    BT.TRIPLE: 3,
    BT.AROMATIC: 4,
}
NUM_BOND_TYPES = 9  # 0-8 inclusive


def parse_xyz_corpus(filename):
    """
    Parse a long xyz file which is a sequence of xyz-block connected without seperator.
    Return a list of xyz-blocks.
    Each xyz-block contains a molecule.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    # First line is the number of atoms
    # Second line is the comment line
    # From third line, each line contains an atom and its coordinates
    # Gather lines to create a xyz-block
    xyzs = []
    i = 0
    while i < len(lines):
        n_atoms = int(lines[i])
        xyz_block = lines[i: i + n_atoms + 2]
        xyzs.append("".join(xyz_block).strip())
        i += n_atoms + 2
    return xyzs


def read_xyz_block(xyz_block):
    """Parse a xyz block and return symbols and positions."""
    sxyz = xyz_block.split("\n")[2:]
    if sxyz and not sxyz[-1]:
        sxyz = sxyz[:-1]

    symbols = []
    pos = []
    for line in sxyz:
        parts = line.strip().split()
        if len(parts) >= 4:
            symbols.append(parts[0])
            pos.append([float(parts[1]), float(parts[2]), float(parts[3])])

    symbols = np.array(symbols)
    pos = np.array(pos)
    return symbols, pos


def build_rdkit_mol(numbers, coords, bond_mat):
    """Build RDKit molecule from atomic numbers, coordinates, and adjacency matrix."""
    mol = Chem.RWMol()
    for num in numbers:
        atom = Chem.Atom(int(num))
        mol.AddAtom(atom)

    # Add bonds from adjacency matrix
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if bond_mat[i, j] == 1:
                bond_type = Chem.BondType.SINGLE
            elif bond_mat[i, j] == 2:
                bond_type = Chem.BondType.DOUBLE
            elif bond_mat[i, j] == 3:
                bond_type = Chem.BondType.TRIPLE
            else:
                bond_type = None

            if bond_type is not None:
                mol.AddBond(i, j, bond_type)

    mol = mol.GetMol()
    conf = Chem.Conformer(len(numbers))
    for i, pos in enumerate(coords):
        # Convert to Python float for Point3D
        conf.SetAtomPosition(i, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(conf, assignId=True)
    return mol


def add_stereo_bonds(mol, chi_bonds, ez_bonds, bmat, from_3D=True):
    result = []
    if from_3D:
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    else:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if bond.GetBondType() == Chem.BondType.DOUBLE and stereo in ez_bonds:
            idx_3, idx_4 = bond.GetStereoAtoms()
            atom_1, atom_2 = bond.GetBeginAtom(), bond.GetEndAtom()
            idx_1, idx_2 = atom_1.GetIdx(), atom_2.GetIdx()

            idx_5 = [nbr.GetIdx() for nbr in atom_1.GetNeighbors() if nbr.GetIdx() not in {idx_2, idx_3}]
            idx_6 = [nbr.GetIdx() for nbr in atom_2.GetNeighbors() if nbr.GetIdx() not in {idx_1, idx_4}]

            inv_stereo = Chem.BondStereo.STEREOE if stereo == Chem.BondStereo.STEREOZ else Chem.BondStereo.STEREOZ
            result.extend([(idx_3, idx_4, ez_bonds[stereo]), (idx_4, idx_3, ez_bonds[stereo])])

            if idx_5:
                result.extend([(idx_5[0], idx_4, ez_bonds[inv_stereo]), (idx_4, idx_5[0], ez_bonds[inv_stereo])])
            if idx_6:
                result.extend([(idx_3, idx_6[0], ez_bonds[inv_stereo]), (idx_6[0], idx_3, ez_bonds[inv_stereo])])
            if idx_5 and idx_6:
                result.extend([(idx_5[0], idx_6[0], ez_bonds[stereo]), (idx_6[0], idx_5[0], ez_bonds[stereo])])

        if bond.GetBeginAtom().HasProp('_CIPCode'):
            idx = bond.GetBeginAtom().GetIdx()
            chirality = bond.GetBeginAtom().GetProp('_CIPCode')
            neighbors = bond.GetBeginAtom().GetNeighbors()
            if all(n.HasProp("_CIPRank") for n in neighbors):
                sorted_neighbors = sorted(neighbors, key=lambda x: int(x.GetProp("_CIPRank")), reverse=True)
                sorted_neighbors = [a.GetIdx() for a in sorted_neighbors]
                a_idx, b_idx, c_idx = sorted_neighbors[:3] if chirality == "R" else sorted_neighbors[:3][::-1]
                d_idx = sorted_neighbors[-1]
                result.extend([
                    (a_idx, d_idx, chi_bonds[0]), (b_idx, d_idx, chi_bonds[0]), (c_idx, d_idx, chi_bonds[0]),
                    (d_idx, a_idx, chi_bonds[0]), (d_idx, b_idx, chi_bonds[0]), (d_idx, c_idx, chi_bonds[0]),
                    (b_idx, a_idx, chi_bonds[1]), (c_idx, b_idx, chi_bonds[1]), (a_idx, c_idx, chi_bonds[1])
                ])

    if len(result) > 0:
        x, y, val = [i[0] for i in result], [i[1] for i in result], [i[2] for i in result]
        # Only overwrite zeros in the original adjacency matrix
        for i, j, v in zip(x, y, val):
            if bmat[i, j] == 0:
                bmat[i, j] = v
    return bmat


def smarts_to_mol(mol):
    """
    Convert a SMARTS-derived molecule to a regular molecule.
    SMARTS molecules have query atoms/bonds that prevent Kekulize from working.
    This rebuilds the molecule with fresh atoms and bonds, preserving atom mappings.
    """
    rwmol = Chem.RWMol()

    # Add atoms (fresh, non-query)
    for atom in mol.GetAtoms():
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        rwmol.AddAtom(new_atom)

    # Add bonds (fresh, non-query)
    for bond in mol.GetBonds():
        rwmol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
        new_bond = rwmol.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        new_bond.SetIsAromatic(bond.GetIsAromatic())

    result = rwmol.GetMol()
    Chem.SanitizeMol(result)
    return result


def process_reaction(r_smarts, p_smarts, xyz_block, kekulize=False, add_stereo=False):
    """
    Process a single reaction from SMARTS and XYZ block.
    Returns a PyG Data object compatible with the megalodon framework.

    Bond encoding:
    - 0: no bond
    - 1: single
    - 2: double
    - 3: triple
    - 4: aromatic
    - 5-8: stereo bonds (5,6: chirality, 7: E, 8: Z)

    Args:
        r_smarts: Reactant SMARTS string
        p_smarts: Product SMARTS string
        xyz_block: XYZ block with TS coordinates
        kekulize: If True, kekulize aromatic bonds to explicit single/double
        add_stereo: If True, add stereo bond information (E/Z and chirality)
    """
    # Parse SMARTS
    r = Chem.MolFromSmarts(r_smarts)
    p = Chem.MolFromSmarts(p_smarts)
    Chem.SanitizeMol(r)
    Chem.SanitizeMol(p)

    if kekulize:
        # Convert SMARTS molecules to regular molecules (removes query features)
        # This is necessary because Kekulize fails silently on query bonds
        r = smarts_to_mol(r)
        p = smarts_to_mol(p)
        Chem.Kekulize(r, clearAromaticFlags=True)
        Chem.Kekulize(p, clearAromaticFlags=True)

    # Parse XYZ block to get TS coordinates
    symbol_xyz, ts_coord = read_xyz_block(xyz_block)
    ts_coord = torch.tensor(ts_coord, dtype=torch.float32)

    # Get atom mappings and reorder
    r_perm = np.array([a.GetAtomMapNum() for a in r.GetAtoms()]) - 1
    p_perm = np.array([a.GetAtomMapNum() for a in p.GetAtoms()]) - 1
    r_perm_inv = np.argsort(r_perm)
    p_perm_inv = np.argsort(p_perm)

    # Get atomic numbers
    r_atomic_numbers = np.array([r.GetAtomWithIdx(int(i)).GetAtomicNum() for i in r_perm_inv])
    p_atomic_numbers = np.array([p.GetAtomWithIdx(int(i)).GetAtomicNum() for i in p_perm_inv])

    assert np.array_equal(r_atomic_numbers, p_atomic_numbers), "Reactant and product must have same atoms"
    numbers = torch.from_numpy(r_atomic_numbers).to(torch.uint8)
    N = len(numbers)

    assert len(ts_coord) == N, f"TS coordinates must match number of atoms: {len(ts_coord)} != {N}"

    # Get adjacency matrices (binary connectivity only)
    r_adj = rdmolops.GetAdjacencyMatrix(r)
    p_adj = rdmolops.GetAdjacencyMatrix(p)

    # Permute adjacency matrices to match atom order
    r_perm_inv_int = [int(i) for i in r_perm_inv]
    p_perm_inv_int = [int(i) for i in p_perm_inv]
    r_adj_perm = r_adj[np.ix_(r_perm_inv_int, r_perm_inv_int)]
    p_adj_perm = p_adj[np.ix_(p_perm_inv_int, p_perm_inv_int)]

    # Union of both adjacency matrices for edge connectivity
    adj = r_adj_perm + p_adj_perm
    row, col = adj.nonzero()

    # Extract bond types: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic
    _nonbond = 0

    r_edge_type = []
    for i, j in zip(r_perm_inv[row], r_perm_inv[col]):
        bond = r.GetBondBetweenAtoms(int(i), int(j))
        if bond is not None:
            r_edge_type.append(BOND_TYPES.get(bond.GetBondType(), 1))  # default to single
        else:
            r_edge_type.append(_nonbond)

    p_edge_type = []
    for i, j in zip(p_perm_inv[row], p_perm_inv[col]):
        bond = p.GetBondBetweenAtoms(int(i), int(j))
        if bond is not None:
            p_edge_type.append(BOND_TYPES.get(bond.GetBondType(), 1))  # default to single
        else:
            p_edge_type.append(_nonbond)

    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    r_edge_type = torch.tensor(r_edge_type, dtype=torch.uint8)
    p_edge_type = torch.tensor(p_edge_type, dtype=torch.uint8)

    # Sort edge index by (row * N + col) - same as TSDiff
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    r_edge_type = r_edge_type[perm]
    p_edge_type = p_edge_type[perm]

    # Store as [r_edge_type, p_edge_type] pairs in edge_attr
    edge_attr = torch.stack([r_edge_type, p_edge_type], dim=-1)

    # Optionally add stereo bond information
    if add_stereo:
        # Stereo bond type indices (5-8):
        # 5: chirality bond type 1, 6: chirality bond type 2
        # 7: E stereo, 8: Z stereo
        chi_bonds = (5, 6)  # Chirality bonds
        ez_bonds = {
            Chem.BondStereo.STEREOE: 7,
            Chem.BondStereo.STEREOZ: 8,
        }

        # Build RDKit mol with 3D coordinates for stereo assignment
        r_mol = build_rdkit_mol(r_atomic_numbers, ts_coord.numpy(), r_adj_perm)
        p_mol = build_rdkit_mol(p_atomic_numbers, ts_coord.numpy(), p_adj_perm)

        # Create bond matrices for stereo augmentation
        r_bmat = np.zeros((N, N), dtype=np.int64)
        p_bmat = np.zeros((N, N), dtype=np.int64)

        # Fill with existing bond types
        for idx, (i, j) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
            r_bmat[i, j] = r_edge_type[idx].item()
            p_bmat[i, j] = p_edge_type[idx].item()

        # Add stereo bonds
        r_bmat = add_stereo_bonds(r_mol, chi_bonds, ez_bonds, r_bmat, from_3D=True)
        p_bmat = add_stereo_bonds(p_mol, chi_bonds, ez_bonds, p_bmat, from_3D=True)

        # Find new edges from stereo bonds (where bmat > 0 but wasn't in original edge_index)
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        new_edges = []
        new_r_types = []
        new_p_types = []

        for i in range(N):
            for j in range(N):
                if i != j and (i, j) not in existing_edges:
                    if r_bmat[i, j] > 0 or p_bmat[i, j] > 0:
                        new_edges.append((i, j))
                        new_r_types.append(r_bmat[i, j])
                        new_p_types.append(p_bmat[i, j])

        if new_edges:
            # Append new stereo edges
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).T
            new_r_edge_type = torch.tensor(new_r_types, dtype=torch.uint8)
            new_p_edge_type = torch.tensor(new_p_types, dtype=torch.uint8)

            edge_index = torch.cat([edge_index, new_edge_index], dim=1)
            r_edge_type = torch.cat([r_edge_type, new_r_edge_type])
            p_edge_type = torch.cat([p_edge_type, new_p_edge_type])

            # Re-sort by (row * N + col)
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            r_edge_type = r_edge_type[perm]
            p_edge_type = p_edge_type[perm]

            # Rebuild edge_attr
            edge_attr = torch.stack([r_edge_type, p_edge_type], dim=-1)

    # Charge (default to 0, may need to extract from SMARTS if available)
    charge_value = 0
    charges = torch.full_like(numbers, charge_value, dtype=torch.int8)

    # Get reactant and product coordinates from SMARTS (we'll need to embed them)
    # For now, use TS coordinates as starting point
    r_coord = ts_coord.clone()
    p_coord = ts_coord.clone()

    # Create PyG Data object
    data = Data(
        numbers=numbers,
        charges=charges,
        ts_coord=ts_coord,
        r_coord=r_coord,
        p_coord=p_coord,
        id=f"{r_smarts}>>{p_smarts}",
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=numbers.shape[0]
    )

    return data


def save_pyg_dataset(pyg_batch, save_path):
    """Save a list of PyG Data objects to disk as a batched tensor."""
    batch = collate(pyg_batch[0].__class__, pyg_batch, increment=False, add_batch=False)
    torch.save(batch[:2], save_path)  # save (data, slices)


def index_split(num_data: int, train: float = 0.8, valid: float = 0.1, seed: int = 42):
    """
    Generate randomly splitted index of data into non-overlapping train/valid/test set.
    This function assumes that the data is augmented so that original samples are placed in even index
    and the corresponding augmented samples are placed in the next index.
    This matches ts_diff/preprocessing.py exactly.
    """
    assert train + valid < 1
    random.seed(seed)
    index_list = list(range(num_data))
    random.shuffle(index_list)

    n_train = int(num_data * train)
    n_valid = int(num_data * valid)
    train_index = np.array(index_list[:n_train])
    valid_index = np.array(index_list[n_train: n_train + n_valid])
    test_index = np.array(index_list[n_train + n_valid:])

    train_index = list(np.concatenate((train_index * 2, train_index * 2 + 1)))
    valid_index = list(np.concatenate((valid_index * 2, valid_index * 2 + 1)))
    test_index = list(np.concatenate((test_index * 2, test_index * 2 + 1)))

    train_index.sort()
    valid_index.sort()
    test_index.sort()
    return train_index, valid_index, test_index


def main(args):
    # Load transition state geometry data (XYZ format)
    print("Loading XYZ corpus...")
    xyz_blocks = parse_xyz_corpus(args.ts_data)
    print(f"Loaded {len(xyz_blocks)} XYZ blocks")

    # Load reaction SMARTS data (CSV format)
    print("Loading reaction SMARTS...")
    df = pd.read_csv(args.rxn_smarts_file)
    rxn_smarts = df.AAM
    print(f"Loaded {len(rxn_smarts)} reaction SMARTS")

    assert len(xyz_blocks) == len(rxn_smarts), \
        f"Mismatch: {len(xyz_blocks)} XYZ blocks vs {len(rxn_smarts)} reaction SMARTS"

    print(f"Bond encoding: 0=none, 1=single, 2=double, 3=triple, 4=aromatic, 5-8=stereo, kekulize={args.kekulize}, add_stereo={args.add_stereo}")

    # Process all reactions
    print("Processing reactions...")
    data_list = []
    for idx, (a_smarts, xyz_block) in enumerate(tqdm(zip(rxn_smarts, xyz_blocks), total=len(rxn_smarts))):
        r, p = a_smarts.split(">>")
        try:
            data = process_reaction(r, p, xyz_block, kekulize=args.kekulize, add_stereo=args.add_stereo)
            data.rxn_index = idx // 2
            data.augmented = False if idx % 2 == 0 else True
            data_list.append(data)
        except Exception as e:
            print(f"[WARNING] Failed to process reaction {idx}: {e}")
            continue
    
    print(f"Successfully processed {len(data_list)} reactions")
    
    # Apply same split as ts_diff/preprocessing.py
    # The data is augmented: original samples in even index, augmented (reverse) in odd index
    num_original_samples = len(data_list) // 2
    train_index, valid_index, test_index = index_split(
        num_original_samples,
        train=args.train,
        valid=args.valid,
        seed=args.seed
    )
    
    # Exclude ban_index if provided
    if args.ban_index and args.ban_index[0] != -1:
        ban_index = set(args.ban_index)
        train_index = [i for i in train_index if i not in ban_index]
        valid_index = [i for i in valid_index if i not in ban_index]
        test_index = [i for i in test_index if i not in ban_index]
        print(f"After excluding ban_index: train={len(train_index)}, valid={len(valid_index)}, test={len(test_index)}")
    
    # Create split data
    train_data = [data_list[i] for i in train_index if i < len(data_list)]
    valid_data = [data_list[i] for i in valid_index if i < len(data_list)]
    test_data = [data_list[i] for i in test_index if i < len(data_list)]
    
    print(f"Split sizes: train={len(train_data)}, val={len(valid_data)}, test={len(test_data)}")
    
    # Create output directory
    save_data_path = Path(args.save_dir)
    save_data_path.mkdir(parents=True, exist_ok=True)
    save_processed_path = save_data_path / "processed"
    save_processed_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    print("\nSaving splits...")
    save_pyg_dataset(train_data, save_processed_path / "train_h.pt")
    print(f"Saved {len(train_data)} reactions to train_h.pt")
    
    save_pyg_dataset(valid_data, save_processed_path / "val_h.pt")
    print(f"Saved {len(valid_data)} reactions to val_h.pt")
    
    save_pyg_dataset(test_data, save_processed_path / "test_h.pt")
    print(f"Saved {len(test_data)} reactions to test_h.pt")
    
    print(f"\nAll splits saved to: {save_processed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TS1x dataset for training with same split as ts_diff")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting (default: 42, same as ts_diff)")
    parser.add_argument("--train", type=float, default=0.8,
                        help="Ratio of train data (default: 0.8, same as ts_diff)")
    parser.add_argument("--valid", type=float, default=0.1,
                        help="Ratio of valid data (default: 0.1, same as ts_diff)")
    parser.add_argument("--ts_data", type=str, required=True,
                        help="Path to transition state XYZ file (e.g., data/TS/wb97xd3/raw_data/wb97xd3_ts.xyz)")
    parser.add_argument("--rxn_smarts_file", type=str, required=True,
                        help="Path to reaction SMARTS CSV file (e.g., data/TS/wb97xd3/raw_data/wb97xd3_fwd_rev_chemprop.csv)")
    parser.add_argument("--save_dir", type=str, default="data/ts1x",
                        help="Directory to save the resulting PyG datasets (default: data/ts1x)")
    parser.add_argument("--ban_index", type=int, nargs="+", default=[20568, 20569, 20580, 20581],
                        help="Indices to exclude from splits (default: [20568, 20569, 20580, 20581])")
    parser.add_argument("--kekulize", action="store_true",
                        help="Kekulize aromatic bonds to explicit single/double (default: False)")
    parser.add_argument("--add_stereo", action="store_true",
                        help="Add stereo bond information (E/Z and chirality) to edge attributes (default: False)")
    args = parser.parse_args()

    main(args)
