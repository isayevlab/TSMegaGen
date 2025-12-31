import os
import tempfile
import subprocess
from copy import deepcopy
from argparse import ArgumentParser

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Geometry import Point3D

from tqdm import tqdm
from torch_geometric.data import DataLoader
import torch
import numpy as np
from omegaconf import OmegaConf
from torch_geometric.data import Data

from megalodon.models.module import Graph3DInterpolantModel
from megalodon.data.ts_batch_preprocessor import TsBatchPreProcessor
from megalodon.metrics.ts_evaluation_callback import convert_coords_to_np

# Simple bond type encoding:
# 0: no bond, 1: single, 2: double, 3: triple, 4: aromatic, 5-8: stereo bonds
BOND_TYPES = {
    BT.SINGLE: 1,
    BT.DOUBLE: 2,
    BT.TRIPLE: 3,
    BT.AROMATIC: 4,
}
NUM_BOND_TYPES = 9  # 0-8 inclusive

Chem.SetUseLegacyStereoPerception(True)


def infer_bonds_with_obabel(xyz_path, charge=0):
    """Infer bonds using Open Babel CLI."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mol', delete=False) as f:
        mol_file = f.name

    cmd = ["obabel", xyz_path, "-O", mol_file, "-c", "--quiet", str(charge)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=30)

    mol = Chem.MolFromMolFile(mol_file, sanitize=False, removeHs=False)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    os.unlink(mol_file)

    return mol


def build_rdkit_mol(numbers, coords, bond_mat):
    """Build RDKit molecule from atomic numbers, coordinates, and bond matrix."""
    mol = Chem.RWMol()
    bond_num_to_type = {v: k for k, v in BOND_TYPES.items()}
    for num in numbers:
        atom = Chem.Atom(int(num))
        mol.AddAtom(atom)

    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            bond_type = bond_num_to_type[bond_mat[i, j]]
            mol.AddBond(i, j, bond_type)

    mol = mol.GetMol()
    conf = Chem.Conformer(len(numbers))
    for i, pos in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*pos))
    mol.AddConformer(conf, assignId=True)
    return mol


def add_stereo_bonds(mol, chi_bonds, ez_bonds, bmat, from_3D=False):
    """
    Add stereo bond information to adjacency matrix.

    Args:
        mol: RDKit molecule
        chi_bonds: tuple of (chi_bond_1, chi_bond_2) indices for chirality
        ez_bonds: dict mapping BondStereo to bond type index
        bmat: adjacency matrix to modify
        from_3D: if True, assign stereo from 3D coords; if False, use SMARTS notation
    """
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


# ============================================================================
# XYZ loading functions
# ============================================================================

def load_molecules_from_input(input_path_or_str, use_3d=True):
    """
    Load molecules from SMILES or XYZ file.
    SMILES should already have hydrogens and may have atom mapping.
    Assumes inputs are valid and sanitizable.
    """
    mols = []

    if os.path.isfile(input_path_or_str):
        if input_path_or_str.endswith(".xyz"):
            # Read XYZ with RDKit and assign bonds with OpenBabel
            mol = infer_bonds_with_obabel(input_path_or_str)
            mols = [mol]

        elif input_path_or_str.endswith((".smi", ".smiles")):
            with open(input_path_or_str) as f:
                smiles_list = [line.strip().split()[0] for line in f if line.strip()]
            for smi in smiles_list:
                parser_params = Chem.SmilesParserParams()
                parser_params.removeHs = False
                parser_params.sanitize = False
                mol = Chem.MolFromSmiles(smi, parser_params)
                Chem.SanitizeMol(mol)
                Chem.Kekulize(mol, clearAromaticFlags=True)

                if use_3d and mol.GetNumConformers() == 0:
                    embed_params = AllChem.ETKDGv3()
                    embed_params.randomSeed = 42
                    embed_params.useRandomCoords = True
                    AllChem.EmbedMolecule(mol, embed_params)

                mols.append(mol)
        else:
            raise ValueError(
                f"Unsupported input format: {input_path_or_str}. Only .xyz and .smi/.smiles files are supported."
            )
    else:
        # Treat as SMILES string (supports atom mapping, explicit H)
        parser_params = Chem.SmilesParserParams()
        parser_params.removeHs = False
        parser_params.sanitize = False
        mol = Chem.MolFromSmiles(input_path_or_str, parser_params)
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)

        if use_3d:
            embed_params = AllChem.ETKDGv3()
            embed_params.randomSeed = 42
            embed_params.useRandomCoords = True
            AllChem.EmbedMolecule(mol, embed_params)

        mols = [mol]

    return mols


def renumber_by_atom_map(mol: Chem.Mol) -> Chem.Mol:
    """
    Renumber atoms so that mapped atoms appear first, ordered by atom-map number.
    Unmapped atoms (mapnum == 0) follow in their original order.

    This does NOT require map numbers to be contiguous or start at 1.
    """
    map_nums = [a.GetAtomMapNum() for a in mol.GetAtoms()]

    # No maps at all: leave molecule as-is
    if all(mn == 0 for mn in map_nums):
        return mol

    # Indices of mapped and unmapped atoms
    mapped = [(idx, mn) for idx, mn in enumerate(map_nums) if mn > 0]
    unmapped = [idx for idx, mn in enumerate(map_nums) if mn == 0]

    # Sort mapped atoms by their map number
    mapped_sorted = [idx for idx, mn in sorted(mapped, key=lambda x: x[1])]

    # New order: all mapped (sorted by mapnum), then unmapped in original order
    order = mapped_sorted + unmapped

    return Chem.RenumberAtoms(mol, order)


def create_reaction_data(r_mol, p_mol, charge=0):
    """
    Create a reaction Data object from reactant and product molecules.
    If atom-mapped, atoms are aligned by map numbers.
    """
    # Align atom order by mapping if present
    r_mol = renumber_by_atom_map(r_mol)
    p_mol = renumber_by_atom_map(p_mol)

    # Get atomic numbers
    r_numbers = np.array([atom.GetAtomicNum() for atom in r_mol.GetAtoms()],
                         dtype=np.uint8)
    p_numbers = np.array([atom.GetAtomicNum() for atom in p_mol.GetAtoms()],
                         dtype=np.uint8)

    # Check that atom counts match
    if len(r_numbers) != len(p_numbers):
        raise ValueError(
            f"Reactant and product have different atom counts: "
            f"{len(r_numbers)} vs {len(p_numbers)}."
        )

    # Check that atomic numbers match
    if not np.array_equal(r_numbers, p_numbers):
        raise ValueError(
            "Reactant and product must have the same atoms in the same order.\n"
            f"Reactant Z: {r_numbers}\nProduct  Z: {p_numbers}"
        )

    # Get coordinates
    if r_mol.GetNumConformers() == 0 or p_mol.GetNumConformers() == 0:
        raise ValueError("Both reactant and product must have 3D coordinates.")

    r_coords = r_mol.GetConformer().GetPositions()
    p_coords = p_mol.GetConformer().GetPositions()

    # Get bond matrices
    r_bond_mat = Chem.rdmolops.GetAdjacencyMatrix(r_mol, useBO=True)
    p_bond_mat = Chem.rdmolops.GetAdjacencyMatrix(p_mol, useBO=True)
    r_bond_mat[r_bond_mat == 1.5] = 4
    p_bond_mat[p_bond_mat == 1.5] = 4
    r_bond_mat = r_bond_mat.astype(np.int32)
    p_bond_mat = p_bond_mat.astype(np.int32)

    # Build RDKit molecules for stereo assignment
    r_mol_for_stereo = build_rdkit_mol(r_numbers, r_coords, r_bond_mat)
    p_mol_for_stereo = build_rdkit_mol(p_numbers, p_coords, p_bond_mat)

    # Add stereo bonds
    chi_bonds = [7, 8]
    ez_bonds = {
        Chem.BondStereo.STEREOE: 5,
        Chem.BondStereo.STEREOZ: 6
    }
    r_bond_mat = add_stereo_bonds(
        r_mol_for_stereo, chi_bonds, ez_bonds, r_bond_mat.copy(), from_3D=True
    )
    p_bond_mat = add_stereo_bonds(
        p_mol_for_stereo, chi_bonds, ez_bonds, p_bond_mat.copy(), from_3D=True
    )

    # Convert to torch tensors
    numbers = torch.from_numpy(r_numbers).to(torch.uint8)
    r_coord = torch.from_numpy(r_coords).float()
    p_coord = torch.from_numpy(p_coords).float()
    bmat_r = torch.from_numpy(r_bond_mat)
    bmat_p = torch.from_numpy(p_bond_mat)
    charges = torch.full_like(numbers, charge, dtype=torch.int8)

    # Create edge index and edge attributes
    edge_index = (bmat_r + bmat_p).nonzero().contiguous().T
    edge_attr = torch.stack([bmat_r, bmat_p], dim=-1)[edge_index[0],
                                                      edge_index[1]].to(torch.uint8)

    # Create initial TS coordinates (midpoint between reactant and product)
    ts_coord = (r_coord + p_coord) / 2.0

    return Data(
        numbers=numbers,
        charges=charges,
        r_coord=r_coord,
        p_coord=p_coord,
        ts_coord=ts_coord,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(numbers),
        id=f"rxn_{len(numbers)}",
    )


def create_reactions_from_inputs(reactant_input, product_input, use_3d=True, charge=0):
    """Create reaction Data objects from reactant and product inputs."""
    r_mols = load_molecules_from_input(reactant_input, use_3d=use_3d)
    p_mols = load_molecules_from_input(product_input, use_3d=use_3d)

    if len(r_mols) != len(p_mols):
        raise ValueError(
            f"Number of reactants ({len(r_mols)}) must match number of products ({len(p_mols)})."
        )

    return [
        create_reaction_data(r_mol, p_mol, charge=charge)
        for r_mol, p_mol in zip(r_mols, p_mols)
    ]


# ============================================================================
# SMARTS processing
# ============================================================================

def process_reaction_smarts(r_smarts, p_smarts, charge=0, kekulize=False, add_stereo=False):
    """
    Process reaction SMARTS into a megalodon-compatible Data object.

    Bond encoding: 0=none, 1=single, 2=double, 3=triple, 4=aromatic, 5-8=stereo

    Args:
        r_smarts: Reactant SMARTS string (with atom mapping)
        p_smarts: Product SMARTS string (with atom mapping)
        charge: Molecular charge
        kekulize: If True, kekulize aromatic bonds to explicit single/double
        add_stereo: If True, add stereo bond information (E/Z and chirality)

    Returns:
        PyG Data object with: numbers, charges, edge_index, edge_attr, ts_coord, etc.
    """
    r = Chem.MolFromSmarts(r_smarts)
    p = Chem.MolFromSmarts(p_smarts)
    Chem.SanitizeMol(r)
    Chem.SanitizeMol(p)

    if kekulize:
        r = smarts_to_mol(r)
        p = smarts_to_mol(p)
        Chem.Kekulize(r, clearAromaticFlags=True)
        Chem.Kekulize(p, clearAromaticFlags=True)

    N = r.GetNumAtoms()
    assert p.GetNumAtoms() == N, f"Reactant has {N} atoms but product has {p.GetNumAtoms()}"

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

    # Get adjacency matrices
    r_adj = Chem.rdmolops.GetAdjacencyMatrix(r)
    p_adj = Chem.rdmolops.GetAdjacencyMatrix(p)
    r_adj_perm = r_adj[r_perm_inv, :].T[r_perm_inv, :].T
    p_adj_perm = p_adj[p_perm_inv, :].T[p_perm_inv, :].T

    # Union of adjacency matrices for edge connectivity
    adj = r_adj_perm + p_adj_perm
    row, col = adj.nonzero()

    # Extract bond types: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic
    _nonbond = 0

    r_edge_type = []
    for i, j in zip(r_perm_inv[row], r_perm_inv[col]):
        b = r.GetBondBetweenAtoms(int(i), int(j))
        if b is not None:
            r_edge_type.append(BOND_TYPES.get(b.GetBondType(), 1))
        else:
            r_edge_type.append(_nonbond)

    p_edge_type = []
    for i, j in zip(p_perm_inv[row], p_perm_inv[col]):
        b = p.GetBondBetweenAtoms(int(i), int(j))
        if b is not None:
            p_edge_type.append(BOND_TYPES.get(b.GetBondType(), 1))
        else:
            p_edge_type.append(_nonbond)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    r_edge_type = torch.tensor(r_edge_type, dtype=torch.uint8)
    p_edge_type = torch.tensor(p_edge_type, dtype=torch.uint8)

    # Sort by (row * N + col)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    r_edge_type = r_edge_type[perm]
    p_edge_type = p_edge_type[perm]

    # Optionally add stereo bond information
    if add_stereo:
        chi_bonds = (5, 6)
        ez_bonds = {
            Chem.BondStereo.STEREOE: 7,
            Chem.BondStereo.STEREOZ: 8,
        }

        r_bmat = np.zeros((N, N), dtype=np.int64)
        p_bmat = np.zeros((N, N), dtype=np.int64)

        for idx, (i, j) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
            r_bmat[i, j] = r_edge_type[idx].item()
            p_bmat[i, j] = p_edge_type[idx].item()

        r_bmat = add_stereo_bonds(r, chi_bonds, ez_bonds, r_bmat, from_3D=False)
        p_bmat = add_stereo_bonds(p, chi_bonds, ez_bonds, p_bmat, from_3D=False)

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
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).T
            new_r_edge_type = torch.tensor(new_r_types, dtype=torch.uint8)
            new_p_edge_type = torch.tensor(new_p_types, dtype=torch.uint8)

            edge_index = torch.cat([edge_index, new_edge_index], dim=1)
            r_edge_type = torch.cat([r_edge_type, new_r_edge_type])
            p_edge_type = torch.cat([p_edge_type, new_p_edge_type])

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            r_edge_type = r_edge_type[perm]
            p_edge_type = p_edge_type[perm]

    # Store edge_attr as [r_edge_type, p_edge_type] - NOT combined encoding
    edge_attr = torch.stack([r_edge_type, p_edge_type], dim=-1)

    # Coordinates - zeros since we're sampling
    pos = torch.zeros(N, 3, dtype=torch.float32)

    smiles = f"{r_smarts}>>{p_smarts}"

    data = Data(
        numbers=numbers,
        charges=torch.full((N,), charge, dtype=torch.int8),
        ts_coord=pos,
        r_coord=pos.clone(),
        p_coord=pos.clone(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=N,
        id=smiles,
    )
    return data


def coords_to_xyz_string(coords, numbers):
    """Convert coordinates and atomic numbers to XYZ format string."""
    n_atoms = len(numbers)
    xyz_lines = [str(n_atoms), ""]

    for atomic_num, coord in zip(numbers, coords):
        symbol = Chem.GetPeriodicTable().GetElementSymbol(int(atomic_num))
        xyz_lines.append(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

    return "\n".join(xyz_lines)


# ============================================================================
# Main function
# ============================================================================

def main():
    parser = ArgumentParser(description="Sample transition states from reaction SMARTS")

    # Input options
    parser.add_argument(
        "--reactant_smi",
        type=str,
        help="Reactant SMARTS string (with atom mapping)",
    )
    parser.add_argument(
        "--product_smi",
        type=str,
        help="Product SMARTS string (with atom mapping)",
    )
    parser.add_argument(
        "--reaction_smarts",
        type=str,
        help="Reaction SMARTS string (e.g., 'R>>P' format with atom mapping)",
    )
    parser.add_argument(
        "--reaction_file",
        type=str,
        help="File containing reaction SMARTS (one per line)",
    )
    parser.add_argument("--reactant_xyz", type=str, help="XYZ file path for reactant(s)")
    parser.add_argument("--product_xyz", type=str, help="XYZ file path for product(s)")

    # Model options
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint file")
    parser.add_argument("--output", type=str, required=True, help="Output XYZ file path")

    # Sampling options
    parser.add_argument(
        "--n_samples", type=int, default=1, help="Number of samples per reaction"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--charge", type=int, default=0, help="Molecular charge")
    parser.add_argument(
        "--num_steps", type=int, default=None, help="Number of diffusion steps (overrides config)"
    )
    parser.add_argument(
        "--kekulize", action="store_true", help="Kekulize aromatic bonds to explicit single/double"
    )
    parser.add_argument(
        "--add_stereo", action="store_true", help="Add stereo bond information (E/Z and chirality)"
    )

    args = parser.parse_args()

    # Determine input type and create data
    use_xyz_input = args.reactant_xyz and args.product_xyz
    use_smarts_input = (args.reactant_smi and args.product_smi) or args.reaction_smarts or args.reaction_file

    if use_xyz_input and use_smarts_input:
        raise ValueError("Cannot use both XYZ and SMARTS inputs simultaneously")

    if not use_xyz_input and not use_smarts_input:
        raise ValueError(
            "Must provide either --reactant_xyz/--product_xyz, "
            "--reactant_smi/--product_smi, --reaction_smarts, or --reaction_file"
        )

    # Load model
    cfg = OmegaConf.load(args.config)
    batch_preprocessor = TsBatchPreProcessor(
        aug_rotations=cfg.data.get("aug_rotations", False),
        scale_coords=cfg.data.get("scale_coords", 1.0),
    )

    model = Graph3DInterpolantModel.load_from_checkpoint(
        args.ckpt,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=batch_preprocessor,
        strict=False,
    )
    model = model.to("cuda").eval()

    # Process reactions based on input type
    all_data_list = []

    if use_xyz_input:
        # XYZ input path
        print(f"Loading reactions from XYZ files: {args.reactant_xyz}, {args.product_xyz}")
        reaction_data_list = create_reactions_from_inputs(
            args.reactant_xyz, args.product_xyz, use_3d=True, charge=args.charge
        )
        # Replicate for n_samples
        for data in reaction_data_list:
            for _ in range(args.n_samples):
                all_data_list.append(deepcopy(data))
        print(f"Created {len(all_data_list)} data samples for {len(reaction_data_list)} reaction(s)")

    else:
        # SMARTS input path
        if args.reactant_smi and args.product_smi:
            reaction_smarts_list = [f"{args.reactant_smi}>>{args.product_smi}"]
        elif args.reaction_smarts:
            reaction_smarts_list = [args.reaction_smarts]
        elif args.reaction_file:
            with open(args.reaction_file) as f:
                reaction_smarts_list = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        print(f"Processing {len(reaction_smarts_list)} reaction(s)")

        for reaction_smarts in reaction_smarts_list:
            try:
                r_smi, p_smi = reaction_smarts.split(">>")
                data = process_reaction_smarts(
                    r_smi, p_smi,
                    charge=args.charge,
                    kekulize=args.kekulize,
                    add_stereo=args.add_stereo
                )
                # Replicate for n_samples
                for _ in range(args.n_samples):
                    all_data_list.append(deepcopy(data))
            except Exception as e:
                print(f"Warning: Failed to process reaction '{reaction_smarts[:50]}...': {e}")
                continue

        print(f"Created {len(all_data_list)} data samples for {len(reaction_smarts_list)} reaction(s)")

    if len(all_data_list) == 0:
        raise ValueError("No valid reactions could be created from inputs")

    loader = DataLoader(all_data_list, batch_size=args.batch_size)

    # Sampling
    generated_ts_coords = []
    reference_data = []
    ids = []

    # Override timesteps if specified
    timesteps = args.num_steps if args.num_steps is not None else cfg.interpolant.timesteps

    for batch in tqdm(loader, desc="Sampling transition states"):
        batch = batch.to(model.device)
        batch = batch_preprocessor(batch)

        with torch.no_grad():
            sample = model.sample(
                batch=batch, timesteps=timesteps, pre_format=False
            )

        coords_list = convert_coords_to_np(sample)
        generated_ts_coords.extend(coords_list)

        # Store reference data
        for i in range(len(coords_list)):
            batch_mask = batch["batch"] == i
            ref_data = {
                "numbers": batch["numbers"][batch_mask].cpu().numpy(),
                "charge": args.charge,
                "id": batch["id"][i] if isinstance(batch["id"][i], str) else str(batch["id"][i]),
            }
            reference_data.append(ref_data)
            ids.append(ref_data["id"])

    # Save output as XYZ file(s)
    if len(generated_ts_coords) == 1:
        # Single sample: write to output file directly
        xyz_content = coords_to_xyz_string(generated_ts_coords[0], reference_data[0]["numbers"])
        with open(args.output, "w") as f:
            f.write(xyz_content)
    else:
        # Multiple samples: write all to a single XYZ file
        with open(args.output, "w") as f:
            for coords, ref_data in zip(generated_ts_coords, reference_data):
                xyz_content = coords_to_xyz_string(coords, ref_data["numbers"])
                f.write(xyz_content + "\n")

    print(
        f"Generated {len(generated_ts_coords)} transition states for {len(set(ids))} unique reactions."
    )
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
