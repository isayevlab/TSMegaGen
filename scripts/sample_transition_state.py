import os
import tempfile
import subprocess
from copy import deepcopy
from argparse import ArgumentParser

from rdkit import Chem
from rdkit.Chem import AllChem
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

Chem.SetUseLegacyStereoPerception(True)


def coords_to_xyz_string(coords, numbers):
    """Convert coordinates and atomic numbers to XYZ format string."""
    n_atoms = len(numbers)
    xyz_lines = [str(n_atoms), ""]

    for atomic_num, coord in zip(numbers, coords):
        symbol = Chem.GetPeriodicTable().GetElementSymbol(int(atomic_num))
        xyz_lines.append(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

    return "\n".join(xyz_lines)


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
    for num in numbers:
        atom = Chem.Atom(int(num))
        mol.AddAtom(atom)

    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if bond_mat[i, j] >= 1.0:
                if bond_mat[i, j] == 1:
                    bond_type = Chem.BondType.SINGLE
                elif bond_mat[i, j] == 2:
                    bond_type = Chem.BondType.DOUBLE
                elif bond_mat[i, j] == 3:
                    bond_type = Chem.BondType.TRIPLE
                else:
                    bond_type = Chem.BondType.SINGLE

                mol.AddBond(i, j, bond_type)

    mol = mol.GetMol()
    conf = Chem.Conformer(len(numbers))
    for i, pos in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*pos))
    mol.AddConformer(conf, assignId=True)
    return mol


def add_stereo_bonds(mol, chi_bonds, ez_bonds, bmat, from_3D=True):
    """Add stereo bonds to bond matrix based on RDKit molecule stereochemistry."""
    result = []

    if from_3D and mol.GetNumConformers() > 0:
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    else:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        # E/Z stereo
        if bond.GetBondType() == Chem.BondType.DOUBLE and stereo in ez_bonds:
            idx_3, idx_4 = bond.GetStereoAtoms()
            atom_1, atom_2 = bond.GetBeginAtom(), bond.GetEndAtom()
            idx_1, idx_2 = atom_1.GetIdx(), atom_2.GetIdx()

            idx_5 = [nbr.GetIdx() for nbr in atom_1.GetNeighbors()
                     if nbr.GetIdx() not in {idx_2, idx_3}]
            idx_6 = [nbr.GetIdx() for nbr in atom_2.GetNeighbors()
                     if nbr.GetIdx() not in {idx_1, idx_4}]

            inv_stereo = (Chem.BondStereo.STEREOE
                          if stereo == Chem.BondStereo.STEREOZ
                          else Chem.BondStereo.STEREOZ)
            result.extend([(idx_3, idx_4, ez_bonds[stereo]),
                           (idx_4, idx_3, ez_bonds[stereo])])

            if idx_5:
                result.extend([(idx_5[0], idx_4, ez_bonds[inv_stereo]),
                               (idx_4, idx_5[0], ez_bonds[inv_stereo])])
            if idx_6:
                result.extend([(idx_3, idx_6[0], ez_bonds[inv_stereo]),
                               (idx_6[0], idx_3, ez_bonds[inv_stereo])])
            if idx_5 and idx_6:
                result.extend([(idx_5[0], idx_6[0], ez_bonds[stereo]),
                               (idx_6[0], idx_5[0], ez_bonds[stereo])])

        # Tetrahedral chirality (CIP)
        if bond.GetBeginAtom().HasProp('_CIPCode'):
            chirality = bond.GetBeginAtom().GetProp('_CIPCode')
            neighbors = bond.GetBeginAtom().GetNeighbors()
            if all(n.HasProp("_CIPRank") for n in neighbors):
                sorted_neighbors = sorted(
                    neighbors,
                    key=lambda x: int(x.GetProp("_CIPRank")),
                    reverse=True,
                )
                sorted_neighbors = [a.GetIdx() for a in sorted_neighbors]
                a, b, c = (sorted_neighbors[:3] if chirality == "R"
                           else sorted_neighbors[:3][::-1])
                d = sorted_neighbors[-1]
                result.extend([
                    (a, d, chi_bonds[0]), (b, d, chi_bonds[0]), (c, d, chi_bonds[0]),
                    (d, a, chi_bonds[0]), (d, b, chi_bonds[0]), (d, c, chi_bonds[0]),
                    (b, a, chi_bonds[1]), (c, b, chi_bonds[1]), (a, c, chi_bonds[1]),
                ])

    if len(result) > 0:
        x, y, val = [i[0] for i in result], [i[1] for i in result], [i[2] for i in result]
        for i, j, v in zip(x, y, val):
            if bmat[i, j] == 0:
                bmat[i, j] = v
    return bmat




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


def main():
    parser = ArgumentParser(description="Sample transition states for reactions")

    parser.add_argument(
        "--reactant_smi",
        type=str,
        help="SMILES string for reactant (should already have hydrogens, supports atom mapping)",
    )
    parser.add_argument(
        "--product_smi",
        type=str,
        help="SMILES string for product (should already have hydrogens, supports atom mapping)",
    )
    parser.add_argument("--reactant_xyz", type=str, help="XYZ file path for reactant(s)")
    parser.add_argument("--product_xyz", type=str, help="XYZ file path for product(s)")

    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint file")
    parser.add_argument("--output", type=str, required=True, help="Output XYZ file path")

    parser.add_argument(
        "--n_samples", type=int, default=1, help="Number of samples per reaction"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--charge", type=int, default=0, help="Molecular charge")
    parser.add_argument(
        "--no_3d", action="store_true", help="Skip 3D embedding generation"
    )

    args = parser.parse_args()

    # Determine input type
    if args.reactant_smi and args.product_smi:
        reactant_input = args.reactant_smi
        product_input = args.product_smi
    elif args.reactant_xyz and args.product_xyz:
        reactant_input = args.reactant_xyz
        product_input = args.product_xyz
    else:
        raise ValueError(
            "Must provide either --reactant_smi/--product_smi or --reactant_xyz/--product_xyz"
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
    )
    model = model.to("cuda").eval()

    # Load reactions
    use_3d = not args.no_3d
    reaction_data_list = create_reactions_from_inputs(
        reactant_input, product_input, use_3d=use_3d, charge=args.charge
    )

    if len(reaction_data_list) == 0:
        raise ValueError("No valid reactions could be created from inputs")

    # Replicate reactions n_samples times
    all_data_list = []
    for data in reaction_data_list:
        for _ in range(args.n_samples):
            all_data_list.append(deepcopy(data))

    loader = DataLoader(all_data_list, batch_size=args.batch_size)

    # Sampling
    generated_ts_coords = []
    reference_data = []
    ids = []

    for batch in tqdm(loader, desc="Sampling transition states"):
        batch = batch.to(model.device)
        batch = batch_preprocessor(batch)

        with torch.no_grad():
            sample = model.sample(
                batch=batch, timesteps=cfg.interpolant.timesteps, pre_format=False
            )

        coords_list = convert_coords_to_np(sample)
        generated_ts_coords.extend(coords_list)

        # Store reference data
        for i in range(len(coords_list)):
            batch_mask = batch["batch"] == i
            ref_data = {
                "r_coord": batch["r_coord"][batch_mask].cpu().numpy(),
                "p_coord": batch["p_coord"][batch_mask].cpu().numpy(),
                "numbers": batch["numbers"][batch_mask].cpu().numpy(),
                "charge": args.charge,
                "id": batch["id"][i]
                if isinstance(batch["id"][i], str)
                else str(batch["id"][i]),
            }
            reference_data.append(ref_data)
            ids.append(ref_data["id"])

    # Save output as XYZ files
    base_path = (
        args.output.rsplit(".", 1)[0] if args.output.endswith(".xyz") else args.output
    )

    for idx, (coords, ref_data) in enumerate(
        zip(generated_ts_coords, reference_data)
    ):
        if len(generated_ts_coords) == 1 and args.output.endswith(".xyz"):
            xyz_path = args.output
        else:
            xyz_path = f"{base_path}_{idx + 1}.xyz"

        xyz_content = coords_to_xyz_string(coords, ref_data["numbers"])
        with open(xyz_path, "w") as f:
            f.write(xyz_content)

    print(
        f"Generated {len(generated_ts_coords)} transition states for {len(set(ids))} unique reactions."
    )
    print(
        f"Output saved to: {args.output}"
        + (f" (and numbered files)" if len(generated_ts_coords) > 1 else "")
    )


if __name__ == "__main__":
    main()
