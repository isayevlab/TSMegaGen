import argparse
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from tqdm import tqdm

from data_processing.sgdataset import SizeGroupedDataset
from rdkit.Geometry import Point3D


from rdkit import Chem


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
        conf.SetAtomPosition(i, Point3D(*pos))
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
                a, b, c = sorted_neighbors[:3] if chirality == "R" else sorted_neighbors[:3][::-1]
                d = sorted_neighbors[-1]
                result.extend([
                    (a, d, chi_bonds[0]), (b, d, chi_bonds[0]), (c, d, chi_bonds[0]),
                    (d, a, chi_bonds[0]), (d, b, chi_bonds[0]), (d, c, chi_bonds[0]),
                    (b, a, chi_bonds[1]), (c, b, chi_bonds[1]), (a, c, chi_bonds[1])
                ])

    if len(result) > 0:
        x, y, val = [i[0] for i in result], [i[1] for i in result], [i[2] for i in result]
        # Only overwrite zeros in the original adjacency matrix
        for i, j, v in zip(x, y, val):
            if bmat[i, j] == 0:
                bmat[i, j] = v
    return bmat

def process_dataset(group):
    mols = []

    for idx in range(len(group["_id"])):
        r_mol = build_rdkit_mol(group["numbers"][idx], group["r_coord"][idx], group["bmat_r"][idx])
        p_mol = build_rdkit_mol(group["numbers"][idx], group["p_coord"][idx], group["bmat_p"][idx])
        chi_bonds = [7, 8]
        ez_bonds = {Chem.BondStereo.STEREOE: 5, Chem.BondStereo.STEREOZ: 6}
        bmat_r = add_stereo_bonds(r_mol, chi_bonds, ez_bonds, group["bmat_r"][idx])
        bmat_p = add_stereo_bonds(p_mol, chi_bonds, ez_bonds, group["bmat_p"][idx])

        bmat_r = torch.from_numpy(bmat_r)
        bmat_p = torch.from_numpy(bmat_p)
        numbers = torch.from_numpy(group["numbers"][idx]).to(torch.uint8)
        r_coord = torch.from_numpy(group["r_coord"][idx])
        p_coord = torch.from_numpy(group["p_coord"][idx])
        ts_coord = torch.from_numpy(group["ts_coord"][idx])
        mol_id = str(group["_id"][idx])

        charge_value = int(group["charge"][idx])  # ensure it's a Python int
        charges = torch.full_like(numbers, charge_value, dtype=torch.int8)


        if ts_coord.shape[0] != numbers.shape[0]:
            print(f"[WARNING] Skipping molecule {mol_id} due to shape mismatch: "
                  f"ts_coord {ts_coord.shape[0]} vs numbers {numbers.shape[0]}")
            breakpoint()
            continue

        edge_index = (bmat_r + bmat_p).nonzero().contiguous().T
        edge_attr = torch.stack([bmat_r, bmat_p], dim=-1)[edge_index[0], edge_index[1]].to(torch.uint8)

        data = Data(
            numbers=numbers,
            charges=charges,
            ts_coord=ts_coord,
            r_coord=r_coord,
            p_coord=p_coord,
            id=mol_id,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=numbers.shape[0]
        )
        mols.append(data)

    return mols


def save_pyg_dataset(pyg_batch, save_path):
    """Save a list of PyG Data objects to disk as a batched tensor."""
    batch = collate(pyg_batch[0].__class__, pyg_batch, increment=False, add_batch=False)
    torch.save(batch[:2], save_path)  # save (data, slices)


def main(args):
    ds = SizeGroupedDataset.from_h5(args.h5_file)

    save_data_path = Path(args.save_data_folder)
    save_data_path.mkdir(parents=True, exist_ok=True)
    save_processed_path = save_data_path / "processed"
    save_processed_path.mkdir(parents=True, exist_ok=True)

    split = ds.random_split(0.96, 0.02, 0.02)

    print(f"Split sizes: train={len(split[0])}, val={len(split[1])}, test={len(split[2])}")

    for s, sname in zip(split, ["train", "val", "test"]):
        pyg_mols = []

        for n in tqdm(ds.keys(), desc=f"Processing {sname} set"):
            group = s[n]
            group = {key: group[key][:] for key in group.keys()}
            _pyg_mols = process_dataset(group)
            pyg_mols.extend(_pyg_mols)

        save_pyg_dataset(pyg_mols, save_processed_path / f"{sname}_h.pt")
        print(f"Saved {len(pyg_mols)} molecules to {sname}_h.pt")

    print(f"All splits saved to: {save_processed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", type=str, required=True,
                        help="Path to the h5 dataset file")
    parser.add_argument("--save_data_folder", type=str, required=True,
                        help="Directory to save the resulting PyG datasets")
    args = parser.parse_args()

    main(args)
