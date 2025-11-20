import os
import sys
import numpy as np
import tempfile
import gc
from tqdm import tqdm
import multiprocessing as mp

from data_processing.sgdataset import SizeGroupedDataset
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

import subprocess


# -------------------------------
# Helper functions
# -------------------------------

def coords_to_xyz_string(coords, numbers):
    """Convert coordinates and atomic numbers to XYZ format string"""
    n_atoms = len(numbers)
    xyz_lines = [str(n_atoms), ""]

    atomic_symbols = {
        1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
        14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
    }

    for atomic_num, coord in zip(numbers, coords):
        symbol = atomic_symbols.get(atomic_num, f'X{atomic_num}')
        xyz_lines.append(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

    return "\n".join(xyz_lines)


def are_bonds_identical(bmat_r, bmat_p, tolerance=1e-6):
    """Check if reactant and product bond matrices are identical"""
    if bmat_r.shape != bmat_p.shape:
        return False
    return np.allclose(bmat_r, bmat_p, atol=tolerance)


def extract_fragments_from_mol(mol):
    """Extract molecular fragments directly from RDKit molecule using GetMolFrags()"""
    from rdkit import Chem
    
    if mol is None:
        return []
    
    try:
        # Get fragment atom indices
        frag_indices = Chem.GetMolFrags(mol, asMols=False)
        
        # Convert to our format: list of (atom_idx, atomic_number) tuples
        fragments = []
        for frag_atom_indices in frag_indices:
            fragment = []
            for atom_idx in frag_atom_indices:
                atom = mol.GetAtomWithIdx(atom_idx)
                fragment.append((atom_idx, atom.GetAtomicNum()))
            fragments.append(sorted(fragment))
        
        return fragments
        
    except Exception as e:
        print(f"Warning: RDKit fragment extraction failed: {e}")
        return []


def extract_fragments(bond_matrix, atomic_numbers):
    """Legacy function - extract fragments from bond matrix (fallback)"""
    from rdkit import Chem
    
    # Create RDKit molecule from bond matrix and atomic numbers
    mol = Chem.RWMol()
    
    # Add atoms (skip padding atoms with Z=0)
    atom_map = {}  # Map from original index to mol atom index
    for i, atomic_num in enumerate(atomic_numbers):
        if atomic_num > 0:  # Skip padding atoms
            atom = Chem.Atom(int(atomic_num))
            mol_idx = mol.AddAtom(atom)
            atom_map[i] = mol_idx
    
    # Add bonds
    for i in range(len(atomic_numbers)):
        if atomic_numbers[i] == 0:  # Skip padding atoms
            continue
        for j in range(i + 1, len(atomic_numbers)):
            if atomic_numbers[j] == 0:  # Skip padding atoms
                continue
            bond_order = bond_matrix[i, j]
            if bond_order > 0:
                # Determine bond type
                if abs(bond_order - 1.0) < 1e-6:
                    bond_type = Chem.BondType.SINGLE
                elif abs(bond_order - 2.0) < 1e-6:
                    bond_type = Chem.BondType.DOUBLE
                elif abs(bond_order - 3.0) < 1e-6:
                    bond_type = Chem.BondType.TRIPLE
                else:
                    bond_type = Chem.BondType.SINGLE  # Default for fractional orders
                
                mol.AddBond(atom_map[i], atom_map[j], bond_type)
    
    # Convert to regular molecule and use the direct function
    mol = mol.GetMol()
    return extract_fragments_from_mol(mol)


def have_common_fragments(fragments_r, fragments_p, min_common_atoms=3):
    """Check if reactants and products have common fragments"""
    # Convert fragments to sets of (atom_idx, atomic_number) tuples for comparison
    r_frag_sets = [frozenset(frag) for frag in fragments_r]
    p_frag_sets = [frozenset(frag) for frag in fragments_p]
    
    for r_frag in r_frag_sets:
        for p_frag in p_frag_sets:
            # Check if fragments have significant overlap
            common = r_frag & p_frag
            if len(common) >= min_common_atoms:
                # Check if the common atoms form a connected subgraph
                common_atoms = list(common)
                if len(common_atoms) >= min_common_atoms:
                    return True
    return False


def filter_molecules_by_bonds_and_fragments(bmat_r, bmat_p, atomic_numbers, 
                                         filter_identical_bonds=True, 
                                         filter_common_fragments=True,
                                         min_common_atoms=3):
    """
    Filter molecules based on bond identity and common fragments
    
    Args:
        bmat_r: Reactant bond matrices (n_molecules, n_atoms, n_atoms)
        bmat_p: Product bond matrices (n_molecules, n_atoms, n_atoms)
        atomic_numbers: Atomic numbers for each molecule (n_molecules, n_atoms)
        filter_identical_bonds: Whether to filter molecules with identical bonds
        filter_common_fragments: Whether to filter molecules with common fragments
        min_common_atoms: Minimum number of atoms for a fragment to be considered common
    
    Returns:
        valid_indices: List of indices for molecules that pass all filters
        filtered_stats: Dictionary with filtering statistics
    """
    n_molecules = bmat_r.shape[0]
    valid_indices = []
    
    # Statistics for reporting
    identical_bonds_filtered = 0
    common_fragments_filtered = 0
    total_filtered = 0
    
    for i in range(n_molecules):
        should_filter = False
        filter_reason = []
        
        # Check for identical bonds
        if filter_identical_bonds and are_bonds_identical(bmat_r[i], bmat_p[i]):
            should_filter = True
            filter_reason.append("identical_bonds")
            identical_bonds_filtered += 1
        
        # Check for common fragments
        if filter_common_fragments and not should_filter:
            fragments_r = extract_fragments(bmat_r[i], atomic_numbers[i])
            fragments_p = extract_fragments(bmat_p[i], atomic_numbers[i])
            
            if have_common_fragments(fragments_r, fragments_p, min_common_atoms):
                should_filter = True
                filter_reason.append("common_fragments")
                common_fragments_filtered += 1
        
        if not should_filter:
            valid_indices.append(i)
        else:
            total_filtered += 1
    
    filtered_stats = {
        'total_molecules': n_molecules,
        'valid_molecules': len(valid_indices),
        'filtered_molecules': total_filtered,
        'identical_bonds_filtered': identical_bonds_filtered,
        'common_fragments_filtered': common_fragments_filtered,
        'filter_reasons': {
            'identical_bonds': filter_identical_bonds,
            'common_fragments': filter_common_fragments,
            'min_common_atoms': min_common_atoms
        }
    }
    
    return valid_indices, filtered_stats


def filter_molecules_efficient(bmat_r, bmat_p, atomic_numbers, min_fragment_size=1):
    """
    Filter molecules that have identical fragments between reactant and product.
    
    Since products and reagents are atom-mapped, we can directly compare fragments
    at the same atom positions. A molecule is filtered if it has ANY fragment 
    (connected component) that is identical in both reactant and product.
    
    This catches problematic cases like:
    - Identical bonds (entire molecule unchanged)  
    - Isolated atoms (Cl, Br, I, etc.) that remain unchanged
    - Small molecular fragments that don't participate in the reaction
    
    Args:
        bmat_r: Reactant bond matrices (n_molecules, n_atoms, n_atoms)
        bmat_p: Product bond matrices (n_molecules, n_atoms, n_atoms)
        atomic_numbers: Atomic numbers for each molecule (n_molecules, n_atoms)
        min_fragment_size: Minimum fragment size (always 1 to catch all identical fragments)
    
    Returns:
        valid_indices: List of indices for molecules that pass filtering
        filtered_stats: Dictionary with filtering statistics
    """
    n_molecules = bmat_r.shape[0]
    valid_indices = []
    
    # Statistics for reporting
    identical_bonds_filtered = 0
    common_fragments_filtered = 0
    total_filtered = 0
    
    for i in range(n_molecules):
        should_filter = False
        filter_reason = []
        
        # Check for identical bonds (same connectivity) - this is a special case of common fragments
        if np.allclose(bmat_r[i], bmat_p[i], atol=1e-6):
            should_filter = True
            filter_reason.append("identical_bonds")
            identical_bonds_filtered += 1
        
        # Check for common fragments at same atom positions
        elif not should_filter:
            # Extract fragments for both reactant and product
            fragments_r = extract_fragments(bmat_r[i], atomic_numbers[i])
            fragments_p = extract_fragments(bmat_p[i], atomic_numbers[i])
            
            # Check if any fragments have the same atom positions and connectivity
            for r_frag in fragments_r:
                for p_frag in fragments_p:
                    
                    # Check if fragments have the same atom positions (atom mapping)
                    r_positions = {pos for pos, _ in r_frag}
                    p_positions = {pos for pos, _ in p_frag}
                    
                    if r_positions == p_positions:
                        # Same atom positions, check if connectivity is identical
                        # Extract submatrices for these atom positions
                        positions = sorted(r_positions)
                        r_submatrix = bmat_r[i][np.ix_(positions, positions)]
                        p_submatrix = bmat_p[i][np.ix_(positions, positions)]
                        
                        if np.allclose(r_submatrix, p_submatrix, atol=1e-6):
                            should_filter = True
                            filter_reason.append("common_fragments")
                            common_fragments_filtered += 1
                            break
                
                if should_filter:
                    break
        
        if not should_filter:
            valid_indices.append(i)
        else:
            total_filtered += 1
    
    filtered_stats = {
        'total_molecules': n_molecules,
        'valid_molecules': len(valid_indices),
        'filtered_molecules': total_filtered,
        'identical_bonds_filtered': identical_bonds_filtered,
        'common_fragments_filtered': common_fragments_filtered,
        'filter_reasons': {
            'description': 'Filter molecules with any identical fragments between reactant and product',
            'catches': ['identical_bonds', 'isolated_atoms', 'unchanged_fragments']
        }
    }
    
    return valid_indices, filtered_stats


def filter_molecules_with_rdkit(mols_r, mols_p, max_fragments=2):
    """
    Filter molecules using RDKit molecules directly.
    
    Much more efficient than reconstructing molecules from bond matrices.
    Uses Chem.GetMolFrags() directly on the already-created molecules.
    
    Args:
        mols_r: List of reactant RDKit molecules
        mols_p: List of product RDKit molecules
        max_fragments: Maximum number of fragments allowed (default: 3)
    
    Returns:
        valid_indices: List of indices for molecules that pass filtering
        filtered_stats: Dictionary with filtering statistics
    """
    from rdkit import Chem
    import numpy as np
    
    n_molecules = len(mols_r)
    valid_indices = []
    
    # Statistics for reporting
    identical_bonds_filtered = 0
    common_fragments_filtered = 0
    failed_molecules_filtered = 0
    too_many_fragments_filtered = 0
    different_atom_counts_filtered = 0
    total_filtered = 0
    
    for mol_idx in range(n_molecules):  # Fixed: renamed 'i' to 'mol_idx' to avoid variable collision
        should_filter = False
        
        r_mol = mols_r[mol_idx]
        p_mol = mols_p[mol_idx]
        
        if r_mol is None or p_mol is None:
            # Fixed: Count failed molecules as filtered instead of silently dropping
            failed_molecules_filtered += 1
            total_filtered += 1
            continue
        
        # FILTER: if molecules have different number of atoms, filter them out
        if r_mol.GetNumAtoms() != p_mol.GetNumAtoms():
            should_filter = True
            different_atom_counts_filtered += 1
            total_filtered += 1
            continue
        
        # NEW FILTER: Check if either molecule has too many fragments
        from rdkit import Chem
        r_frag_indices = Chem.GetMolFrags(r_mol, asMols=False)
        p_frag_indices = Chem.GetMolFrags(p_mol, asMols=False)
        
        if len(r_frag_indices) > max_fragments or len(p_frag_indices) > max_fragments:
            should_filter = True
            too_many_fragments_filtered += 1
        
        # Check for identical bonds (same connectivity)
        r_bonds = set()
        for bond in r_mol.GetBonds():
            i_atom, j_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()  # Fixed: no variable collision
            bond_type = bond.GetBondTypeAsDouble()
            r_bonds.add((min(i_atom, j_atom), max(i_atom, j_atom), bond_type))
        
        p_bonds = set()
        for bond in p_mol.GetBonds():
            i_atom, j_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()  # Fixed: no variable collision
            bond_type = bond.GetBondTypeAsDouble()
            p_bonds.add((min(i_atom, j_atom), max(i_atom, j_atom), bond_type))
        
        if r_bonds == p_bonds:
            should_filter = True
            identical_bonds_filtered += 1
        else:
            # Extract fragments using RDKit
            r_fragments = extract_fragments_from_mol(r_mol)
            p_fragments = extract_fragments_from_mol(p_mol)
            
            # Check for common fragments at same atom positions with identical connectivity
            for r_frag in r_fragments:
                for p_frag in p_fragments:
                    # Check if fragments have the same atom positions
                    r_positions = {pos for pos, _ in r_frag}
                    p_positions = {pos for pos, _ in p_frag}
                    
                    if r_positions == p_positions and len(r_positions) > 0:  # Fixed: added length check
                        # Same atom positions - now check if connectivity is actually identical
                        positions = sorted(list(r_positions))
                        
                        # Extract bonds within this fragment for both molecules
                        r_frag_bonds = set()
                        for bond in r_mol.GetBonds():
                            i_atom, j_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()  # Fixed: no collision
                            if i_atom in r_positions and j_atom in r_positions:
                                bond_type = bond.GetBondTypeAsDouble()
                                r_frag_bonds.add((min(i_atom, j_atom), max(i_atom, j_atom), bond_type))
                        
                        p_frag_bonds = set()
                        for bond in p_mol.GetBonds():
                            i_atom, j_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()  # Fixed: no collision
                            if i_atom in p_positions and j_atom in p_positions:
                                bond_type = bond.GetBondTypeAsDouble()
                                p_frag_bonds.add((min(i_atom, j_atom), max(i_atom, j_atom), bond_type))
                        
                        # Only filter if connectivity within fragment is identical
                        if r_frag_bonds == p_frag_bonds:
                            should_filter = True
                            common_fragments_filtered += 1
                            break
                
                if should_filter:
                    break
        
        if not should_filter:
            valid_indices.append(mol_idx)
        else:
            total_filtered += 1
    
    filtered_stats = {
        'total_molecules': n_molecules,
        'valid_molecules': len(valid_indices),
        'filtered_molecules': total_filtered,
        'identical_bonds_filtered': identical_bonds_filtered,
        'common_fragments_filtered': common_fragments_filtered,
        'failed_molecules_filtered': failed_molecules_filtered,
        'too_many_fragments_filtered': too_many_fragments_filtered,
        'different_atom_counts_filtered': different_atom_counts_filtered,  # NEW
        'filter_reasons': {
            'description': 'Filter molecules with any identical fragments using RDKit molecules directly (FIXED VERSION + ADDITIONAL FILTERS)',
            'method': 'Chem.GetMolFrags()',
            'catches': ['identical_bonds', 'isolated_atoms', 'unchanged_fragments', 'failed_bond_inference', 'too_many_fragments', 'different_atom_counts'],
            'fixes_applied': ['variable_collision_fix', 'failed_molecule_counting', 'empty_fragment_handling'],
            'new_filters': [f'max_fragments={max_fragments}', 'filter_different_atom_counts=True']
        }
    }
    
    return valid_indices, filtered_stats


def infer_bonds_with_obabel_cli(coords, numbers, charge):
    """Infer bond matrix and charges using Open Babel CLI"""
    xyz_file = mol_file = None
    try:
        # Write temporary XYZ file
        xyz_string = coords_to_xyz_string(coords, numbers)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write(xyz_string)
            xyz_file = f.name

        # Output MOL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mol', delete=False) as f:
            mol_file = f.name

        # Run Open Babel with timeout
        cmd = ["obabel", xyz_file, "-O", mol_file, "-c", "--quiet", str(charge)]
        result = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=30)

        # Read MOL into RDKit
        mol = Chem.MolFromMolFile(mol_file, sanitize=False, removeHs=False)
        if mol is None:
            return None, None, None

        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)

        n_atoms = mol.GetNumAtoms()
        bond_orders = np.zeros((n_atoms, n_atoms), dtype=np.float32)
        formal_charges = np.array([mol.GetAtomWithIdx(i).GetFormalCharge() for i in range(n_atoms)],
                                  dtype=np.int8)

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            order = bond.GetBondTypeAsDouble()
            bond_orders[i, j] = bond_orders[j, i] = order

        return bond_orders, formal_charges, mol, True

    except Exception as e:
        return None, None, None, False
    finally:
        # Cleanup temp files
        for tmp in [xyz_file, mol_file]:
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except:
                    pass


def process_single_molecule(args):
    """Worker function for multiprocessing - processes one molecule"""
    idx, coords, numbers, charge, use_obabel = args

    if use_obabel:
        bond_orders, formal_charges, mol, success = infer_bonds_with_obabel_cli(coords, numbers, charge)
    else:
        # Fallback to RDKit if needed
        bond_orders, formal_charges, mol, success = infer_bonds_rdkit(coords, numbers, charge)

    return idx, bond_orders, formal_charges, mol, success


def infer_bonds_rdkit(coords, numbers, charge):
    """Fallback RDKit-based bond inference"""
    temp_file = None
    try:
        xyz_string = coords_to_xyz_string(coords, numbers)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write(xyz_string)
            temp_file = f.name

        mol = Chem.MolFromXYZFile(temp_file)
        if mol is None:
            return None, None, False

        conn_mol = Chem.Mol(mol)
        rdDetermineBonds.DetermineBonds(conn_mol, charge=charge,
                                        allowChargedFragments=True, embedChiral=True)

        n_atoms = conn_mol.GetNumAtoms()
        bond_orders = np.zeros((n_atoms, n_atoms), dtype=np.float32)
        formal_charges = np.array(
            [conn_mol.GetAtomWithIdx(i).GetFormalCharge() for i in range(n_atoms)], dtype=np.int8)

        for bond in conn_mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            order = bond.GetBondTypeAsDouble()
            bond_orders[i, j] = bond_orders[j, i] = order

        return bond_orders, formal_charges, conn_mol, True

    except Exception as e:
        return None, None, None, False
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass


def process_group_simple(group, group_idx, use_obabel=True, max_workers=4):
    """Process a single group using simple multiprocessing.Pool.imap"""
    n_samples = len(group)
    n_atoms = group_idx

    print(f"\nProcessing group {group_idx} with {n_samples} samples, {n_atoms} atoms")

    # Initialize result arrays
    bmat_p = np.zeros((n_samples, n_atoms, n_atoms), dtype=np.float32)
    bmat_r = np.zeros((n_samples, n_atoms, n_atoms), dtype=np.float32)
    charges_p = np.zeros((n_samples, n_atoms), dtype=np.int8)
    charges_r = np.zeros((n_samples, n_atoms), dtype=np.int8)
    
    # Store RDKit molecules for fragment analysis
    mols_p = [None] * n_samples
    mols_r = [None] * n_samples

    valid_idxs = []

    # Process products
    print("Processing products...")
    product_args = [
        (i, group['p_coord'][i], group['numbers'][i], int(group['charge'][i]), use_obabel)
        for i in range(n_samples)]

    with mp.Pool(processes=max_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_molecule, product_args),
            total=n_samples,
            desc=f"Group {group_idx} products"
        ))

    # Collect product results
    p_success = []
    for idx, bond_orders, formal_charges, mol, success in results:
        if success and bond_orders is not None and formal_charges is not None:
            bmat_p[idx] = bond_orders
            charges_p[idx] = formal_charges
            mols_p[idx] = mol
            p_success.append(idx)

    print(f"Products: {len(p_success)}/{n_samples} successful")

    # Process reactants
    print("Processing reactants...")
    reactant_args = [
        (i, group['r_coord'][i], group['numbers'][i], int(group['charge'][i]), use_obabel)
        for i in range(n_samples)]

    with mp.Pool(processes=max_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_molecule, reactant_args),
            total=n_samples,
            desc=f"Group {group_idx} reactants"
        ))

    # Collect reactant results
    r_success = []
    for idx, bond_orders, formal_charges, mol, success in results:
        if success and bond_orders is not None and formal_charges is not None:
            bmat_r[idx] = bond_orders
            charges_r[idx] = formal_charges
            mols_r[idx] = mol
            r_success.append(idx)

    print(f"Reactants: {len(r_success)}/{n_samples} successful")

    # Find molecules that succeeded for both products and reactants
    valid_idxs = sorted(list(set(p_success) & set(r_success)))
    print(f"Group {group_idx}: {len(valid_idxs)}/{n_samples} molecules processed successfully")

    if len(valid_idxs) == 0:
        print(f"Warning: No valid molecules in group {group_idx}")
        return None

    # Apply additional filtering for identical bonds and common fragments
    print("Applying bond and fragment filtering...")
    
    # Collect molecules for valid indices
    valid_mols_r = [mols_r[i] for i in valid_idxs]
    valid_mols_p = [mols_p[i] for i in valid_idxs]
    
    valid_idxs_filtered, filter_stats = filter_molecules_with_rdkit(
        valid_mols_r, valid_mols_p
    )
    
    # Convert back to original indices
    final_valid_idxs = [valid_idxs[i] for i in valid_idxs_filtered]
    
    # Print filtering statistics
    print(f"Bond/Fragment filtering results:")
    print(f"  - Total molecules after bond inference: {filter_stats['total_molecules']}")
    print(f"  - Molecules with identical bonds: {filter_stats['identical_bonds_filtered']}")
    print(f"  - Molecules with common fragments: {filter_stats['common_fragments_filtered']}")
    print(f"  - Molecules with too many fragments: {filter_stats.get('too_many_fragments_filtered', 0)}")
    print(f"  - Molecules with different atom counts: {filter_stats.get('different_atom_counts_filtered', 0)}")
    print(f"  - Failed molecules (None): {filter_stats.get('failed_molecules_filtered', 0)}")
    print(f"  - Total filtered out: {filter_stats['filtered_molecules']}")
    print(f"  - Final valid molecules: {filter_stats['valid_molecules']}")
    
    if len(final_valid_idxs) == 0:
        print(f"Warning: No valid molecules in group {group_idx} after filtering")
        return None

    # Create filtered data - only keep successful molecules that pass all filters
    filtered_data = {}
    for key in group.keys():
        filtered_data[key] = group[key][final_valid_idxs]

    # Add bond and charge data
    filtered_data['bmat_p'] = bmat_p[final_valid_idxs]
    filtered_data['bmat_r'] = bmat_r[final_valid_idxs]
    filtered_data['charge_p'] = charges_p[final_valid_idxs]
    filtered_data['charge_r'] = charges_r[final_valid_idxs]
    
    # Add filter statistics to the returned data
    filtered_data['filter_stats'] = filter_stats

    return filtered_data


def print_filtering_summary(all_filter_stats):
    """Print summary of filtering statistics across all groups"""
    print("\n" + "="*80)
    print("OVERALL FILTERING SUMMARY")
    print("="*80)
    
    total_molecules = sum(stats['total_molecules'] for stats in all_filter_stats)
    total_identical_bonds = sum(stats['identical_bonds_filtered'] for stats in all_filter_stats)
    total_common_fragments = sum(stats['common_fragments_filtered'] for stats in all_filter_stats)
    total_too_many_fragments = sum(stats.get('too_many_fragments_filtered', 0) for stats in all_filter_stats)
    total_different_atom_counts = sum(stats.get('different_atom_counts_filtered', 0) for stats in all_filter_stats)
    total_failed_molecules = sum(stats.get('failed_molecules_filtered', 0) for stats in all_filter_stats)
    total_filtered = sum(stats['filtered_molecules'] for stats in all_filter_stats)
    total_valid = sum(stats['valid_molecules'] for stats in all_filter_stats)
    
    print(f"Total molecules processed: {total_molecules}")
    print(f"Total molecules filtered out: {total_filtered}")
    print(f"  - Molecules with identical bonds: {total_identical_bonds}")
    print(f"  - Molecules with common fragments: {total_common_fragments}")
    print(f"  - Molecules with too many fragments: {total_too_many_fragments}")
    print(f"  - Molecules with different atom counts: {total_different_atom_counts}")
    print(f"  - Failed molecules (None): {total_failed_molecules}")
    print(f"Total molecules remaining: {total_valid}")
    if total_molecules > 0:
        print(f"Filtering efficiency: {total_filtered/total_molecules*100:.2f}%")
    
    if total_filtered > 0:
        print(f"\nBreakdown by filter type:")
        print(f"  - Identical bonds: {total_identical_bonds/total_filtered*100:.1f}% of filtered molecules")
        print(f"  - Common fragments: {total_common_fragments/total_filtered*100:.1f}% of filtered molecules")
        print(f"  - Too many fragments: {total_too_many_fragments/total_filtered*100:.1f}% of filtered molecules")
        print(f"  - Different atom counts: {total_different_atom_counts/total_filtered*100:.1f}% of filtered molecules")
        print(f"  - Failed molecules: {total_failed_molecules/total_filtered*100:.1f}% of filtered molecules")
    
    print("="*80)


def main():
    use_obabel = True
    max_workers = mp.cpu_count() // 2  # Use half of available cores

    dataset_path = '/home/fnikitin/3d_bench/data/ts_july_2025.h5'
    save_path = '/home/fnikitin/3d_bench/data/ts_july_2025_with_bonds_and_charges.h5'

    print(f"Loading dataset {dataset_path}...")
    print(f"Using {max_workers} worker processes")

    # Load dataset without loading everything to memory initially
    dataset = SizeGroupedDataset(dataset_path, to_memory=False)

    print(f"Dataset loaded. Groups: {list(dataset.keys())}")

    # Create new dataset for results
    processed_dataset = SizeGroupedDataset()

    # Process each group individually
    all_filter_stats = [] # Collect all filter statistics for summary
    for group_idx in sorted(dataset.keys()):
        print(f"\n{'=' * 50}")
        print(f"Processing group {group_idx}")
        print(f"{'=' * 50}")

        try:
            # Load only current group to memory
            group = dataset[group_idx]
            group.to_memory()

            # Process the group
            filtered_data = process_group_simple(group, group_idx, use_obabel, max_workers)

            if filtered_data is not None:
                # Extract filter statistics before creating DataGroup
                filter_stats = filtered_data.pop('filter_stats')
                
                # Create new DataGroup and add to processed dataset
                from bench3d.chembl_processing.sgdataset import DataGroup
                processed_group = DataGroup(filtered_data)
                processed_dataset[group_idx] = processed_group
                print(f"Successfully processed group {group_idx}")
                
                # Collect filter statistics for summary
                all_filter_stats.append(filter_stats)
            else:
                print(f"Skipping group {group_idx} - no valid molecules")

        except Exception as e:
            print(f"Error processing group {group_idx}: {e}")
            continue
        finally:
            # Cleanup
            if 'group' in locals():
                del group
            gc.collect()

        # Save intermediate results every 5 groups
        if group_idx % 5 == 0 and len(processed_dataset.keys()) > 0:
            temp_save_path = f"{save_path}.tmp_{group_idx}"
            print(f"Saving intermediate results to {temp_save_path}")
            try:
                processed_dataset.save_h5(temp_save_path)
            except Exception as e:
                print(f"Warning: Could not save intermediate results: {e}")

    # Save final results
    print(f"\nSaving final dataset to {save_path}...")
    processed_dataset.save_h5(save_path)
    print("Done!")

    # Print overall filtering summary
    print_filtering_summary(all_filter_stats)


if __name__ == "__main__":
    # Ensure proper multiprocessing behavior
    mp.set_start_method('spawn', force=True)
    main()