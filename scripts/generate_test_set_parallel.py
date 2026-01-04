#!/usr/bin/env python3
"""
Parallel TS generation script for test set.
Generates 100 samples per reaction using multiple GPUs.
Loads model once per worker to avoid checkpoint loading race conditions.
"""

import os
import sys
import argparse
from pathlib import Path
from multiprocessing import Process, Queue, Manager
import time
import traceback

# Add src directory to PYTHONPATH
project_root = Path(__file__).parent.parent.absolute()
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def parse_reaction_smiles(reaction_smiles):
    """Parse reaction SMILES into reactant and product."""
    parts = reaction_smiles.strip().split('>>')
    if len(parts) != 2:
        raise ValueError(f"Invalid reaction SMILES: {reaction_smiles}")
    return parts[0], parts[1]


def worker(gpu_id, job_queue, completed_queue, config, ckpt, output_dir, n_samples, num_steps, kekulize, add_stereo, batch_size):
    """Worker process that loads model once and generates TSs for all assigned reactions."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Import here to avoid issues with CUDA initialization before fork
    import torch
    from copy import deepcopy
    from rdkit import Chem
    from omegaconf import OmegaConf
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm

    from megalodon.models.module import Graph3DInterpolantModel
    from megalodon.data.ts_batch_preprocessor import TsBatchPreProcessor
    from megalodon.metrics.ts_evaluation_callback import convert_coords_to_np

    # Import processing function from sample_transition_state
    sys.path.insert(0, str(project_root / "scripts"))
    from sample_transition_state import process_reaction_smarts, coords_to_xyz_string

    Chem.SetUseLegacyStereoPerception(True)

    # Load model ONCE
    try:
        cfg = OmegaConf.load(config)
        batch_preprocessor = TsBatchPreProcessor(
            aug_rotations=cfg.data.get("aug_rotations", False),
            scale_coords=cfg.data.get("scale_coords", 1.0),
        )

        model = Graph3DInterpolantModel.load_from_checkpoint(
            ckpt,
            loss_params=cfg.loss,
            interpolant_params=cfg.interpolant,
            sampling_params=cfg.sample,
            batch_preprocessor=batch_preprocessor,
            strict=False,  # Allow missing interpolant buffer keys from older checkpoints
        )
        model = model.to("cuda").eval()

        timesteps = num_steps if num_steps is not None else cfg.interpolant.timesteps

    except Exception as e:
        # If model loading fails, report error for all jobs and exit
        while True:
            try:
                job = job_queue.get(timeout=1)
                if job is None:
                    break
                rxn_idx, _ = job
                completed_queue.put((rxn_idx, False, f"Model loading failed: {e}"))
            except:
                break
        return

    # Process jobs
    while True:
        try:
            job = job_queue.get(timeout=5)
        except:
            break

        if job is None:  # Poison pill
            break

        rxn_idx, reaction_smi = job
        output_file = output_dir / f"rxn_{rxn_idx:04d}.xyz"

        try:
            r_smi, p_smi = parse_reaction_smiles(reaction_smi)

            # Process reaction
            data = process_reaction_smarts(
                r_smi, p_smi,
                charge=0,
                kekulize=kekulize,
                add_stereo=add_stereo
            )

            # Replicate for n_samples
            all_data_list = [deepcopy(data) for _ in range(n_samples)]

            loader = DataLoader(all_data_list, batch_size=batch_size)

            # Sample
            generated_ts_coords = []
            numbers_list = []

            for batch in loader:
                batch = batch.to(model.device)

                with torch.no_grad():
                    sample = model.sample(
                        batch=batch, timesteps=timesteps, pre_format=True
                    )

                coords_list = convert_coords_to_np(sample)
                generated_ts_coords.extend(coords_list)

                # Get numbers for each sample in batch
                for i in range(len(coords_list)):
                    batch_mask = batch["batch"] == i
                    numbers_list.append(batch["numbers"][batch_mask].cpu().numpy())

            # Write output
            with open(output_file, "w") as f:
                for coords, numbers in zip(generated_ts_coords, numbers_list):
                    xyz_content = coords_to_xyz_string(coords, numbers)
                    f.write(xyz_content + "\n")

            completed_queue.put((rxn_idx, True, None))

        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()[-500:]}"
            completed_queue.put((rxn_idx, False, error_msg))


def progress_monitor(completed_queue, total_jobs, output_dir, n_samples):
    """Monitor and display progress."""
    completed = 0
    failed = 0
    start_time = time.time()

    log_file = output_dir / "generation_log.txt"

    with open(log_file, 'w') as f:
        f.write(f"Starting generation of {total_jobs} reactions ({n_samples} samples each)\n")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    while completed + failed < total_jobs:
        try:
            rxn_idx, success, error = completed_queue.get(timeout=60)

            if success:
                completed += 1
                status = "✓"
            else:
                failed += 1
                status = "✗"

            elapsed = time.time() - start_time
            rate = (completed + failed) / elapsed if elapsed > 0 else 0
            eta = (total_jobs - completed - failed) / rate if rate > 0 else 0

            progress_msg = (
                f"[{completed + failed}/{total_jobs}] "
                f"{status} Rxn {rxn_idx:4d} ({n_samples} samples) | "
                f"Success: {completed}, Failed: {failed} | "
                f"Rate: {rate:.2f} rxn/s | ETA: {eta/60:.1f}m"
            )

            print(progress_msg, flush=True)

            with open(log_file, 'a') as f:
                f.write(progress_msg + "\n")
                if not success:
                    f.write(f"  Error: {error}\n")

        except:
            continue

    total_time = time.time() - start_time
    total_structures = completed * n_samples
    print(f"\n{'='*70}")
    print(f"Generation complete!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Reactions: {completed}/{total_jobs} ({100*completed/total_jobs:.1f}%)")
    print(f"Total structures: {total_structures} ({n_samples} per reaction)")
    print(f"Failed: {failed}/{total_jobs} ({100*failed/total_jobs:.1f}%)")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Generate TSs for test set in parallel")
    parser.add_argument('--reactions', type=str, required=True,
                        help='File with reaction SMILES (one per line)')
    parser.add_argument('--config', type=str, default='scripts/conf/ts1x.yaml',
                        help='Config file')
    parser.add_argument('--ckpt', type=str, default='results/ts1x/checkpoints/last.ckpt',
                        help='Checkpoint file')
    parser.add_argument('--output_dir', type=str, default='data/generated_megatsgen',
                        help='Output directory')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples per reaction')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7',
                        help='Comma-separated GPU IDs to use (e.g., "0,1,2,3")')
    parser.add_argument('--jobs_per_gpu', type=int, default=1,
                        help='Number of parallel jobs per GPU (default: 1, recommended)')
    parser.add_argument('--start_rxn', type=int, default=0,
                        help='Start reaction index')
    parser.add_argument('--end_rxn', type=int, default=None,
                        help='End reaction index (exclusive)')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Number of diffusion steps (overrides config value)')
    parser.add_argument('--kekulize', action='store_true',
                        help='Kekulize aromatic bonds to explicit single/double')
    parser.add_argument('--add_stereo', action='store_true',
                        help='Add stereo bond information (E/Z and chirality)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for sampling')

    args = parser.parse_args()

    # Parse GPU list
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    n_gpus = len(gpu_ids)

    # Load reactions
    with open(args.reactions) as f:
        reactions = [line.strip() for line in f if line.strip()]

    end_rxn = args.end_rxn if args.end_rxn is not None else len(reactions)
    reactions = reactions[args.start_rxn:end_rxn]

    print(f"{'='*70}")
    print(f"TS Generation Configuration")
    print(f"{'='*70}")
    print(f"Reactions: {len(reactions)} (indices {args.start_rxn} to {end_rxn-1})")
    print(f"Samples per reaction: {args.n_samples}")
    print(f"Total jobs: {len(reactions)} (1 per reaction)")
    print(f"Total structures: {len(reactions) * args.n_samples}")
    print(f"GPUs: {gpu_ids}")
    print(f"Jobs per GPU: {args.jobs_per_gpu}")
    print(f"Total parallel workers: {n_gpus * args.jobs_per_gpu}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    if args.num_steps is not None:
        print(f"Diffusion steps: {args.num_steps} (overriding config)")
    print(f"Kekulize: {args.kekulize}")
    print(f"Add stereo: {args.add_stereo}")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create job queue: (rxn_idx, reaction_smi)
    manager = Manager()
    job_queue = manager.Queue()
    completed_queue = manager.Queue()

    # Fill job queue - one job per reaction
    total_jobs = 0
    for rxn_idx, reaction_smi in enumerate(reactions):
        job_queue.put((rxn_idx + args.start_rxn, reaction_smi))
        total_jobs += 1

    # Add poison pills
    for _ in range(n_gpus * args.jobs_per_gpu):
        job_queue.put(None)

    # Start progress monitor
    monitor = Process(target=progress_monitor, args=(completed_queue, total_jobs, output_dir, args.n_samples))
    monitor.start()

    # Start worker processes
    workers = []
    for gpu_id in gpu_ids:
        for _ in range(args.jobs_per_gpu):
            p = Process(
                target=worker,
                args=(gpu_id, job_queue, completed_queue, args.config, args.ckpt,
                      output_dir, args.n_samples, args.num_steps,
                      args.kekulize, args.add_stereo, args.batch_size)
            )
            p.start()
            workers.append(p)

    # Wait for workers
    for p in workers:
        p.join()

    # Wait for monitor
    monitor.join()

    print("\nSplitting and aggregating results...")
    split_and_aggregate(output_dir, args.n_samples, len(reactions), args.start_rxn)


def split_and_aggregate(output_dir, n_samples, n_reactions, start_rxn):
    """Split multi-sample XYZ files and aggregate into seed files (like TSDiff structure)."""
    output_dir = Path(output_dir)

    print(f"Splitting {n_reactions} reaction files ({n_samples} samples each)...")

    # For each reaction, split its XYZ file into n_samples individual structures
    for rxn_idx in range(n_reactions):
        rxn_file = output_dir / f"rxn_{rxn_idx + start_rxn:04d}.xyz"

        if not rxn_file.exists():
            print(f"Warning: Missing {rxn_file}")
            continue

        # Read all structures from this reaction file
        with open(rxn_file) as f:
            content = f.read()

        # Split into individual structures
        structures = []
        lines = content.strip().split('\n')
        i = 0
        while i < len(lines):
            try:
                n_atoms = int(lines[i].strip())
                # Each structure: n_atoms line + comment + n_atoms coordinate lines
                structure_lines = lines[i:i+2+n_atoms]
                if len(structure_lines) == 2 + n_atoms:
                    structures.append('\n'.join(structure_lines))
                i += 2 + n_atoms
            except (ValueError, IndexError):
                i += 1

        if len(structures) != n_samples:
            print(f"Warning: Rxn {rxn_idx+start_rxn} has {len(structures)} structures, expected {n_samples}")

        # Save each structure to appropriate seed directory
        for seed_idx, structure in enumerate(structures):
            seed_dir = output_dir / f"seed_{seed_idx}"
            seed_dir.mkdir(exist_ok=True)

            seed_file = seed_dir / f"rxn_{rxn_idx + start_rxn:04d}.xyz"
            with open(seed_file, 'w') as f:
                f.write(structure + '\n')

        if (rxn_idx + 1) % 100 == 0:
            print(f"  Split {rxn_idx + 1}/{n_reactions} reactions")

    print(f"\n✓ Split complete! Created {n_samples} seed directories")


if __name__ == '__main__':
    main()
