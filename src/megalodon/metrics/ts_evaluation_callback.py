import os
from typing import List, Dict
from pathlib import Path
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


def convert_coords_to_np(out):
    """Convert model output to list of coordinate arrays."""
    coords_list = []
    x = out["ts_coord"].cpu().numpy()
    batch = out["batch"].cpu().numpy()
    unique_mols = np.unique(batch)

    for mol_id in unique_mols:
        coords_list.append(x[batch == mol_id])

    return coords_list


class TransitionStatesEvaluationCallback(Callback):
    """
    Simplified callback for saving transition state structures during training.

    Args:
        max_molecules (int): Maximum number of molecules to save per epoch.
        timesteps (int): Number of timesteps for sampling.
        scale_coords (float): Coordinate scaling factor.
        save_dir (str): Directory to save the data (default: 'ts_data').
    """

    def __init__(
            self,
            max_molecules: int = 20,
            timesteps: int = 100,
            scale_coords: float = 1.0,
            save_dir: str = 'ts_data',
    ):
        super().__init__()
        self.max_molecules = max_molecules
        self.timesteps = timesteps
        self.scale_coords = scale_coords
        self.save_dir = Path(save_dir)
        self.molecules_data = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
            dataloader_idx=0):
        """Collect molecule data during validation."""
        if len(self.molecules_data) < self.max_molecules:
            # Generate molecules - use pre_format=False to avoid batch structure mismatch
            batch.pos = None
            out = pl_module.sample(batch=batch, timesteps=self.timesteps, pre_format=False)
            # Apply coordinate scaling
            out["ts_coord"] = self.scale_coords * out["ts_coord"]
            coords_list = convert_coords_to_np(out)

            # Store molecule data
            for i, coords in enumerate(coords_list):
                if len(self.molecules_data) >= self.max_molecules:
                    break

                # Extract atomic numbers from batch
                batch_mask = batch["batch"] == i
                numbers = batch["numbers"][batch_mask].cpu().numpy()

                # Get charge
                charge = batch.get("charges", torch.zeros_like(batch["numbers"]))[batch_mask][
                             0].cpu().item() - 4

                # Get molecule ID
                mol_id = batch["id"][i] if isinstance(batch["id"][i], str) else batch["id"][
                    i].decode('utf-8') if isinstance(batch["id"][i], bytes) else str(batch["id"][i])
                edge_index = batch["edge_index"]
                edge_mask = batch["batch"][edge_index[0]] == i
                mol_edges = edge_index[:, edge_mask]

                # Extract bond matrices
                mol_bmat_r = batch["bmat_r"][edge_mask].cpu().numpy()
                mol_bmat_p = batch["bmat_p"][edge_mask].cpu().numpy()

                # Find the starting atom index for this molecule
                mol_start_idx = edge_index[0][edge_mask].min().item()

                # Convert edge indices to local molecule indices (0-based)
                mol_edges_local = mol_edges.cpu().numpy() - mol_start_idx

                # Create adjacency matrices
                n_atoms = len(numbers)
                bmat_r = np.zeros((n_atoms, n_atoms), dtype=np.int32)
                bmat_p = np.zeros((n_atoms, n_atoms), dtype=np.int32)

                bmat_r[mol_edges_local[0, :], mol_edges_local[1, :]] = mol_bmat_r
                bmat_r[mol_edges_local[1, :], mol_edges_local[0, :]] = mol_bmat_r
                bmat_p[mol_edges_local[0, :], mol_edges_local[1, :]] = mol_bmat_p
                bmat_p[mol_edges_local[1, :], mol_edges_local[0, :]] = mol_bmat_p

                r_coords = batch["r_coord"][batch_mask].cpu().numpy()
                p_coords = batch["p_coord"][batch_mask].cpu().numpy()

                mol_data = {
                    'coords': coords,
                    'numbers': numbers,
                    'charge': charge,
                    'mol_id': mol_id,
                    'bmat_r': bmat_r,
                    'bmat_p': bmat_p,
                    'r_coords': r_coords,
                    'p_coords': p_coords,
                }
                self.molecules_data.append(mol_data)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Save collected molecules to disk."""
        if self.molecules_data:
            if len(self.molecules_data) > self.max_molecules:
                self.molecules_data = self.molecules_data[:self.max_molecules]

            # Create save directory if it doesn't exist
            self.save_dir.mkdir(exist_ok=True)

            # Get current epoch and world rank
            epoch = trainer.current_epoch
            world_rank = trainer.global_rank

            # Save the molecular data
            filename = f"generated_ts_epoch_{epoch}_world_rank_{world_rank}.pt"
            save_path = self.save_dir / filename

            torch.save(self.molecules_data, save_path)

            print(f"💾 Saved {len(self.molecules_data)} TS molecules to {save_path}")

            # Save model checkpoint from rank 0 only
            if world_rank == 0:
                model_filename = f"model_epoch_{epoch}_world_rank_{world_rank}.ckpt"
                model_save_path = self.save_dir / model_filename

                # Create checkpoint dict compatible with Lightning 2.5+
                checkpoint = {
                    'epoch': epoch,
                    'global_step': trainer.global_step,
                    'pytorch-lightning_version':
                        trainer.lightning_module.__class__.__module__.split('.')[0],
                    'state_dict': pl_module.state_dict(),
                    'lr_schedulers': [],
                    'optimizer_states': [],
                    'hyper_parameters': pl_module.hparams if hasattr(pl_module, 'hparams') else {},
                }

                # Add optimizer states if available
                if hasattr(trainer, 'optimizers') and trainer.optimizers:
                    checkpoint['optimizer_states'] = [opt.state_dict() for opt in
                                                      trainer.optimizers]

                # Add lr scheduler states if available
                if hasattr(trainer, 'lr_scheduler_configs') and trainer.lr_scheduler_configs:
                    checkpoint['lr_schedulers'] = [sch.scheduler.state_dict() for sch in
                                                   trainer.lr_scheduler_configs]

                # Save the checkpoint
                torch.save(checkpoint, model_save_path)

                print(f"💾 Saved model checkpoint to {model_save_path}")

            # Reset for next epoch
            self.molecules_data = []

    def on_train_start(self, trainer, pl_module):
        pass

    def on_train_end(self, trainer, pl_module):
        pass
