# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_add
from torch_geometric.data import Data

from megalodon.dynamics.eqgat.eqgat_wrapper import EQGATWrapper
from megalodon.dynamics.fn_model import MegaFNV3, MegaFNV3Conf, MegaFNV3TS
from megalodon.dynamics.mega_large import MegalodonDotFN
from megalodon.dynamics.megaflow_semla_ckpt.mimic_semla_wrapper import MimicSemlaWrapper
from megalodon.dynamics.megaflow_semla_ckpt.original_semla_ckpt import \
    MimicOriginalSemlaWrapper
from megalodon.dynamics.megaflow_semla_ckpt.semla_wrapper import SemlaWrapper
from megalodon.dynamics.jodo import DGT_concat
from megalodon.dynamics.nextmol import DGTDiffusion


class ModelBuilder:
    """A builder class for creating model instances based on a given model name and arguments."""

    def __init__(self):
        """Initializes the ModelBuilder with a dictionary of available model classes."""
        self.model_classes = {"megav3": MegaFNV3Wrapper,
                              "megav3conf": MegaFNV3ConfWrapper,
                              "megav3ts": MegaFNV3TSWrapper,
                              "mimic_semla": MimicSemlaWrapper, 
                              "mega_large": MegaLargeWrapper,
                              "original_semla": MimicOriginalSemlaWrapper,
                              "semla": SemlaWrapper,
                              "eqgat": EQGATWrapper,
                              "jodo": JODOWrapper,
                              "nextmolconf": NextMolConfWrapper,
                              "gotennet": GotenNetWrapper,
                              }

    def create_model(self, model_name: str, args_dict: dict, wrapper_args: dict):
        """
        Creates an instance of the specified model.

        Args:
            model_name (str): The name of the model to create.
            args_dict (dict): A dictionary of arguments to pass to the model.

        Returns:
            nn.Module: An instance of the specified model.

        Raises:
            ValueError: If the model name is not recognized.
        """
        args_dict = args_dict if args_dict is not None else {}
        wrapper_args = wrapper_args if wrapper_args is not None else {}
        model_class = self.model_classes.get(model_name.lower())
        if model_class is None:
            raise ValueError(f"Unknown model name: {model_name}")
        return model_class(args_dict, **wrapper_args)

    def register_model(self, model_name: str, model_class):
        """
        Registers a new model class with the builder.

        Args:
            model_name (str): The name of the model.
            model_class (type): The class of the model.
        """
        self.model_classes[model_name.lower()] = model_class


class MegaFNV3Wrapper(MegaFNV3):
    """A wrapper class for the MoCo model."""

    def __init__(self, args_dict, time_type="continuous", timesteps=None):
        """
        Initializes the DiTWrapper.

        Args:
            args_dict (dict): A dictionary of arguments for initializing the MoCo model.
        """
        self.args = args_dict
        self.time_type = time_type
        self.timesteps = timesteps
        super().__init__(**args_dict)

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        """
        Forward pass of the MoCo model.

        Args:
            batch (torch_geometric.data.Batch): The input batch.
            time (Tensor): The time tensor.

        Returns:
            dict: The output of the MoCo model.
        """
        timesteps = timesteps if timesteps is not None else self.timesteps
        if self.time_type == "discrete" and timesteps is not None:
            time = (timesteps - time.float()) / timesteps
        out = super().forward(
            batch=batch["batch"],
            X=batch["x_t"],
            H=batch["h_t"],
            E=batch["edge_attr_t"],
            E_idx=batch["edge_index"],
            t=time,
        )
        return out


class MegaSmallWrapper(MegaFNV3):
    """A wrapper class for the MoCo model."""

    def __init__(self, args_dict, time_type="continuous", timesteps=None):
        """
        Initializes the DiTWrapper.

        Args:
            args_dict (dict): A dictionary of arguments for initializing the MoCo model.
        """
        self.args = args_dict
        self.time_type = time_type
        self.timesteps = timesteps
        super().__init__(**args_dict)

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        """
        Forward pass of the MoCo model.

        Args:
            batch (torch_geometric.data.Batch): The input batch.
            time (Tensor): The time tensor.

        Returns:
            dict: The output of the MoCo model.
        """
        timesteps = timesteps if timesteps is not None else self.timesteps
        if self.time_type == "discrete" and timesteps is not None:
            time = (timesteps - time.float()) / timesteps
        out = super().forward(
            batch=batch["batch"],
            X=batch["x_t"],
            H=batch["h_t"],
            E=batch["edge_attr_t"],
            E_idx=batch["edge_index"],
            t=time,
        )
        return out


class MegaLargeWrapper(MegalodonDotFN):
    """A wrapper class for the MoCo model."""

    def __init__(self, args_dict, time_type="continuous", timesteps=None):
        """
        Initializes the DiTWrapper.

        Args:
            args_dict (dict): A dictionary of arguments for initializing the MoCo model.
        """
        self.args = args_dict
        self.time_type = time_type
        self.timesteps = timesteps
        super().__init__(**args_dict)

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        """
        Forward pass of the MoCo model.

        Args:
            batch (torch_geometric.data.Batch): The input batch.
            time (Tensor): The time tensor.

        Returns:
            dict: The output of the MoCo model.
        """
        timesteps = timesteps if timesteps is not None else self.timesteps
        if self.time_type == "discrete" and timesteps is not None:
            time = (timesteps - time.float()) / timesteps
        out = super().forward(
            batch=batch["batch"],
            X=batch["x_t"],
            H=batch["h_t"],
            E=batch["edge_attr_t"],
            E_idx=batch["edge_index"],
            t=time,
        )
        return out
    

class MegaFNV3ConfWrapper(MegaFNV3Conf):
    """A wrapper class for the MoCo model."""

    def __init__(self, args_dict, time_type="continuous", timesteps=None):
        """
        Initializes the DiTWrapper.

        Args:
            args_dict (dict): A dictionary of arguments for initializing the MoCo model.
        """
        self.args = args_dict
        self.time_type = time_type
        self.timesteps = timesteps
        super().__init__(**args_dict)

    def forward(self, batch, time, conditional_batch=None, timesteps=None, return_features=False):
        """
        Forward pass of the MoCo model.

        Args:
            batch (torch_geometric.data.Batch): The input batch.
            time (Tensor): The time tensor.
            conditional_batch: Optional conditional batch for self-conditioning.
            timesteps: Number of timesteps.
            return_features: Whether to return intermediate features.

        Returns:
            dict: The output of the MoCo model.
        """
        timesteps = timesteps if timesteps is not None else self.timesteps
        if self.time_type == "discrete" and timesteps is not None:
            time = (timesteps - time.float()) / timesteps
        
        # Set the return_features flag on the underlying model
        self.return_features = return_features
        
        out = super().forward(
            batch=batch["batch"],
            X=batch["x_t"],
            H=batch["h_t"],
            E=batch["edge_attr_t"],
            E_idx=batch["edge_index"],
            t=time,
        )
        out["h_logits"] = batch["h_t"]
        out["edge_attr_logits"] = batch["edge_attr_t"]
        return out


class MegaFNV3TSWrapper(MegaFNV3TS):
    """A wrapper class for the MoCo model."""

    def __init__(self, args_dict, time_type="continuous", timesteps=None):
        """
        Initializes the DiTWrapper.

        Args:
            args_dict (dict): A dictionary of arguments for initializing the MoCo model.
        """
        self.args = args_dict
        self.time_type = time_type
        self.timesteps = timesteps
        super().__init__(**args_dict)

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        """
        Forward pass of the MoCo model.

        Args:
            batch (torch_geometric.data.Batch): The input batch.
            time (Tensor): The time tensor.

        Returns:
            dict: The output of the MoCo model.
        """
        timesteps = timesteps if timesteps is not None else self.timesteps
        if self.time_type == "discrete" and timesteps is not None:
            time = (timesteps - time.float()) / timesteps
        x = super().forward(
            batch=batch["batch"],
            X=batch["ts_coord_t"],
            H=batch["numbers_t"],
            ER=batch["bmat_r_t"],
            EP=batch["bmat_p_t"],
            E_idx=batch["edge_index"],
            t=time,
        )
        out = {}
        out["bmat_r_logits"] = batch["bmat_r_t"]
        out["bmat_p_logits"] = batch["bmat_p_t"]
        out["numbers_logits"] = batch["numbers_t"]
        out["ts_coord_hat"] = x
        return out
    

class NextMolConfWrapper(DGTDiffusion):
    def __init__(self, args_dict, time_type="continuous", timesteps=None):
        """
        Initialize the NextMolConformer model.

        Args:
            args_dict (dict): Arguments for initializing the DGTDiffusion model.
            time_type (str, optional): Type of time ('continuous' or 'discrete'). Defaults to "continuous".
            timesteps (int, optional): Number of timesteps for discrete time. Defaults to None.
        """
        self.args = args_dict
        self.time_type = time_type
        self.timesteps = timesteps
        super().__init__(**args_dict)

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        """
        Perform a forward pass of the NextMolConformer model.

        Args:
            batch (torch_geometric.data.Batch): Input batch containing node and edge features.
            time (torch.Tensor): Time tensor for diffusion process.
            conditional_batch (optional): Placeholder for conditional data. Defaults to None.
            timesteps (int, optional): Number of timesteps for discrete time. Defaults to None.

        Returns:
            dict: Model outputs including updated positions and logits for node and edge attributes.
        """
        # Use the provided timesteps or fall back to the instance attribute
        timesteps = timesteps if timesteps is not None else self.timesteps

        # Convert time for discrete diffusion
        if self.time_type == "discrete" and timesteps is not None:
            time = (timesteps - time.float()) / timesteps

        # Construct the data object for the diffusion model
        data = Data(
            x=batch["h_t"],
            pos=batch["x_t"],
            edge_index=batch["edge_index"],
            edge_attr=batch["edge_attr_t"],
            t_cond=time[batch["batch"]],
            batch=batch["batch"]
        )

        # Forward pass through the diffusion model
        pos = super().forward(data)

        # Prepare the output dictionary
        return {
            "x_hat": pos,  # Updated positions
            "h_logits": batch["h_t"],  # Node attribute logits
            "edge_attr_logits": batch["edge_attr_t"]  # Edge attribute logits
        }


class JODOWrapper(DGT_concat):
    """A wrapper class for the JODO model."""

    def __init__(self, args_dict, time_type="continuous", timesteps=None, self_cond=False):
        """
        Initializes the JODOWrapper.

        Args:
            args_dict (dict): A dictionary of arguments for initializing the EQGAT model.
        """
        self.args = args_dict
        self.time_type = time_type
        self.timesteps = timesteps
        self.self_cond = self_cond
        super().__init__(**args_dict)

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        """
        Forward pass of the JODO model.

        Args:
            batch (torch_geometric.data.Batch): The input batch.
            time (Tensor): The time tensor.

        Returns:
            dict: The output of the JODO model.
        """
        if conditional_batch is None:
            conditional_batch = {}

        timesteps = timesteps if timesteps is not None else self.timesteps
        if self.time_type == "discrete" and timesteps is not None:
            time = (timesteps - time.float()) / timesteps
        temb = time.clamp(min=0.001)

        edge_x = to_dense_adj(batch["edge_index"], batch["batch"], batch["edge_attr_t"])
        bs, n, _, _ = edge_x.shape
        _, h_ch = batch["h_t"].shape

        h = torch.zeros((bs, n, h_ch), device=batch["batch"].device)
        x = torch.zeros((bs, n, 3), device=batch["batch"].device)
        node_mask = torch.zeros((bs, n, 1), device=batch["batch"].device)
        edge_mask = torch.zeros((bs, n, n, 1), device=batch["batch"].device)

        n_atoms = scatter_add(torch.ones_like(batch["batch"]), batch["batch"])

        for i, n in enumerate(n_atoms):
            h[i, :n] = batch["h_t"][batch["batch"] == i]
            node_mask[i, :n] = 1
            x[i, :n] = batch["x_t"][batch["batch"] == i]
            edge_mask[i, :n, :n] = 1

        xh = torch.cat([x, h], dim=-1)

        cond_x, cond_edge_x, cond_adj_2d = None, None, None
        if self.self_cond:
            if self.training:
                if random.random() > 0.5:
                    with torch.no_grad():
                        cond_x, cond_edge_x = super().forward(
                            t=temb, xh=xh, node_mask=node_mask, edge_mask=edge_mask, edge_x=edge_x, noise_level=temb
                        )
                        cond_x.detach()
                        cond_edge_x.detach()

            else:
                if "cond_x" in conditional_batch:
                    cond_x = conditional_batch["cond_x"]
                    cond_edge_x = conditional_batch["cond_edge_x"]

            if cond_edge_x is not None:
                with torch.no_grad():
                    dense_index = edge_mask.nonzero(as_tuple=True)
                    cond_adj_2d = cond_edge_x.softmax(dim=-1).argmax(dim=-1, keepdim=True)[dense_index].view(-1, 1)
                    cond_adj_2d[cond_adj_2d != 0] = 1
        out = super().forward(
            t=temb,
            xh=xh,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_x=edge_x,
            noise_level=temb,
            cond_x=cond_x,
            cond_edge_x=cond_edge_x,
            cond_adj_2d=cond_adj_2d,
        )

        x_pred = out[0][..., :3]
        h_pred = out[0][..., 3:]
        edge_attr_pred = out[1]

        if self.self_cond:
            cond_x = out[0].detach()
            cond_edge_x = out[1].detach()

        h = torch.zeros_like(batch["h_t"])
        x = torch.zeros_like(batch["x_t"])
        edge_attr = torch.zeros_like(batch["edge_attr_t"])

        edge_batch = batch["batch"][batch["edge_index"][0]]
        for i, n in enumerate(n_atoms):
            x[batch["batch"] == i] = x_pred[i, :n]
            h[batch["batch"] == i] = h_pred[i, :n]
            A = edge_attr_pred[i, :n, :n]
            edge_index_global = batch["edge_index"][:, edge_batch == i]
            edge_index_local = edge_index_global - edge_index_global.min()
            edge_attr[edge_batch == i] = A[edge_index_local[0], edge_index_local[1]]
        out = {
            "x_hat": x,
            "h_logits": h,
            "edge_attr_logits": edge_attr,
        }

        if self.self_cond:
            out["cond_x"] = cond_x
            out["cond_edge_x"] = cond_edge_x
        return out


class GotenNetWrapper(nn.Module):
    """
    GotenNet wrapper for Megalodon.

    Wraps GoFlow's GotenNet architecture to work with Megalodon's data format.
    Key features:
    1. Filters edges where type is 0 at the beginning (removes no-bond edges)
    2. Uses GotenNet's internal graph augmentation (_extend_condensed_graph_edge)
    3. Uses the exact same Atomwise3DOut layer as GoFlow for velocity prediction

    Edge type encoding (from TSDiff/GoFlow):
        edge_type = r_bond * 22 + p_bond
        where r_bond and p_bond are RDKit bond type indices (0=none, 1=single, 2=double, etc.)
    """

    def __init__(self, args_dict, time_type="continuous", timesteps=None):
        super().__init__()
        self.args = args_dict
        self.time_type = time_type
        self.timesteps = timesteps

        # Lazy import to avoid circular imports and path issues
        from gotennet.models.representation.gotennet import GotenNet
        from gotennet.models.components.ops import CosineCutoff
        from gotennet.models.components.outputs import Atomwise3DOut
        import torch.nn.functional as F

        # Build cutoff function (GoFlow uses CosineCutoff without scaling param)
        cutoff = args_dict.get('cutoff', 5.0)
        cutoff_fn = CosineCutoff(cutoff=cutoff)

        # Build GotenNet with GoFlow-compatible defaults
        self.gotennet = GotenNet(
            n_atom_basis=args_dict.get('n_atom_basis', 64),
            n_atom_rdkit_feats=args_dict.get('n_atom_rdkit_feats', 28),
            n_interactions=args_dict.get('n_interactions', 3),
            radial_basis=args_dict.get('radial_basis', 'expnorm'),
            n_rbf=args_dict.get('n_rbf', 20),
            cutoff_fn=cutoff_fn,
            edge_order=args_dict.get('edge_order', 4),
            activation=args_dict.get('activation', 'swish'),
            max_z=args_dict.get('max_z', 100),
            weight_init=args_dict.get('weight_init', 'xavier_uniform'),
            bias_init=args_dict.get('bias_init', 'zeros'),
            int_layer_norm=args_dict.get('int_layer_norm', ''),
            int_vector_norm=args_dict.get('int_vector_norm', ''),
            num_heads=args_dict.get('num_heads', 8),
            attn_dropout=args_dict.get('attn_dropout', 0.1),
            edge_updates=args_dict.get('edge_updates', 'norej'),
            aggr=args_dict.get('aggr', 'add'),
            edge_ln=args_dict.get('edge_ln', ''),
            sep_int_vec=args_dict.get('sep_int_vec', True),
            lmax=args_dict.get('lmax', 2),
        )

        # Use the same Atomwise3DOut output layer as GoFlow
        # This layer takes (h, X[:, :3, :]) and produces velocity (N, 3)
        hidden_dim = args_dict.get('n_atom_basis', 64)
        n_hidden = args_dict.get('output_n_hidden', 64)
        self.atomwise_3D_out_layer = Atomwise3DOut(
            n_in=hidden_dim,
            n_hidden=n_hidden,
            activation=F.silu
        )

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        """
        Forward pass matching Megalodon interface.

        Args:
            batch: Dict with ts_coord_t, numbers_t, bmat_r_t, bmat_p_t, edge_index, batch
            time: Time tensor (G,)
            conditional_batch: Optional conditional batch (not used)
            timesteps: Number of timesteps for discrete time

        Returns:
            Dict with ts_coord_hat and logits for unchanged variables
        """
        timesteps = timesteps if timesteps is not None else self.timesteps
        if self.time_type == "discrete" and timesteps is not None:
            time = (timesteps - time.float()) / timesteps

        device = batch["ts_coord_t"].device
        pos = batch["ts_coord_t"]
        graph_batch = batch["batch"]

        # Get atom types
        if batch["numbers_t"].dim() > 1:
            atom_type = batch["numbers_t"].argmax(dim=-1)
        else:
            atom_type = batch["numbers_t"]

        # Get bond types from one-hot
        if batch["bmat_r_t"].dim() > 1:
            bmat_r = batch["bmat_r_t"].argmax(dim=-1)
            bmat_p = batch["bmat_p_t"].argmax(dim=-1)
        else:
            bmat_r = batch["bmat_r_t"]
            bmat_p = batch["bmat_p_t"]

        # Encode edge_type as r_bond * 22 + p_bond (TSDiff/GoFlow encoding)
        edge_type = bmat_r * 22 + bmat_p

        # Filter edges to match GoFlow's edge structure:
        # 1. No self-loops (i != j)
        # 2. Only edges with bonds in R or P (edge_type != 0)
        # 3. Only intra-graph edges (same batch)
        edge_index = batch["edge_index"]

        # Self-loops filter
        no_self_loop_mask = edge_index[0] != edge_index[1]

        # Bond filter: keep only edges where there's a bond in R or P
        # edge_type == 0 means r_bond=0 AND p_bond=0 (no bond in either)
        has_bond_mask = edge_type != 0

        # Intra-graph filter
        same_graph_mask = graph_batch[edge_index[0]] == graph_batch[edge_index[1]]

        # Combine all filters
        valid_edge_mask = no_self_loop_mask & has_bond_mask & same_graph_mask

        edge_index_filtered = edge_index[:, valid_edge_mask]
        edge_type_filtered = edge_type[valid_edge_mask]

        # Create r_feat and p_feat from atom features if available
        # Default to zeros if not present
        N = atom_type.size(0)
        n_atom_rdkit_feats = self.args.get('n_atom_rdkit_feats', 28)

        if "r_feat" in batch:
            r_feat = batch["r_feat"]
        else:
            r_feat = torch.zeros(N, n_atom_rdkit_feats, device=device)

        if "p_feat" in batch:
            p_feat = batch["p_feat"]
        else:
            p_feat = torch.zeros(N, n_atom_rdkit_feats, device=device)

        # Create inputs object for GotenNet
        class GotenNetInputs:
            pass

        inputs = GotenNetInputs()
        inputs.edge_index = edge_index_filtered
        inputs.edge_type = edge_type_filtered
        inputs.batch = graph_batch
        inputs.atom_type = atom_type
        inputs.r_feat = r_feat
        inputs.p_feat = p_feat

        # Run GotenNet
        # Returns (h, X) where:
        #   h: scalar node features [N, hidden_dim]
        #   X: equivariant node features [N, L, hidden_dim] where L = (lmax+1)^2 - 1
        h, X = self.gotennet(pos, time, inputs)

        # Use Atomwise3DOut layer (same as GoFlow)
        # Takes scalar features h and vector features X[:, :3, :] (l=1 irreps)
        # Returns velocity (N, 3)
        velocity = self.atomwise_3D_out_layer(h, X[:, :3, :])

        # Output predicted coordinates (velocity for flow matching)
        out = {
            "ts_coord_hat": velocity,
            "bmat_r_logits": batch["bmat_r_t"],
            "bmat_p_logits": batch["bmat_p_t"],
            "numbers_logits": batch["numbers_t"],
        }
        return out
