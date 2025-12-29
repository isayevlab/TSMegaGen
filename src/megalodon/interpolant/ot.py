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

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def scale_prior(prior, batch, num_atoms, c=0.2):
    return c * prior * np.log(num_atoms + 1)[batch]


def align_prior(prior_feat: torch.Tensor, dst_feat: torch.Tensor, permutation=False,
        rigid_body=False, n_alignments: int = 1):
    """
    Aligns a prior feature to a destination feature.
    """
    for _ in range(n_alignments):
        if permutation:
            # solve assignment problem
            cost_mat = torch.cdist(dst_feat, prior_feat, p=2).cpu().detach().numpy()
            _, prior_idx = linear_sum_assignment(cost_mat)

            # reorder prior to according to optimal assignment
            prior_feat = prior_feat[prior_idx]

        if rigid_body:
            # perform rigid alignment
            prior_feat = rigid_alignment(prior_feat, dst_feat)

    return prior_feat


def rigid_alignment(x_0, x_1, pre_centered=False):
    """
    Align x_0 to x_1 using the Kabsch algorithm.
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm

    Finds optimal rotation R that minimizes ||x_0 @ R - x_1||.
    Returns x_0 rotated and translated to match x_1.
    """
    d = x_0.shape[1]
    assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape"

    # Center both point clouds
    if pre_centered:
        x_0_mean = torch.zeros(1, d, device=x_0.device)
        x_1_mean = torch.zeros(1, d, device=x_0.device)
        x_0_c = x_0
        x_1_c = x_1
    else:
        x_0_mean = x_0.mean(dim=0, keepdim=True)
        x_1_mean = x_1.mean(dim=0, keepdim=True)
        x_0_c = x_0 - x_0_mean
        x_1_c = x_1 - x_1_mean

    # Covariance matrix H = x_0^T @ x_1
    H = x_0_c.T.mm(x_1_c)
    U, S, V = torch.svd(H)

    # Handle reflection: ensure proper rotation (det(R) = 1)
    d_sign = torch.det(V.mm(U.T))
    D = torch.eye(d, device=x_0.device)
    D[-1, -1] = d_sign.sign()

    # Optimal rotation: R = V @ D @ U^T
    R = V.mm(D).mm(U.T)

    # Apply rotation to centered x_0, then translate to x_1's centroid
    x_0_aligned = x_0_c.mm(R.T) + x_1_mean

    return x_0_aligned
