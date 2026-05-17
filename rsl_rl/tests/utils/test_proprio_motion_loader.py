# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import pytest
import yaml

from rsl_rl.utils.proprio_motion_loader import ProprioceptionMotionExpert


def test_proprioception_motion_expert_smoke(tmp_path) -> None:
    ds = tmp_path / "datasets"
    ds.mkdir(parents=True)
    t_len, n_dof, n_body = 11, 5, 3
    dof_names = [f"j{i}" for i in range(n_dof)]
    body_names = [f"b{i}" for i in range(n_body)]

    np.savez(
        ds / "clip0.npz",
        dof_names=np.asarray(dof_names, dtype=object),
        key_body_names=np.asarray(body_names, dtype=object),
        fps=np.float32(50.0),
        dof_positions=np.zeros((t_len, n_dof), dtype=np.float32),
        dof_velocities=np.zeros((t_len, n_dof), dtype=np.float32),
        key_body_positions_local=np.zeros((t_len, n_body * 3), dtype=np.float32),
    )

    manifest = tmp_path / "motions.yaml"
    with open(manifest, "w") as f:
        yaml.dump({"motion_files": [{"file": "datasets/clip0.npz", "weight": 1.0}]}, f)

    feat_dim = n_dof * 2 + n_body * 3
    expert = ProprioceptionMotionExpert(
        motion_yaml_path=str(manifest),
        amp_obs_dim=feat_dim,
        device="cpu",
        motion_project_root=str(tmp_path),
    )

    s0, s1 = next(expert.feed_forward_generator(1, 8))
    assert s0.shape == (8, feat_dim)
    assert s1.shape == (8, feat_dim)


def test_dof_subsets_match_dim(tmp_path) -> None:
    ds = tmp_path / "datasets"
    ds.mkdir(parents=True)
    t_len, n_dof, n_body = 5, 6, 2
    dof_names = [f"d{i}" for i in range(n_dof)]
    body_names = [f"k{i}" for i in range(n_body)]
    np.savez(
        ds / "c.npz",
        dof_names=np.asarray(dof_names, dtype=object),
        key_body_names=np.asarray(body_names, dtype=object),
        fps=np.float32(20.0),
        dof_positions=np.random.randn(t_len, n_dof).astype(np.float32),
        dof_velocities=np.random.randn(t_len, n_dof).astype(np.float32),
        key_body_positions_local=np.random.randn(t_len, n_body, 3).astype(np.float32),
    )
    manifest = tmp_path / "m.yaml"
    with open(manifest, "w") as f:
        yaml.dump({"motion_files": [{"file": "datasets/c.npz", "weight": 2.0}]}, f)

    k = 4
    dof_idx = [0, 1, 2, 3]
    kb_idx = [1]
    amp_dim = len(dof_idx) * 2 + len(kb_idx) * 3
    expert = ProprioceptionMotionExpert(
        motion_yaml_path=str(manifest),
        amp_obs_dim=amp_dim,
        device="cpu",
        motion_dof_indexes=dof_idx,
        motion_key_body_indexes=kb_idx,
        motion_project_root=str(tmp_path),
    )
    a, b = next(expert.feed_forward_generator(1, 2))
    assert a.shape[1] == amp_dim
    assert pytest.approx(float(expert.per_frame_weights.sum().cpu())) == 1.0
