#!/usr/bin/env python3
"""可视化 dataset_voxel 中的 collision_voxel（无需 Isaac Sim）。"""

import argparse
import json
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from voxel_collision_utils import GRID_NX, GRID_NY, GRID_NZ, VOXEL_RES_M, LOCAL_XY_MIN, Z_MIN


def _voxel_centers_m() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """体素中心在局部系下的坐标 (x前, y左, z上)，单位 m。"""
    xs = LOCAL_XY_MIN + (np.arange(GRID_NX) + 0.5) * VOXEL_RES_M
    ys = LOCAL_XY_MIN + (np.arange(GRID_NY) + 0.5) * VOXEL_RES_M
    zs = Z_MIN + (np.arange(GRID_NZ) + 0.5) * VOXEL_RES_M
    return xs, ys, zs


def _steps_with_voxels(voxels: np.ndarray) -> list[int]:
    flat = voxels.reshape(voxels.shape[0], -1)
    return [int(t) for t in range(voxels.shape[0]) if flat[t].any()]


def _pick_default_step(voxels: np.ndarray, collisions_json: list, prefer: int | None) -> int:
    if prefer is not None:
        return int(np.clip(prefer, 0, voxels.shape[0] - 1))
    # 优先：有体素且当步有碰撞记录
    for t in range(voxels.shape[0]):
        if voxels[t].any() and t < len(collisions_json) and len(collisions_json[t]) > 0:
            return t
    occ = _steps_with_voxels(voxels)
    return occ[0] if occ else 0


def visualize_episode(
    npz_path: str,
    step: int | None,
    output: str | None,
    show: bool,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    voxels = data["collision_voxel"]
    collisions = json.loads(str(data["collisions_json"].item()))
    t = _pick_default_step(voxels, collisions, step)

    vol = voxels[t]
    n_on = int(vol.sum())
    cmd = data["command"][0] if data["command"].ndim == 2 else data["command"]
    cmd_xy = float(np.hypot(cmd[0], cmd[1]))
    sim_t = float(data["time"][t])

    # 三向最大投影，便于看 3D 占用分布
    xy = vol.max(axis=2)  # 俯视 x-前 / y-左
    xz = vol.max(axis=1)  # 侧视 x-前 / z-上
    yz = vol.max(axis=0)  # 侧视 y-左 / z-上

    import matplotlib.pyplot as plt

    xs, ys, zs = _voxel_centers_m()
    extent_xy = [xs[0] - 0.05, xs[-1] + 0.05, ys[0] - 0.05, ys[-1] + 0.05]
    extent_xz = [xs[0] - 0.05, xs[-1] + 0.05, zs[0] - 0.05, zs[-1] + 0.05]
    extent_yz = [ys[0] - 0.05, ys[-1] + 0.05, zs[0] - 0.05, zs[-1] + 0.05]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(
        f"{os.path.basename(npz_path)} | step={t} | t_sim={sim_t:.2f}s | "
        f"voxels_on={n_on} | cmd_xy={cmd_xy:.3f} m/s",
        fontsize=11,
    )

    im0 = axes[0, 0].imshow(xy.T, origin="lower", extent=extent_xy, cmap="Reds", vmin=0, vmax=1, aspect="equal")
    axes[0, 0].set_title("XY top (max over z)")
    axes[0, 0].set_xlabel("x forward (m)")
    axes[0, 0].set_ylabel("y left (m)")
    axes[0, 0].plot(0, 0, "b+", markersize=12, label="robot center")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[0, 1].imshow(xz.T, origin="lower", extent=extent_xz, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    axes[0, 1].set_title("XZ side (max over y)")
    axes[0, 1].set_xlabel("x forward (m)")
    axes[0, 1].set_ylabel("z up (m)")

    axes[1, 0].imshow(yz.T, origin="lower", extent=extent_yz, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    axes[1, 0].set_title("YZ side (max over x)")
    axes[1, 0].set_xlabel("y left (m)")
    axes[1, 0].set_ylabel("z up (m)")

    # 3D 占用点（体素中心）
    ax3d = fig.add_subplot(2, 2, 4, projection="3d")
    ii, jj, kk = np.where(vol > 0)
    if ii.size > 0:
        ax3d.scatter(xs[ii], ys[jj], zs[kk], c="crimson", s=18, alpha=0.85, depthshade=True)
    ax3d.set_xlabel("x forward")
    ax3d.set_ylabel("y left")
    ax3d.set_zlabel("z up")
    ax3d.set_title(f"3D occupied ({n_on} cells)")
    ax3d.set_xlim(xs[0], xs[-1])
    ax3d.set_ylim(ys[0], ys[-1])
    ax3d.set_zlim(zs[0], zs[-1])

    # 当步碰撞点（世界系，仅作数量提示）
    n_col = len(collisions[t]) if t < len(collisions) else 0
    fig.text(
        0.5,
        0.02,
        f"collisions@{t}: {n_col} records | steps_with_voxel: {len(_steps_with_voxels(voxels))}/{voxels.shape[0]}",
        ha="center",
        fontsize=9,
    )

    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"[INFO] Saved → {output}")
    if show or not output:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize collision_voxel in dataset_voxel NPZ.")
    parser.add_argument("--file", type=str, required=True, help="例如 dataset_voxel/episode_00002.npz")
    parser.add_argument("--step", type=int, default=None, help="时间步索引；默认选有体素且有碰撞的步")
    parser.add_argument("--output", type=str, default=None, help="保存 PNG 路径（不填则弹窗）")
    parser.add_argument("--list-steps", action="store_true", help="列出所有有占用的步号后退出")
    parser.add_argument("--no-show", action="store_true", help="与 --output 合用，不弹窗")
    args = parser.parse_args()

    voxels = np.load(args.file)["collision_voxel"]
    if args.list_steps:
        steps = _steps_with_voxels(voxels)
        print(f"[INFO] {args.file}: {len(steps)} steps with voxel occupancy")
        print(steps[:50], "..." if len(steps) > 50 else "")
        return

    visualize_episode(args.file, args.step, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
