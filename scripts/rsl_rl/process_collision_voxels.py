#!/usr/bin/env python3
"""离线将 dataset/ 碰撞点转为局部体素，写入 dataset_voxel/（见 docs/collision_voxel_design.md）。"""

import argparse
import json
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from voxel_collision_utils import (
    DEFAULT_MIN_CMD_SPEED,
    build_collision_voxel_sequence,
    command_xy_speed,
)


def _list_episodes(input_dir: str) -> list[str]:
    paths = []
    for name in sorted(os.listdir(input_dir)):
        if name.startswith("episode_") and name.endswith(".npz"):
            paths.append(os.path.join(input_dir, name))
    return paths


def process_one_episode(
    src_path: str,
    dst_path: str,
    min_cmd_speed: float,
) -> dict:
    """拷贝原 NPZ 并追加 collision_voxel；线速度不足则跳过、不写输出文件。"""
    data = dict(np.load(src_path, allow_pickle=True))
    cmd_speed = command_xy_speed(data["command"][0])

    # 整条 episode 的 command 线速度 < 阈值：直接丢弃，不进入 dataset_voxel
    if cmd_speed < min_cmd_speed:
        return {
            "skipped": True,
            "cmd_speed": cmd_speed,
            "steps": int(data["root_pos_w"].shape[0]),
        }

    collisions = json.loads(str(data["collisions_json"].item()))
    voxels = build_collision_voxel_sequence(
        collisions,
        data["root_pos_w"],
        data["root_quat_w"],
    )
    data["collision_voxel"] = voxels

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    np.savez_compressed(dst_path, **data)

    occupied_steps = int(np.any(voxels.reshape(voxels.shape[0], -1), axis=1).sum())
    return {
        "skipped": False,
        "cmd_speed": cmd_speed,
        "steps": voxels.shape[0],
        "occupied_steps": occupied_steps,
        "total_voxels_on": int(voxels.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Build dataset_voxel from dataset NPZ files.")
    parser.add_argument("--input", type=str, default="dataset", help="源目录（episode_*.npz）")
    parser.add_argument("--output", type=str, default="dataset_voxel", help="输出目录")
    parser.add_argument(
        "--min-cmd-speed",
        type=float,
        default=DEFAULT_MIN_CMD_SPEED,
        help="command 线速度 sqrt(vx^2+vy^2) 低于此值的整条 episode 不写入 dataset_voxel",
    )
    args = parser.parse_args()

    episodes = _list_episodes(args.input)
    if not episodes:
        print(f"[WARN] No episode_*.npz under {args.input}")
        return

    print(f"[INFO] Processing {len(episodes)} episodes: {args.input} -> {args.output}")
    n_ok, n_skip = 0, 0
    for src in episodes:
        name = os.path.basename(src)
        dst = os.path.join(args.output, name)
        stats = process_one_episode(src, dst, args.min_cmd_speed)
        if stats["skipped"]:
            n_skip += 1
            print(
                f"[INFO] {name} | SKIP(low cmd, not saved) | cmd_xy={stats['cmd_speed']:.3f} | "
                f"steps={stats['steps']}"
            )
            continue
        n_ok += 1
        print(
            f"[INFO] {name} | OK | cmd_xy={stats['cmd_speed']:.3f} | "
            f"steps={stats['steps']} | steps_with_voxel={stats['occupied_steps']} | "
            f"sum(voxel)={stats['total_voxels_on']}"
        )
    print(f"[INFO] Done. saved={n_ok} skipped={n_skip} -> {args.output}/")


if __name__ == "__main__":
    main()
