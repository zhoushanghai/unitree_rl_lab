#!/usr/bin/env python3
"""Convert proprioception/collision datasets from NPZ format back to JSON format.

Usage:
    # Convert a single file
    python scripts/rsl_rl/npz_to_json.py --input dataset/episode_00001.npz
    
    # Convert all NPZ files in a folder
    python scripts/rsl_rl/npz_to_json.py --input dataset/
    
    # Convert and delete original NPZ files
    python scripts/rsl_rl/npz_to_json.py --input dataset/ --delete
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm


def convert_single_file(npz_path: str, json_path: str, delete_source: bool = False):
    """Converts a single .npz file back to the original .json format."""
    if not os.path.exists(npz_path):
        print(f"[Error] File not found: {npz_path}")
        return False

    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # Check for required collisions field
        if "collisions_json" not in data:
            print(f"[Warning] Skipped {npz_path}: 'collisions_json' not found in file.")
            return False

        # Safely extract the JSON string of collisions
        collisions_field = data["collisions_json"]
        if hasattr(collisions_field, "item"):
            try:
                collisions_str = collisions_field.item()
            except ValueError:
                collisions_str = str(collisions_field)
        else:
            collisions_str = str(collisions_field)
            
        collisions = json.loads(collisions_str)

        # Standard keys present in the original dataset
        keys = [
            "time", "command", "base_ang_vel", "projected_gravity",
            "joint_pos", "joint_vel", "last_action", "root_pos_w",
            "root_quat_w", "base_lin_vel", "joint_torques"
        ]

        num_steps = len(data["time"])
        episode_data = []

        for step_idx in range(num_steps):
            step_data = {}
            for key in keys:
                if key in data:
                    val = data[key][step_idx]
                    # Convert numpy array/types back to python standard list/float
                    if isinstance(val, np.ndarray):
                        step_data[key] = val.tolist()
                    elif hasattr(val, "item"):
                        step_data[key] = val.item()
                    else:
                        step_data[key] = float(val)

            # Insert the step collisions list
            step_data["collisions"] = collisions[step_idx]
            episode_data.append(step_data)

        # Save to JSON file
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(episode_data, f, indent=2)

        if delete_source:
            os.remove(npz_path)
            
        return True

    except Exception as e:
        print(f"[Error] Failed to convert {npz_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert proprioception dataset from NPZ to JSON.")
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        default="dataset", 
        help="Path to an .npz file or a directory containing .npz files (default: dataset/)."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default=None, 
        help="Path to output .json file or directory (default: same as input)."
    )
    parser.add_argument(
        "-d", "--delete", 
        action="store_true", 
        help="Delete the original .npz files after successful conversion."
    )
    args = parser.parse_args()

    # Determine input type
    if os.path.isfile(args.input):
        # Single file conversion
        npz_path = args.input
        if args.output:
            json_path = args.output if args.output.endswith(".json") else os.path.join(args.output, os.path.basename(npz_path).replace(".npz", ".json"))
        else:
            json_path = npz_path.replace(".npz", ".json")

        print(f"[INFO] Converting {npz_path} -> {json_path} ...")
        success = convert_single_file(npz_path, json_path, args.delete)
        if success:
            print("[INFO] Conversion completed successfully.")
        else:
            print("[Error] Conversion failed.")

    elif os.path.isdir(args.input):
        # Directory conversion
        npz_files = [f for f in os.listdir(args.input) if f.endswith(".npz")]
        if not npz_files:
            print(f"[INFO] No .npz files found in directory: {args.input}")
            return

        out_dir = args.output if args.output else args.input
        os.makedirs(out_dir, exist_ok=True)

        print(f"[INFO] Found {len(npz_files)} .npz files in {args.input}. Converting to {out_dir} ...")
        
        success_count = 0
        for npz_file in tqdm(npz_files, desc="Converting"):
            npz_path = os.path.join(args.input, npz_file)
            json_path = os.path.join(out_dir, npz_file.replace(".npz", ".json"))
            if convert_single_file(npz_path, json_path, args.delete):
                success_count += 1

        print(f"[INFO] Conversion finished. Successfully converted {success_count}/{len(npz_files)} files.")
    else:
        print(f"[Error] Input path '{args.input}' does not exist.")


if __name__ == "__main__":
    main()
