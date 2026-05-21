import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description="Plot linear velocity tracking and reward distribution.")
parser.add_argument('--csv_path', type=str, default='/home/hz/proprioception/unitree_rl_lab/logs/unitree_g1_29dof_velocity_gru/0520_2256_no_speed/lin_vel_track_dist.csv', help='Path to the tracking distribution CSV file.')
parser.add_argument('--workspace_out_dir', type=str, default='/home/hz/proprioception/unitree_rl_lab/logs/unitree_g1_29dof_velocity_gru/0520_2256_no_speed', help='Workspace directory to save output plots.')
parser.add_argument('--artifact_out_dir', type=str, default='/home/hz/.gemini/antigravity-ide/brain/607935ba-02bb-45d6-831d-6a89848daeb2', help='Artifact directory to save duplicate plots.')

args = parser.parse_args()
csv_path = args.csv_path
workspace_out_dir = args.workspace_out_dir
artifact_out_dir = args.artifact_out_dir

# Ensure directories exist
os.makedirs(workspace_out_dir, exist_ok=True)
os.makedirs(artifact_out_dir, exist_ok=True)

# Load data
df = pd.read_csv(csv_path)

# Calculate linear velocity tracking reward: r = exp(-error^2 / std^2) where std^2 = 0.25 (std = sqrt(0.25) = 0.5)
# Formula from track_lin_vel_xy_yaw_frame_exp_apf in rewards.py
df['tracking_reward'] = np.exp(- (df['speed_error'] ** 2) / 0.25)

# Save back the CSV with the computed reward column so the user has it
df.to_csv(csv_path, index=False)
print("Saved computed tracking_reward column back to CSV.")

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Colors
colors = {
    'cmd_speed_raw': '#2c3e50',      # Dark Slate Blue
    'cmd_speed_apf': '#e67e22',      # Vibrant Orange
    'actual_speed': '#27ae60',       # Emerald Green
    'speed_error': '#c0392b',         # Crimson Red
    'tracking_reward': '#8e44ad'     # Deep Purple
}

iterations = sorted(df['iteration'].unique())

for iteration in iterations:
    # Sort environments by commanded speed
    iter_df = df[df['iteration'] == iteration].sort_values('cmd_speed_raw').reset_index(drop=True)
    
    fig, ax1 = plt.subplots(figsize=(15, 6), dpi=150)
    
    x = iter_df.index
    
    # 1. Plot commanded speeds on left axis (ax1)
    ax1.plot(x, iter_df['cmd_speed_raw'], color=colors['cmd_speed_raw'], linewidth=2.0, label='cmd_speed_raw')
    ax1.plot(x, iter_df['cmd_speed_apf'], color=colors['cmd_speed_apf'], linewidth=1.5, linestyle='--', label='cmd_speed_apf')
    
    # Calculate rolling statistics for speed and error
    window_size = 100
    smoothed_actual = iter_df['actual_speed'].rolling(window=window_size, min_periods=1, center=True).mean()
    std_actual = iter_df['actual_speed'].rolling(window=window_size, min_periods=1, center=True).std()
    
    smoothed_error = iter_df['speed_error'].rolling(window=window_size, min_periods=1, center=True).mean()
    std_error = iter_df['speed_error'].rolling(window=window_size, min_periods=1, center=True).std()
    
    # Plot actual speed: Scatter points + Shaded standard deviation area + Smooth line
    ax1.scatter(x, iter_df['actual_speed'], color=colors['actual_speed'], s=3.0, alpha=0.25, label='actual_speed (raw)', edgecolors='none', zorder=2)
    ax1.fill_between(x, smoothed_actual - std_actual, smoothed_actual + std_actual, color=colors['actual_speed'], alpha=0.12, zorder=1)
    ax1.plot(x, smoothed_actual, color=colors['actual_speed'], linewidth=2.2, alpha=0.95, label='actual_speed (smooth)', zorder=3)
    
    # Plot speed error: Scatter points + Shaded standard deviation area + Smooth line
    ax1.scatter(x, iter_df['speed_error'], color=colors['speed_error'], s=3.0, alpha=0.25, label='speed_error (raw)', edgecolors='none', zorder=2)
    ax1.fill_between(x, smoothed_error - std_error, smoothed_error + std_error, color=colors['speed_error'], alpha=0.12, zorder=1)
    ax1.plot(x, smoothed_error, color=colors['speed_error'], linewidth=2.2, alpha=0.95, label='speed_error (smooth)', zorder=3)
    
    # 2. Setup right axis (ax2) for tracking reward (range 0 to 1)
    ax2 = ax1.twinx()
    
    # Calculate rolling statistics for reward
    smoothed_reward = iter_df['tracking_reward'].rolling(window=window_size, min_periods=1, center=True).mean()
    std_reward = iter_df['tracking_reward'].rolling(window=window_size, min_periods=1, center=True).std()
    
    # Plot tracking reward: Scatter points (smaller) + Shaded std dev + Smooth line
    ax2.scatter(x, iter_df['tracking_reward'], color=colors['tracking_reward'], s=1.5, alpha=0.15, edgecolors='none', zorder=1)
    ax2.fill_between(x, (smoothed_reward - std_reward).clip(0, 1), (smoothed_reward + std_reward).clip(0, 1), color=colors['tracking_reward'], alpha=0.08, zorder=0)
    ax2.plot(x, smoothed_reward, color=colors['tracking_reward'], linewidth=2.5, alpha=0.95, label='tracking_reward (smooth)', zorder=4)
    
    # Title & Labels
    ax1.set_title(f'Linear Velocity Tracking & Reward - Iteration {iteration}', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel('Environments (sorted by target speed)', fontsize=12, labelpad=8)
    ax1.set_ylabel('Velocity / Error (m/s)', fontsize=12, labelpad=8)
    ax2.set_ylabel('Tracking Reward [0.0, 1.0]', color=colors['tracking_reward'], fontsize=12, labelpad=8)
    
    # Adjust axes limits
    ax1.set_ylim(-0.1, 1.6)
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis='y', labelcolor=colors['tracking_reward'])
    
    # Grid customization (only on left axis to avoid double lines)
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax2.grid(False)
    
    # Legend
    # Merge legends from ax1 and ax2
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    by_label = dict(zip(l1 + l2, h1 + h2))
    legend_keys = ['cmd_speed_raw', 'cmd_speed_apf', 'actual_speed (smooth)', 'speed_error (smooth)', 'tracking_reward (smooth)']
    legend_handles = [by_label[k] for k in legend_keys if k in by_label]
    legend_labels = ['Cmd Speed (Raw)', 'Cmd Speed (APF)', 'Actual Speed (Smooth)', 'Tracking Error (Smooth)', 'Tracking Reward (Smooth)']
    
    ax1.legend(legend_handles, legend_labels, loc='upper left', frameon=True, facecolor='white', edgecolor='#e0e0e0', framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    
    filename = f'lin_vel_track_iter_{iteration}.png'
    
    # Save to workspace
    workspace_path = os.path.join(workspace_out_dir, filename)
    plt.savefig(workspace_path, dpi=150, bbox_inches='tight')
    
    # Save to artifact directory
    artifact_path = os.path.join(artifact_out_dir, filename)
    plt.savefig(artifact_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    print(f"Generated plot with reward for Iteration {iteration}")
