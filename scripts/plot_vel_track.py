import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = '/home/hz/proprioception/unitree_rl_lab/logs/unitree_g1_29dof_velocity_gru/0520_2256_no_speed/lin_vel_track_dist.csv'
workspace_out_dir = '/home/hz/proprioception/unitree_rl_lab/logs/unitree_g1_29dof_velocity_gru/0520_2256_no_speed'
artifact_out_dir = '/home/hz/.gemini/antigravity-ide/brain/607935ba-02bb-45d6-831d-6a89848daeb2'

# Ensure directories exist
os.makedirs(workspace_out_dir, exist_ok=True)
os.makedirs(artifact_out_dir, exist_ok=True)

# Load data
df = pd.read_csv(csv_path)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Colors
colors = {
    'cmd_speed_raw': '#2c3e50',   # Dark Slate Blue
    'cmd_speed_apf': '#e67e22',   # Vibrant Orange
    'actual_speed': '#27ae60',    # Emerald Green
    'speed_error': '#c0392b'      # Crimson Red
}

iterations = sorted(df['iteration'].unique())

for iteration in iterations:
    # Sort environments by commanded speed
    iter_df = df[df['iteration'] == iteration].sort_values('cmd_speed_raw').reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    
    x = iter_df.index
    
    # 1. Plot commanded speeds
    ax.plot(x, iter_df['cmd_speed_raw'], color=colors['cmd_speed_raw'], linewidth=2.0, label='cmd_speed_raw')
    ax.plot(x, iter_df['cmd_speed_apf'], color=colors['cmd_speed_apf'], linewidth=1.5, linestyle='--', label='cmd_speed_apf')
    
    # Calculate rolling statistics
    window_size = 100
    smoothed_actual = iter_df['actual_speed'].rolling(window=window_size, min_periods=1, center=True).mean()
    std_actual = iter_df['actual_speed'].rolling(window=window_size, min_periods=1, center=True).std()
    
    smoothed_error = iter_df['speed_error'].rolling(window=window_size, min_periods=1, center=True).mean()
    std_error = iter_df['speed_error'].rolling(window=window_size, min_periods=1, center=True).std()
    
    # 2. Plot actual speed: Clearer scatter points + Shaded standard deviation area + Smooth line
    ax.scatter(x, iter_df['actual_speed'], color=colors['actual_speed'], s=3.5, alpha=0.28, label='actual_speed (raw)', edgecolors='none', zorder=2)
    ax.fill_between(x, smoothed_actual - std_actual, smoothed_actual + std_actual, color=colors['actual_speed'], alpha=0.15, zorder=1)
    ax.plot(x, smoothed_actual, color=colors['actual_speed'], linewidth=2.5, alpha=0.95, label='actual_speed (smooth)', zorder=3)
    
    # 3. Plot speed error: Clearer scatter points + Shaded standard deviation area + Smooth line
    ax.scatter(x, iter_df['speed_error'], color=colors['speed_error'], s=3.5, alpha=0.28, label='speed_error (raw)', edgecolors='none', zorder=2)
    ax.fill_between(x, smoothed_error - std_error, smoothed_error + std_error, color=colors['speed_error'], alpha=0.15, zorder=1)
    ax.plot(x, smoothed_error, color=colors['speed_error'], linewidth=2.5, alpha=0.95, label='speed_error (smooth)', zorder=3)
    
    # Title & Labels
    ax.set_title(f'Linear Velocity Tracking (Sorted by cmd_speed_raw) - Iteration {iteration}', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Environments (sorted by target speed)', fontsize=12, labelpad=8)
    ax.set_ylabel('Velocity / Error (m/s)', fontsize=12, labelpad=8)
    
    # Grid customization
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_keys = ['cmd_speed_raw', 'cmd_speed_apf', 'actual_speed (smooth)', 'speed_error (smooth)']
    legend_handles = [by_label[k] for k in legend_keys if k in by_label]
    legend_labels = ['Cmd Speed (Raw)', 'Cmd Speed (APF)', 'Actual Speed (Smooth)', 'Tracking Error (Smooth)']
    
    ax.legend(legend_handles, legend_labels, loc='upper left', frameon=True, facecolor='white', edgecolor='#e0e0e0', framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    
    filename = f'lin_vel_track_iter_{iteration}.png'
    
    # Save to workspace
    workspace_path = os.path.join(workspace_out_dir, filename)
    plt.savefig(workspace_path, dpi=150, bbox_inches='tight')
    
    # Save to artifact directory
    artifact_path = os.path.join(artifact_out_dir, filename)
    plt.savefig(artifact_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    print(f"Generated enhanced plot for Iteration {iteration}")
