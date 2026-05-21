import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Plot linear velocity tracking and reward distribution for curriculum upgrades.")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to curriculum_upgrade_vel_track.csv')
    parser.add_argument('--workspace_out_dir', type=str, required=True, help='Workspace directory to save output plots.')
    parser.add_argument('--artifact_out_dir', type=str, default=None, help='Artifact directory to save duplicate plots.')
    
    args = parser.parse_args()
    
    csv_path = args.csv_path
    workspace_out_dir = args.workspace_out_dir
    artifact_out_dir = args.artifact_out_dir
    
    os.makedirs(workspace_out_dir, exist_ok=True)
    if artifact_out_dir:
        os.makedirs(artifact_out_dir, exist_ok=True)
        
    # Load data
    df = pd.read_csv(csv_path)
    
    # Set plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = {
        'cmd_speed_raw': '#2c3e50',      # Dark Slate Blue
        'cmd_speed_apf': '#e67e22',      # Vibrant Orange
        'actual_speed': '#27ae60',       # Emerald Green
        'speed_error': '#c0392b',         # Crimson Red
        'tracking_reward': '#8e44ad'     # Deep Purple
    }
    
    iterations = sorted(df['iteration'].unique())
    
    for iteration in iterations:
        # Filter and sort environments by commanded speed
        iter_df = df[df['iteration'] == iteration].sort_values('cmd_speed_raw').reset_index(drop=True)
        
        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
        x = iter_df.index
        
        # Since we have very few data points (5-8 environments), we draw lines and clear markers instead of smoothed rolling statistics
        # Plot commanded speeds on left axis (ax1)
        ax1.plot(x, iter_df['cmd_speed_raw'], color=colors['cmd_speed_raw'], marker='o', linewidth=2.0, label='Cmd Speed (Raw)')
        ax1.plot(x, iter_df['cmd_speed_apf'], color=colors['cmd_speed_apf'], marker='x', linewidth=1.5, linestyle='--', label='Cmd Speed (APF)')
        
        # Plot actual speed and speed error
        ax1.plot(x, iter_df['actual_speed'], color=colors['actual_speed'], marker='s', linewidth=2.0, label='Actual Speed')
        ax1.plot(x, iter_df['speed_error'], color=colors['speed_error'], marker='d', linewidth=2.0, label='Speed Error')
        
        # Setup right axis (ax2) for tracking reward (range 0 to 1)
        ax2 = ax1.twinx()
        ax2.plot(x, iter_df['tracking_reward'], color=colors['tracking_reward'], marker='^', linewidth=2.0, label='Tracking Reward')
        
        # Format X ticks to show Env IDs
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Env {env}" for env in iter_df['env_id']], rotation=30)
        
        # Title, Labels and Ranges
        ax1.set_title(f'Curriculum Upgrade Velocity Tracking & Reward - Iteration {iteration}', fontsize=12, fontweight='bold', pad=15)
        ax1.set_xlabel('Upgrading Environments', fontsize=10, labelpad=8)
        ax1.set_ylabel('Velocity / Error (m/s)', fontsize=10, labelpad=8)
        ax2.set_ylabel('Tracking Reward [0.0, 1.0]', color=colors['tracking_reward'], fontsize=10, labelpad=8)
        
        # Range bounds info text
        x_min_val = iter_df['new_range_x_min'].iloc[0]
        x_max_val = iter_df['new_range_x_max'].iloc[0]
        y_min_val = iter_df['new_range_y_min'].iloc[0]
        y_max_val = iter_df['new_range_y_max'].iloc[0]
        
        info_text = f"New Range X: [{x_min_val:.1f}, {x_max_val:.1f}]\nNew Range Y: [{y_min_val:.1f}, {y_max_val:.1f}]"
        ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#e0e0e0'))
        
        # Y limits
        max_vel = max(iter_df['cmd_speed_raw'].max(), iter_df['actual_speed'].max(), 0.5)
        ax1.set_ylim(-0.05, max_vel * 1.3)
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis='y', labelcolor=colors['tracking_reward'])
        
        # Grid customization
        ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax2.grid(False)
        
        # Merge legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper right', frameon=True, facecolor='white', edgecolor='#e0e0e0', framealpha=0.9, fontsize=9)
        
        plt.tight_layout()
        filename = f'curriculum_vel_track_iter_{iteration}.png'
        
        # Save to workspace
        workspace_path = os.path.join(workspace_out_dir, filename)
        plt.savefig(workspace_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to workspace: {workspace_path}")
        
        # Save to artifact directory
        if artifact_out_dir:
            artifact_path = os.path.join(artifact_out_dir, filename)
            plt.savefig(artifact_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to artifact: {artifact_path}")
            
        plt.close()

if __name__ == '__main__':
    main()
