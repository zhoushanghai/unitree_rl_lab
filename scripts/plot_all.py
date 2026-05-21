import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Style Setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

colors = {
    'cmd_speed_raw': '#2c3e50',      # Dark Slate Blue
    'cmd_speed_apf': '#e67e22',      # Vibrant Orange
    'actual_speed': '#27ae60',       # Emerald Green
    'speed_error': '#c0392b',         # Crimson Red
    'tracking_reward': '#8e44ad',     # Deep Purple
    'tb_reward': '#1f77b4',           # Blue for TB mean reward
    'tb_curr': '#ff7f0e'              # Orange for TB curriculum
}

def parse_tfevents(log_dir):
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "events.out.tfevents" in f]
    if not event_files:
        return {}
    
    event_file = event_files[0]
    print(f"-> Loading TensorBoard events from: {event_file}...")
    try:
        ea = event_accumulator.EventAccumulator(event_file, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        
        data = {}
        tags = ea.Tags()['scalars']
        for tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = pd.DataFrame({'step': steps, 'value': values})
        return data
    except Exception as e:
        print(f"Warning: Failed to parse tfevents: {e}")
        return {}

def save_and_copy(fig, filename, workspace_dir, artifact_dir, dpi=150):
    workspace_path = os.path.join(workspace_dir, filename)
    fig.savefig(workspace_path, dpi=dpi, bbox_inches='tight')
    
    if artifact_dir:
        artifact_path = os.path.join(artifact_dir, filename)
        fig.savefig(artifact_path, dpi=dpi, bbox_inches='tight')

def plot_tensorboard_summary(log_dir, workspace_dir, artifact_dir):
    tb_data = parse_tfevents(log_dir)
    if not tb_data:
        print("-> No TensorBoard events found or parsing failed. Skipping TB summary plot.")
        return
        
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Training Progress Summary - {os.path.basename(log_dir)}", y=0.98, weight='bold', fontsize=15)
    
    # Helper to plot scalar
    def plot_helper(ax, tag, title, ylabel, color):
        if tag in tb_data:
            df = tb_data[tag]
            window = max(1, len(df) // 100)
            smoothed = df['value'].rolling(window=window, min_periods=1).mean()
            ax.plot(df['step'], df['value'], color=color, alpha=0.15)
            ax.plot(df['step'], smoothed, color=color, linewidth=2, label='Smoothed')
            ax.set_title(title, weight='semibold')
            ax.set_ylabel(ylabel)
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, f"Tag '{tag}' not found", ha='center', va='center', color='gray')
            ax.set_title(title)

    plot_helper(axs[0, 0], 'Train/mean_reward', "Overall Mean Reward", "Reward", colors['tb_reward'])
    plot_helper(axs[0, 1], 'diag/track_lin/mean_tracking_reward', "Linear Velocity Tracking Reward", "Reward", colors['tracking_reward'])
    plot_helper(axs[1, 0], 'diag/track_lin/mean_speed_error', "Mean Linear Velocity Speed Error", "Error (m/s)", colors['speed_error'])
    
    # Curriculum
    ax_curr = axs[1, 1]
    has_curr = False
    if 'Curriculum/lin_vel_cmd_levels' in tb_data:
        df_vel = tb_data['Curriculum/lin_vel_cmd_levels']
        ax_curr.plot(df_vel['step'], df_vel['value'], color=colors['tb_curr'], linewidth=2.5, label='Vel Cmd Level')
        has_curr = True
    if 'Curriculum/terrain_levels' in tb_data:
        df_terr = tb_data['Curriculum/terrain_levels']
        ax_curr.plot(df_terr['step'], df_terr['value'], color='#9467bd', linewidth=2.5, label='Terrain Level')
        has_curr = True
        
    if has_curr:
        ax_curr.set_title("Curriculum Progression", weight='semibold')
        ax_curr.set_ylabel("Curriculum Level")
        ax_curr.legend(loc='upper left')
        ax_curr.grid(True)
    else:
        ax_curr.text(0.5, 0.5, "No curriculum tags found", ha='center', va='center', color='gray')
        ax_curr.set_title("Curriculum Progression")
        
    for ax in axs.flat:
        ax.set_xlabel("Iteration")
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_and_copy(fig, "training_metrics_summary.png", workspace_dir, artifact_dir, dpi=200)
    plt.close(fig)
    print("-> Generated training_metrics_summary.png")

def plot_vel_tracking_dist(csv_path, workspace_dir, artifact_dir):
    if not os.path.exists(csv_path):
        print(f"-> {os.path.basename(csv_path)} not found. Skipping velocity tracking distribution plots.")
        return
        
    print(f"-> Loading tracking distribution from: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Compute reward if not present
    if 'tracking_reward' not in df.columns:
        df['tracking_reward'] = np.exp(- (df['speed_error'] ** 2) / 0.25)
        df.to_csv(csv_path, index=False)
        
    iterations = sorted(df['iteration'].unique())
    
    # 1. Per-iteration tracking curves
    for iteration in iterations:
        iter_df = df[df['iteration'] == iteration].sort_values('cmd_speed_raw').reset_index(drop=True)
        
        fig, ax1 = plt.subplots(figsize=(15, 6), dpi=150)
        x = iter_df.index
        
        ax1.plot(x, iter_df['cmd_speed_raw'], color=colors['cmd_speed_raw'], linewidth=2.0, label='cmd_speed_raw')
        ax1.plot(x, iter_df['cmd_speed_apf'], color=colors['cmd_speed_apf'], linewidth=1.5, linestyle='--', label='cmd_speed_apf')
        
        window_size = min(100, len(iter_df) // 4)
        smoothed_actual = iter_df['actual_speed'].rolling(window=window_size, min_periods=1, center=True).mean()
        std_actual = iter_df['actual_speed'].rolling(window=window_size, min_periods=1, center=True).std()
        smoothed_error = iter_df['speed_error'].rolling(window=window_size, min_periods=1, center=True).mean()
        std_error = iter_df['speed_error'].rolling(window=window_size, min_periods=1, center=True).std()
        
        ax1.scatter(x, iter_df['actual_speed'], color=colors['actual_speed'], s=3.0, alpha=0.25, label='actual_speed (raw)', edgecolors='none')
        ax1.fill_between(x, smoothed_actual - std_actual, smoothed_actual + std_actual, color=colors['actual_speed'], alpha=0.12)
        ax1.plot(x, smoothed_actual, color=colors['actual_speed'], linewidth=2.2, alpha=0.95, label='actual_speed (smooth)')
        
        ax1.scatter(x, iter_df['speed_error'], color=colors['speed_error'], s=3.0, alpha=0.25, label='speed_error (raw)', edgecolors='none')
        ax1.fill_between(x, smoothed_error - std_error, smoothed_error + std_error, color=colors['speed_error'], alpha=0.12)
        ax1.plot(x, smoothed_error, color=colors['speed_error'], linewidth=2.2, alpha=0.95, label='speed_error (smooth)')
        
        ax2 = ax1.twinx()
        smoothed_reward = iter_df['tracking_reward'].rolling(window=window_size, min_periods=1, center=True).mean()
        std_reward = iter_df['tracking_reward'].rolling(window=window_size, min_periods=1, center=True).std()
        
        ax2.scatter(x, iter_df['tracking_reward'], color=colors['tracking_reward'], s=1.5, alpha=0.15, edgecolors='none')
        ax2.fill_between(x, (smoothed_reward - std_reward).clip(0, 1), (smoothed_reward + std_reward).clip(0, 1), color=colors['tracking_reward'], alpha=0.08)
        ax2.plot(x, smoothed_reward, color=colors['tracking_reward'], linewidth=2.5, alpha=0.95, label='tracking_reward (smooth)')
        
        ax1.set_title(f'Linear Velocity Tracking & Reward - Iteration {iteration}', fontsize=14, fontweight='bold', pad=12)
        ax1.set_xlabel('Environments (sorted by target speed)', fontsize=11, labelpad=8)
        ax1.set_ylabel('Velocity / Error (m/s)', fontsize=11, labelpad=8)
        ax2.set_ylabel('Tracking Reward [0.0, 1.0]', color=colors['tracking_reward'], fontsize=11, labelpad=8)
        
        ax1.set_ylim(-0.1, max(iter_df['cmd_speed_raw'].max() * 1.5, 1.5))
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis='y', labelcolor=colors['tracking_reward'])
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax2.grid(False)
        
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        by_label = dict(zip(l1 + l2, h1 + h2))
        legend_keys = ['cmd_speed_raw', 'cmd_speed_apf', 'actual_speed (smooth)', 'speed_error (smooth)', 'tracking_reward (smooth)']
        legend_handles = [by_label[k] for k in legend_keys if k in by_label]
        legend_labels = ['Cmd Speed (Raw)', 'Cmd Speed (APF)', 'Actual Speed (Smooth)', 'Tracking Error (Smooth)', 'Tracking Reward (Smooth)']
        ax1.legend(legend_handles, legend_labels, loc='upper left', frameon=True, facecolor='white', edgecolor='#e0e0e0', framealpha=0.9)
        
        plt.tight_layout()
        save_and_copy(fig, f'lin_vel_track_iter_{iteration}.png', workspace_dir, artifact_dir)
        plt.close(fig)
    print(f"-> Generated {len(iterations)} per-iteration environment detailed tracking curves.")

    # 2. Scatter grid cmd vs actual speed
    selected_iters = iterations if len(iterations) <= 6 else [iterations[i] for i in np.linspace(0, len(iterations) - 1, 6, dtype=int)]
    cols = min(3, len(selected_iters))
    rows = (len(selected_iters) + cols - 1) // cols
    fig_sc, axs_sc = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.2), sharex=True, sharey=True)
    axs_sc_flat = axs_sc.flatten() if hasattr(axs_sc, 'flatten') else [axs_sc]
    
    for idx, it in enumerate(selected_iters):
        ax = axs_sc_flat[idx]
        df_sub = df[df['iteration'] == it]
        ax.scatter(df_sub['cmd_speed_raw'], df_sub['actual_speed'], alpha=0.15, s=6, color=colors['actual_speed'], rasterized=True)
        max_val = max(df_sub['cmd_speed_raw'].max(), df_sub['actual_speed'].max(), 0.1)
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, linewidth=1.5, label='Ideal')
        mean_err = df_sub['speed_error'].mean()
        ax.set_title(f"Iter {it}\nMean Error: {mean_err:.3f} m/s", fontsize=10, weight='semibold')
        ax.set_xlabel("Cmd Speed (m/s)")
        ax.set_ylabel("Actual Speed (m/s)")
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
        ax.grid(True)
        
    for idx in range(len(selected_iters), len(axs_sc_flat)):
        axs_sc_flat[idx].axis('off')
        
    plt.tight_layout()
    save_and_copy(fig_sc, "vel_tracking_accuracy.png", workspace_dir, artifact_dir, dpi=200)
    plt.close(fig_sc)
    print("-> Generated vel_tracking_accuracy.png (Scatter Grid)")

    # 3. Box plots error and reward progression
    fig_bx, (axb1, axb2) = plt.subplots(1, 2, figsize=(15, 6))
    step_size = max(1, len(iterations) // 12)
    sampled_iters = iterations[::step_size]
    if iterations[-1] not in sampled_iters:
        sampled_iters.append(iterations[-1])
        
    boxplot_data_err = [df[df['iteration'] == it]['speed_error'].values for it in sampled_iters]
    boxplot_data_rew = [df[df['iteration'] == it]['tracking_reward'].values for it in sampled_iters]
    labels = [str(it) for it in sampled_iters]
    
    bp1 = axb1.boxplot(boxplot_data_err, patch_artist=True, tick_labels=labels, showfliers=False)
    for patch in bp1['boxes']:
        patch.set_facecolor(colors['speed_error'])
        patch.set_alpha(0.6)
    for median in bp1['medians']:
        median.set(color='black', linewidth=1.5)
    axb1.set_title("Speed Error Distribution Progression", weight='semibold')
    axb1.set_xlabel("Iteration")
    axb1.set_ylabel("Speed Error (m/s)")
    axb1.tick_params(axis='x', rotation=45)
    axb1.grid(True)
    
    bp2 = axb2.boxplot(boxplot_data_rew, patch_artist=True, tick_labels=labels, showfliers=False)
    for patch in bp2['boxes']:
        patch.set_facecolor(colors['tracking_reward'])
        patch.set_alpha(0.6)
    for median in bp2['medians']:
        median.set(color='black', linewidth=1.5)
    axb2.set_title("Tracking Reward Distribution Progression", weight='semibold')
    axb2.set_xlabel("Iteration")
    axb2.set_ylabel("Tracking Reward")
    axb2.tick_params(axis='x', rotation=45)
    axb2.grid(True)
    
    plt.tight_layout()
    save_and_copy(fig_bx, "vel_tracking_distribution_trends.png", workspace_dir, artifact_dir, dpi=200)
    plt.close(fig_bx)
    print("-> Generated vel_tracking_distribution_trends.png (Boxplot trends)")


def plot_curriculum_vel_tracking(csv_path, workspace_dir, artifact_dir):
    if not os.path.exists(csv_path):
        print(f"-> {os.path.basename(csv_path)} not found. Skipping curriculum upgrades plots.")
        return
        
    print(f"-> Loading curriculum upgrades from: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    iterations = sorted(df['iteration'].unique())
    
    for iteration in iterations:
        iter_df = df[df['iteration'] == iteration].sort_values('cmd_speed_raw').reset_index(drop=True)
        
        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
        x = iter_df.index
        
        ax1.plot(x, iter_df['cmd_speed_raw'], color=colors['cmd_speed_raw'], marker='o', linewidth=2.0, label='Cmd Speed (Raw)')
        ax1.plot(x, iter_df['cmd_speed_apf'], color=colors['cmd_speed_apf'], marker='x', linewidth=1.5, linestyle='--', label='Cmd Speed (APF)')
        ax1.plot(x, iter_df['actual_speed'], color=colors['actual_speed'], marker='s', linewidth=2.0, label='Actual Speed')
        ax1.plot(x, iter_df['speed_error'], color=colors['speed_error'], marker='d', linewidth=2.0, label='Speed Error')
        
        ax2 = ax1.twinx()
        ax2.plot(x, iter_df['tracking_reward'], color=colors['tracking_reward'], marker='^', linewidth=2.0, label='Tracking Reward')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Env {env}" for env in iter_df['env_id']], rotation=30)
        
        ax1.set_title(f'Curriculum Upgrade Velocity Tracking & Reward - Iteration {iteration}', fontsize=12, fontweight='bold', pad=15)
        ax1.set_xlabel('Upgrading Environments', fontsize=10, labelpad=8)
        ax1.set_ylabel('Velocity / Error (m/s)', fontsize=10, labelpad=8)
        ax2.set_ylabel('Tracking Reward [0.0, 1.0]', color=colors['tracking_reward'], fontsize=10, labelpad=8)
        
        # Display new range info
        x_min_val = iter_df['new_range_x_min'].iloc[0]
        x_max_val = iter_df['new_range_x_max'].iloc[0]
        y_min_val = iter_df['new_range_y_min'].iloc[0]
        y_max_val = iter_df['new_range_y_max'].iloc[0]
        info_text = f"New Range X: [{x_min_val:.1f}, {x_max_val:.1f}]\nNew Range Y: [{y_min_val:.1f}, {y_max_val:.1f}]"
        ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#e0e0e0'))
                 
        max_vel = max(iter_df['cmd_speed_raw'].max(), iter_df['actual_speed'].max(), 0.5)
        ax1.set_ylim(-0.05, max_vel * 1.3)
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis='y', labelcolor=colors['tracking_reward'])
        
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax2.grid(False)
        
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper right', frameon=True, facecolor='white', edgecolor='#e0e0e0', framealpha=0.9, fontsize=9)
        
        plt.tight_layout()
        save_and_copy(fig, f'curriculum_vel_track_iter_{iteration}.png', workspace_dir, artifact_dir)
        plt.close(fig)
        
    print(f"-> Generated {len(iterations)} curriculum upgrade tracking plots.")


def main():
    parser = argparse.ArgumentParser(description="Unified RL Log Visualization Skill")
    parser.add_argument('--log_dir', type=str, required=True, help='Path to the log directory containing events and CSVs.')
    parser.add_argument('--artifact_dir', type=str, default=None, help='Optional path to duplicate generated plots for ide artifacts.')
    
    args = parser.parse_args()
    log_dir = os.path.abspath(args.log_dir)
    artifact_dir = os.path.abspath(args.artifact_dir) if args.artifact_dir else None
    
    if not os.path.isdir(log_dir):
        print(f"Error: {log_dir} is not a valid directory.")
        return
        
    print(f"=== Starting RL Visualizations for {os.path.basename(log_dir)} ===")
    
    # 1. Plot overall training metrics from TensorBoard
    plot_tensorboard_summary(log_dir, log_dir, artifact_dir)
    
    # 2. Plot detailed tracking metrics from lin_vel_track_dist.csv
    plot_vel_tracking_dist(os.path.join(log_dir, "lin_vel_track_dist.csv"), log_dir, artifact_dir)
    
    # 3. Plot curriculum upgrade environments tracking metrics
    plot_curriculum_vel_tracking(os.path.join(log_dir, "curriculum_upgrade_vel_track.csv"), log_dir, artifact_dir)
    
    print("=== All visualizations completed successfully! ===")

if __name__ == '__main__':
    main()
