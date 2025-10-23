#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SDWPF Dataset Visualization for Demonstration
Shows the structure and characteristics of the preprocessed wind power data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

print("="*60)
print("SDWPF Wind Power Dataset Visualization")
print("="*60)

# Load preprocessed data
print("\nLoading preprocessed data...")
data = np.load('Data/cvitrnn_full_134turbines.npz', allow_pickle=True)

# Extract components
train_frames = data['train_frames']
val_frames = data['val_frames']
test_frames = data['test_frames']
grid_shape = data['grid_shape']
turbine_positions = data['turbine_positions'].item()
feature_columns = data['feature_columns']
grid = data['grid']

print(f"\nDataset Information:")
print(f"  Training samples: {train_frames.shape[0]:,} ({train_frames.shape})")
print(f"  Validation samples: {val_frames.shape[0]:,} ({val_frames.shape})")
print(f"  Test samples: {test_frames.shape[0]:,} ({test_frames.shape})")
print(f"  Grid layout: {grid_shape[0]}×{grid_shape[1]}")
print(f"  Number of turbines: {len(turbine_positions)}")
print(f"  Features per turbine: {len(feature_columns)}")
print(f"  Total data points: {(train_frames.shape[0] + val_frames.shape[0] + test_frames.shape[0]) * 134 * 16:,}")

# Feature names
feature_names = [str(f) for f in feature_columns]
print(f"\nFeatures: {feature_names}")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

# ========== 1. Wind Farm Spatial Layout (Large) ==========
ax1 = fig.add_subplot(gs[0:2, 0:2])

# 创建二值化的显示矩阵
display_grid = np.where(grid > 0, 1, 0)

# 使用自定义颜色：浅灰背景，深蓝涡轮机
from matplotlib.colors import ListedColormap
colors = ['#f0f0f0', '#2c3e50']  # 浅灰和深蓝
cmap = ListedColormap(colors)

im = ax1.imshow(display_grid, cmap=cmap, aspect='auto')

# Add turbine IDs on the grid
for h in range(grid_shape[0]):
    for w in range(grid_shape[1]):
        if grid[h, w] > 0:
            # 深色背景用白色文字
            ax1.text(w, h, str(int(grid[h, w])),
                    ha='center', va='center', color='white',
                    fontsize=7, fontweight='bold')

ax1.set_title('Wind Farm Layout - 134 Turbines on 23×6 Grid',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Grid Width (6 columns)')
ax1.set_ylabel('Grid Height (23 rows)')
ax1.set_xticks(range(grid_shape[1]))
ax1.set_yticks(range(0, grid_shape[0], 2))
ax1.grid(False)


# ========== 2. Dataset Split Pie Chart ==========
ax2 = fig.add_subplot(gs[0, 2])
sizes = [train_frames.shape[0], val_frames.shape[0], test_frames.shape[0]]
labels = [f'Train\n{sizes[0]:,}', f'Val\n{sizes[1]:,}', f'Test\n{sizes[2]:,}']
colors = ['#2E7D32', '#1976D2', '#F57C00']
explode = (0.05, 0.05, 0.05)

wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90,
                                     explode=explode, shadow=True,
                                     textprops={'fontsize': 10})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax2.set_title('Dataset Split (80/10/10)', fontsize=12, fontweight='bold')

# ========== 3. Time Coverage ==========
ax3 = fig.add_subplot(gs[0, 3])
time_info = [
    ('Total Duration', '2 years'),
    ('Start', '2020-01-01'),
    ('End', '2022-01-01'),
    ('Sampling', '10 minutes'),
    ('Daily Points', '144'),
    ('Total Timestamps', f'{train_frames.shape[0] + val_frames.shape[0] + test_frames.shape[0]:,}')
]

for i, (label, value) in enumerate(time_info):
    ax3.text(0.1, 0.9 - i*0.15, f'{label}:', fontsize=10, fontweight='bold')
    ax3.text(0.6, 0.9 - i*0.15, value, fontsize=10)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('Temporal Information', fontsize=12, fontweight='bold')
ax3.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='gray', lw=1))

# ========== 4. Feature Categories ==========
ax4 = fig.add_subplot(gs[1, 2])

# Categorize features
categories = {
    'Wind': ['Wspd', 'Wdir', 'Wspd_w', 'Wdir_w', 'Ndir'],
    'Temperature': ['Etmp', 'Itmp', 'T2m', 'Tp'],
    'Power': ['Patv', 'Prtv', 'Pab1', 'Pab2', 'Pab3'],
    'Others': ['Sp', 'RelH']
}

# Count features per category
cat_names = list(categories.keys())
cat_counts = [len(v) for v in categories.values()]

bars = ax4.bar(cat_names, cat_counts, color=['#4CAF50', '#FF5722', '#FFC107', '#9C27B0'])
ax4.set_ylabel('Number of Features')
ax4.set_title('Feature Categories', fontsize=12, fontweight='bold')
ax4.set_ylim(0, max(cat_counts) + 1)

# Add value labels on bars
for bar, count in zip(bars, cat_counts):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{count}', ha='center', va='bottom', fontweight='bold')

# ========== 5. Power Output Time Series ==========
ax5 = fig.add_subplot(gs[2, :2])

# Get power data for multiple turbines
power_idx = feature_names.index('Patv')
time_points = 2000  # Show first 2000 time points (~2 weeks)
turbine_samples = [0, 10, 20, 30, 40]  # Sample 5 turbines

for t_idx in turbine_samples:
    row = t_idx // 6
    col = t_idx % 6
    power_series = train_frames[:time_points, row, col, power_idx]
    ax5.plot(power_series, alpha=0.7, linewidth=0.8,
            label=f'Turbine {grid[row, col]}')

ax5.set_xlabel('Time (10-minute intervals)')
ax5.set_ylabel('Normalized Power Output')
ax5.set_title('Power Output Time Series - Multiple Turbines (2 weeks)',
              fontsize=12, fontweight='bold')
ax5.legend(loc='upper right', ncol=5, fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, time_points)

# ========== 6. Wind Speed vs Power Scatter ==========
ax6 = fig.add_subplot(gs[2, 2:])

# Sample data for scatter plot
wspd_idx = feature_names.index('Wspd')
sample_size = 5000
sample_frames = np.random.choice(train_frames.shape[0],
                                 min(sample_size, train_frames.shape[0]),
                                 replace=False)

wind_speeds = []
powers = []
for frame_idx in sample_frames[:1000]:  # Use 1000 points for clarity
    frame_wind = train_frames[frame_idx, :, :, wspd_idx].flatten()
    frame_power = train_frames[frame_idx, :, :, power_idx].flatten()

    # Filter out padding (zeros)
    valid_mask = (frame_wind != 0) & (frame_power != 0)
    wind_speeds.extend(frame_wind[valid_mask])
    powers.extend(frame_power[valid_mask])

# Create hexbin plot for density
hb = ax6.hexbin(wind_speeds[:5000], powers[:5000], gridsize=30,
                cmap='YlOrRd', mincnt=1)
ax6.set_xlabel('Normalized Wind Speed')
ax6.set_ylabel('Normalized Power Output')
ax6.set_title('Wind Speed vs Power Relationship', fontsize=12, fontweight='bold')
plt.colorbar(hb, ax=ax6, label='Count')

# ========== 7. Feature Correlation Matrix ==========
ax7 = fig.add_subplot(gs[1, 3])

# Calculate correlation for a sample of data
sample_data = train_frames[0, :, :, :].reshape(-1, len(feature_columns))
# Remove zero padding
non_zero_mask = ~(sample_data == 0).all(axis=1)
sample_data = sample_data[non_zero_mask]

# Select key features for correlation
key_features = ['Wspd', 'Patv', 'Etmp', 'Prtv', 'Wdir']
key_indices = [feature_names.index(f) for f in key_features]
correlation_data = sample_data[:, key_indices]

# Calculate correlation
correlation_matrix = np.corrcoef(correlation_data.T)

# Plot heatmap
im = ax7.imshow(correlation_matrix, cmap='coolwarm', aspect='auto',
                vmin=-1, vmax=1)
ax7.set_xticks(range(len(key_features)))
ax7.set_yticks(range(len(key_features)))
ax7.set_xticklabels(key_features, rotation=45, ha='right')
ax7.set_yticklabels(key_features)
ax7.set_title('Feature Correlations', fontsize=12, fontweight='bold')

# Add correlation values
for i in range(len(key_features)):
    for j in range(len(key_features)):
        text = ax7.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                       ha='center', va='center',
                       color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black',
                       fontsize=8)

# ========== 8. Daily Power Pattern ==========
ax8 = fig.add_subplot(gs[3, :2])

# Calculate average daily pattern
hours_per_day = 144  # 24 hours * 6 (10-min intervals)
days_to_analyze = min(7, train_frames.shape[0] // hours_per_day)

daily_patterns = []
for day in range(days_to_analyze):
    start_idx = day * hours_per_day
    end_idx = start_idx + hours_per_day
    if end_idx <= train_frames.shape[0]:
        day_data = train_frames[start_idx:end_idx, :, :, power_idx]
        # Average across all turbines
        daily_avg = np.mean(day_data, axis=(1, 2))
        daily_patterns.append(daily_avg)

if daily_patterns:
    daily_patterns = np.array(daily_patterns)
    mean_pattern = np.mean(daily_patterns, axis=0)
    std_pattern = np.std(daily_patterns, axis=0)

    time_hours = np.arange(len(mean_pattern)) / 6  # Convert to hours

    ax8.plot(time_hours, mean_pattern, 'b-', linewidth=2, label='Mean')
    ax8.fill_between(time_hours,
                     mean_pattern - std_pattern,
                     mean_pattern + std_pattern,
                     alpha=0.3, color='blue', label='±1 STD')

    ax8.set_xlabel('Hour of Day')
    ax8.set_ylabel('Average Normalized Power')
    ax8.set_title('24-Hour Power Generation Pattern (7-day average)',
                  fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(0, 24)
    ax8.set_xticks(range(0, 25, 3))

# ========== 9. Power Distribution Histogram ==========
ax9 = fig.add_subplot(gs[3, 2])

# Get all power values
all_power = train_frames[:1000, :, :, power_idx].flatten()
all_power = all_power[all_power != 0]  # Remove padding

ax9.hist(all_power, bins=50, color='#4CAF50', alpha=0.7,
         edgecolor='black', density=True)
ax9.axvline(np.mean(all_power), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {np.mean(all_power):.3f}')
ax9.axvline(np.median(all_power), color='blue', linestyle='--',
           linewidth=2, label=f'Median: {np.median(all_power):.3f}')

ax9.set_xlabel('Normalized Power Output')
ax9.set_ylabel('Density')
ax9.set_title('Power Output Distribution', fontsize=12, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

# ========== 10. Data Quality Metrics ==========
ax10 = fig.add_subplot(gs[3, 3])

# Calculate data quality metrics
total_points = train_frames.size
non_zero_points = np.count_nonzero(train_frames)
zero_points = total_points - non_zero_points

quality_metrics = [
    ('Total Points', f'{total_points:,}'),
    ('Non-zero', f'{non_zero_points:,}'),
    ('Zero (padding)', f'{zero_points:,}'),
    ('Coverage', f'{100*non_zero_points/total_points:.1f}%'),
    ('Memory Size', f'{train_frames.nbytes / (1024**3):.2f} GB'),
]

for i, (label, value) in enumerate(quality_metrics):
    ax10.text(0.1, 0.85 - i*0.18, f'{label}:', fontsize=9, fontweight='bold')
    ax10.text(0.55, 0.85 - i*0.18, value, fontsize=9)

ax10.set_xlim(0, 1)
ax10.set_ylim(0, 1)
ax10.axis('off')
ax10.set_title('Data Quality Metrics', fontsize=12, fontweight='bold')
ax10.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                         edgecolor='gray', lw=1))

# Overall title and layout
fig.suptitle('SDWPF Wind Power Dataset - Comprehensive Visualization',
             fontsize=16, fontweight='bold', y=0.98)

# Add description
description = ("134 Wind Turbines | 2 Years Data (2020-2022) | "
               "10-min Sampling | 16 Features | 84,785 Timestamps")
fig.text(0.5, 0.01, description, ha='center', fontsize=11,
         style='italic', color='gray')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# Save figure
output_file = 'sdwpf_dataset_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nVisualization saved as '{output_file}'")

# Don't show to avoid timeout
# plt.show()

print("\n" + "="*60)
print("Visualization Complete!")
print("="*60)