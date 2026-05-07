"""
Aerial Robotics Lab - Part 3: Python Simulator + nhfc Controller Plot Script
Same layout as Assignment 1 (plot_gazebo.py) for easy comparison.

Row 1: Tracking (desired=dashed, measured=solid)
Row 2: Errors + Forces/Torques

Usage: Run from your host machine (not Docker):
    python3 plot_part2b_old.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# ============================================================
#  DIRECTORIES
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, 'logs_part2b_old')
PLOT_DIR = os.path.join(SCRIPT_DIR, 'plots_part2b_old')

# ============================================================
#  PARSE LOG FILES
# ============================================================

def parse_log(filepath):
    """Parse a GenoM3 log file, skipping comment lines."""
    lines = []
    header = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            if header is None:
                header = line
                continue
            lines.append(line)

    columns = header.split()
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) == len(columns):
            try:
                data.append([float(x) for x in parts])
            except ValueError:
                continue

    return pd.DataFrame(data, columns=columns)


def quat_to_euler(q):
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])

# ============================================================
#  LOAD DATA
# ============================================================
print("Loading log files...")

# nhfc controller log
nhfc = parse_log(os.path.join(LOG_DIR, 'nhfc.log'))

# find simulation start: look for first large time gap (>1s) to skip setup period
dt_nhfc = nhfc['ts'].diff()
gap_idx = dt_nhfc[dt_nhfc > 1.0].index
if len(gap_idx) > 0:
    sim_start_idx = gap_idx[-1]
    nhfc = nhfc.iloc[sim_start_idx:].reset_index(drop=True)
    print(f"Skipped {sim_start_idx} pre-simulation samples (gap detected)")

# simulation data (our Python simulator output)
sim = np.load(os.path.join(LOG_DIR, 'simulation_data.npz'))
t_sim = sim['t']
x_sim = sim['x']
u_sim = sim['u']

sim_duration = t_sim[-1]

# Remap nhfc time to simulation time.
# nhfc.log uses wall-clock timestamps but simulation runs slower than real-time,
# so we linearly map nhfc sample indices to simulation time range.
nhfc['t'] = np.linspace(0, sim_duration, len(nhfc))
print(f"Remapped nhfc time: {len(nhfc)} samples -> 0..{sim_duration:.1f}s")

# compute euler angles from quaternions
euler_sim = np.array([quat_to_euler(x_sim[i, 3:7]) for i in range(len(t_sim))])

RAD2DEG = 180.0 / math.pi

print(f"nhfc samples: {len(nhfc)}")
print(f"sim samples:  {len(t_sim)}")
print(f"Duration:     {t_sim[-1]:.1f}s")

# ============================================================
#  PLOT - Same 2x5 layout as Assignment 1
# ============================================================
plt.rcParams.update({'font.size': 9, 'figure.dpi': 120})

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
fig.suptitle('Python Simulator + nhfc Controller (Part 3) - Assignment Plots', fontsize=14)

# ---- Row 1: Tracking (desired=dashed, measured=solid) ----

# Position [m]
ax = axes[0, 0]
ax.set_title('Position [m]')
for c, col, idx in [('x', 'red', 0), ('y', 'green', 1), ('z', 'blue', 2)]:
    ax.plot(t_sim, x_sim[:, idx], color=col, lw=0.8, label=f'{c} meas')
    ax.plot(nhfc['t'], nhfc[f'{c}d'], color=col, ls='--', lw=0.8, label=f'{c} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[m]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Attitude [deg]
ax = axes[0, 1]
ax.set_title('Attitude [deg]')
for c, col, idx in [('roll', 'red', 0), ('pitch', 'green', 1), ('yaw', 'blue', 2)]:
    ax.plot(t_sim, euler_sim[:, idx] * RAD2DEG, color=col, lw=0.8, label=f'{c} meas')
    ax.plot(nhfc['t'], nhfc[f'{c}d'] * RAD2DEG, color=col, ls='--', lw=0.8, label=f'{c} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[deg]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Linear Velocity [m/s]
ax = axes[0, 2]
ax.set_title('Linear Velocity [m/s]')
for c, col, idx in [('vx', 'red', 7), ('vy', 'green', 8), ('vz', 'blue', 9)]:
    ax.plot(t_sim, x_sim[:, idx], color=col, lw=0.8, label=f'{c} meas')
    ax.plot(nhfc['t'], nhfc[f'{c}d'], color=col, ls='--', lw=0.8, label=f'{c} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[m/s]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Angular Velocity [deg/s]
ax = axes[0, 3]
ax.set_title('Angular Velocity [deg/s]')
for c, col, idx in [('wx', 'red', 10), ('wy', 'green', 11), ('wz', 'blue', 12)]:
    ax.plot(t_sim, x_sim[:, idx] * RAD2DEG, color=col, lw=0.8, label=f'{c} meas')
    ax.plot(nhfc['t'], nhfc[f'{c}d'] * RAD2DEG, color=col, ls='--', lw=0.8, label=f'{c} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[deg/s]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Linear Acceleration [m/s²]
ax = axes[0, 4]
ax.set_title('Linear Acceleration [m/s²]')
for c, col in [('ax', 'red'), ('ay', 'green'), ('az', 'blue')]:
    ax.plot(nhfc['t'], nhfc[f'{c}d'], color=col, ls='--', lw=0.8, label=f'{c} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[m/s²]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# ---- Row 2: Errors + Forces ----

# Position Error [m]
ax = axes[1, 0]
ax.set_title('Position Error [m]')
for c, col in [('x', 'red'), ('y', 'green'), ('z', 'blue')]:
    ax.plot(nhfc['t'], nhfc[f'e_{c}'], color=col, lw=0.8, label=f'e_{c}')
ax.set_xlabel('t [s]'); ax.set_ylabel('[m]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Attitude Error [deg]
ax = axes[1, 1]
ax.set_title('Attitude Error [deg]')
for c, col, n in [('rx', 'red', 'roll'), ('ry', 'green', 'pitch'), ('rz', 'blue', 'yaw')]:
    ax.plot(nhfc['t'], nhfc[f'e_{c}'] * RAD2DEG, color=col, lw=0.8, label=f'e_{n}')
ax.set_xlabel('t [s]'); ax.set_ylabel('[deg]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Velocity Error [m/s]
ax = axes[1, 2]
ax.set_title('Velocity Error [m/s]')
for c, col in [('vx', 'red'), ('vy', 'green'), ('vz', 'blue')]:
    ax.plot(nhfc['t'], nhfc[f'e_{c}'], color=col, lw=0.8, label=f'e_{c}')
ax.set_xlabel('t [s]'); ax.set_ylabel('[m/s]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Angular Velocity Error [deg/s]
ax = axes[1, 3]
ax.set_title('Ang. Velocity Error [deg/s]')
for c, col in [('wx', 'red'), ('wy', 'green'), ('wz', 'blue')]:
    ax.plot(nhfc['t'], nhfc[f'e_{c}'] * RAD2DEG, color=col, lw=0.8, label=f'e_{c}')
ax.set_xlabel('t [s]'); ax.set_ylabel('[deg/s]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Force [N] and Torque [Nm]
ax = axes[1, 4]
ax.set_title('Force [N] & Torque [Nm]')
ax.plot(nhfc['t'], nhfc['fx'], color='red', lw=0.8, label='fx [N]')
ax.plot(nhfc['t'], nhfc['fy'], color='green', lw=0.8, label='fy [N]')
ax.plot(nhfc['t'], nhfc['fz'], color='blue', lw=0.8, label='fz [N]')
ax.plot(nhfc['t'], nhfc['tx'], color='red', ls='--', lw=0.8, label='tx [Nm]')
ax.plot(nhfc['t'], nhfc['ty'], color='green', ls='--', lw=0.8, label='ty [Nm]')
ax.plot(nhfc['t'], nhfc['tz'], color='blue', ls='--', lw=0.8, label='tz [Nm]')
ax.set_xlabel('t [s]'); ax.set_ylabel('[N] / [Nm]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

fig.tight_layout()

# Save
os.makedirs(PLOT_DIR, exist_ok=True)
fig.savefig(os.path.join(PLOT_DIR, 'part2b_old_plots.png'), dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {PLOT_DIR}/part2b_old_plots.png")

plt.show()
