"""
Aerial Robotics Lab - Part 1: Gazebo Simulation Plot Script
Plots tracking (desired vs measured), controller errors, and force/torque inputs.

Usage: Run from your host machine (not Docker):
    python3 plot_gazebo.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# ============================================================
#  LOG DIRECTORY - change this to your log path
# ============================================================
LOG_DIR = os.path.expanduser('~/tk3lab-ws/logs/01-intro/hexa-fa')

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


def parse_pom_log(filepath):
    """Parse pom.log - header is inside comments starting with 'ts'."""
    lines = []
    header = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                stripped = line.lstrip('# ').strip()
                if stripped.startswith('ts'):
                    header = stripped
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


# ============================================================
#  LOAD DATA
# ============================================================
print("Loading log files...")

nhfc = parse_log(os.path.join(LOG_DIR, 'nhfc.log'))
pom = parse_log(os.path.join(LOG_DIR, 'pom.log'))

# t0 = smallest timestamp
t0 = min(nhfc['ts'].min(), pom['ts'].min())
nhfc['t'] = nhfc['ts'] - t0
pom['t'] = pom['ts'] - t0

RAD2DEG = 180.0 / math.pi

print(f"nhfc samples: {len(nhfc)}")
print(f"pom samples:  {len(pom)}")
print(f"Duration:     {pom['t'].max():.1f}s")

# ============================================================
#  PLOT
# ============================================================
plt.rcParams.update({'font.size': 9, 'figure.dpi': 120})

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
fig.suptitle('Quadrotor Gazebo Simulation - Assignment Plots', fontsize=14)

# ---- Row 1: Tracking (desired=dashed, measured=solid) ----

# Position [m]
ax = axes[0, 0]
ax.set_title('Position [m]')
for c, col in [('x', 'red'), ('y', 'green'), ('z', 'blue')]:
    ax.plot(pom['t'], pom[c], color=col, lw=0.8, label=f'{c} meas')
    ax.plot(nhfc['t'], nhfc[f'{c}d'], color=col, ls='--', lw=0.8, label=f'{c} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[m]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Attitude [deg]
ax = axes[0, 1]
ax.set_title('Attitude [deg]')
for c, col in [('roll', 'red'), ('pitch', 'green'), ('yaw', 'blue')]:
    ax.plot(pom['t'], pom[c] * RAD2DEG, color=col, lw=0.8, label=f'{c} meas')
    ax.plot(nhfc['t'], nhfc[f'{c}d'] * RAD2DEG, color=col, ls='--', lw=0.8, label=f'{c} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[deg]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Linear Velocity [m/s]
ax = axes[0, 2]
ax.set_title('Linear Velocity [m/s]')
for c, col, n in [('vx', 'red', 'x'), ('vy', 'green', 'y'), ('vz', 'blue', 'z')]:
    ax.plot(pom['t'], pom[c], color=col, lw=0.8, label=f'v{n} meas')
    ax.plot(nhfc['t'], nhfc[f'{c}d'], color=col, ls='--', lw=0.8, label=f'v{n} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[m/s]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Angular Velocity [deg/s]
ax = axes[0, 3]
ax.set_title('Angular Velocity [deg/s]')
for c, col, n in [('wx', 'red', 'x'), ('wy', 'green', 'y'), ('wz', 'blue', 'z')]:
    ax.plot(pom['t'], pom[c] * RAD2DEG, color=col, lw=0.8, label=f'w{n} meas')
    ax.plot(nhfc['t'], nhfc[f'{c}d'] * RAD2DEG, color=col, ls='--', lw=0.8, label=f'w{n} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[deg/s]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Linear Acceleration [m/s^2]
ax = axes[0, 4]
ax.set_title('Linear Acceleration [m/s²]')
for c, col, n in [('ax', 'red', 'x'), ('ay', 'green', 'y'), ('az', 'blue', 'z')]:
    ax.plot(pom['t'], pom[c], color=col, lw=0.8, label=f'a{n} meas')
    ax.plot(nhfc['t'], nhfc[f'{c}d'], color=col, ls='--', lw=0.8, label=f'a{n} des')
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
output_dir = os.path.join(LOG_DIR, 'plots')
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, 'quadrotor_gazebo_plots.png'), dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_dir}/quadrotor_gazebo_plots.png")

plt.show()