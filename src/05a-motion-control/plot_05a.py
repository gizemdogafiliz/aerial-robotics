"""
Assignment 05a: Motion Control — uavpos/uavatt vs nhfc comparison
Plots trajectory tracking + wrench in body frame.

Usage:
  python3 plot_05a.py             # plot uavpos/uavatt results
  python3 plot_05a.py compare     # overlay nhfc (03a) vs uavpos/uavatt (05a)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, 'plots_05a')

LOG_DIR_05A = os.path.expanduser('~/tk3lab-ws/logs/05a-motion-control')
LOG_DIR_03A = os.path.expanduser('~/tk3lab-ws/logs/03a-trajectory/hexa-fa')

RAD2DEG = 180.0 / math.pi

COMPARE = len(sys.argv) > 1 and sys.argv[1] == 'compare'


def parse_log(filepath):
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


def load_sim_data(log_dir):
    sim = np.load(os.path.join(log_dir, 'simulation_data.npz'))
    t = sim['t']
    x = sim['x']
    u = sim['u']
    w = sim['w'] if 'w' in sim else None
    euler = np.array([quat_to_euler(x[i, 3:7]) for i in range(len(t))])
    return t, x, u, w, euler


def align_log(df, sim_duration):
    dt = df['ts'].diff()
    gap_idx = dt[dt > 1.0].index
    if len(gap_idx) > 0:
        df = df.iloc[gap_idx[-1]:].reset_index(drop=True)
    df['t'] = np.linspace(0, sim_duration, len(df))
    return df


# ============================================================
#  LOAD DATA
# ============================================================
print(f"Loading 05a data from: {LOG_DIR_05A}")
t_05a, x_05a, u_05a, w_05a, euler_05a = load_sim_data(LOG_DIR_05A)
sim_duration = t_05a[-1]

uavpos_log = None
uavatt_log = None
uavpos_path = os.path.join(LOG_DIR_05A, 'uavpos.log')
uavatt_path = os.path.join(LOG_DIR_05A, 'uavatt.log')
if os.path.exists(uavpos_path):
    uavpos_log = parse_log(uavpos_path)
    uavpos_log = align_log(uavpos_log, sim_duration)
    print(f"  uavpos samples: {len(uavpos_log)}, cols: {list(uavpos_log.columns[:10])}...")
if os.path.exists(uavatt_path):
    uavatt_log = parse_log(uavatt_path)
    uavatt_log = align_log(uavatt_log, sim_duration)
    print(f"  uavatt samples: {len(uavatt_log)}, cols: {list(uavatt_log.columns[:10])}...")

t_03a, x_03a, euler_03a = None, None, None
nhfc_log = None
if COMPARE:
    print(f"Loading 03a data from: {LOG_DIR_03A}")
    t_03a, x_03a, _, _, euler_03a = load_sim_data(LOG_DIR_03A)
    nhfc_path = os.path.join(LOG_DIR_03A, 'nhfc.log')
    if os.path.exists(nhfc_path):
        nhfc_log = parse_log(nhfc_path)
        nhfc_log = align_log(nhfc_log, t_03a[-1])

print(f"Duration: {sim_duration:.1f}s")

# ============================================================
#  FIGURE 1: Trajectory Tracking (3x2)
# ============================================================
fig1, axes1 = plt.subplots(3, 2, figsize=(14, 10))
title = 'Trajectory Tracking — uavpos/uavatt'
if COMPARE:
    title += ' vs nhfc'
fig1.suptitle(title, fontsize=13)

state_groups = [
    (0, 0, 'Position [m]',         ['x', 'y', 'z'],    [0, 1, 2],   1.0),
    (0, 1, 'Attitude [deg]',       ['roll', 'pitch', 'yaw'], None,   RAD2DEG),
    (1, 0, 'Linear Velocity [m/s]', ['vx', 'vy', 'vz'], [7, 8, 9],  1.0),
    (1, 1, 'Angular Velocity [deg/s]', ['wx', 'wy', 'wz'], [10, 11, 12], RAD2DEG),
]

colors = {'x': 'red', 'y': 'green', 'z': 'blue',
          'roll': 'red', 'pitch': 'green', 'yaw': 'blue',
          'vx': 'red', 'vy': 'green', 'vz': 'blue',
          'wx': 'red', 'wy': 'green', 'wz': 'blue'}

for row, col, title_str, labels, sim_idx, scale in state_groups:
    ax = axes1[row, col]
    ax.set_title(title_str)

    for j, lbl in enumerate(labels):
        c = colors[lbl]
        if sim_idx is not None:
            ax.plot(t_05a, x_05a[:, sim_idx[j]] * scale, color=c, lw=1.0,
                    label=f'{lbl} uavpos')
        else:
            ax.plot(t_05a, euler_05a[:, j] * scale, color=c, lw=1.0,
                    label=f'{lbl} uavpos')

        if COMPARE and x_03a is not None:
            if sim_idx is not None:
                ax.plot(t_03a, x_03a[:, sim_idx[j]] * scale, color=c, ls='--', lw=0.7,
                        alpha=0.7, label=f'{lbl} nhfc')
            else:
                ax.plot(t_03a, euler_03a[:, j] * scale, color=c, ls='--', lw=0.7,
                        alpha=0.7, label=f'{lbl} nhfc')

    ax.set_xlabel('t [s]')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

# Position error from uavpos log
ax = axes1[2, 0]
ax.set_title('Position Error [m]')
if uavpos_log is not None:
    for c_name, c_col in [('e_x', 'red'), ('e_y', 'green'), ('e_z', 'blue')]:
        if c_name in uavpos_log.columns:
            ax.plot(uavpos_log['t'], uavpos_log[c_name], color=c_col, lw=0.8, label=f'{c_name} uavpos')
if COMPARE and nhfc_log is not None:
    for c_name, c_col in [('e_x', 'red'), ('e_y', 'green'), ('e_z', 'blue')]:
        if c_name in nhfc_log.columns:
            ax.plot(nhfc_log['t'], nhfc_log[c_name], color=c_col, ls='--', lw=0.7,
                    alpha=0.7, label=f'{c_name} nhfc')
ax.set_xlabel('t [s]')
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

# Attitude error from uavatt log
ax = axes1[2, 1]
ax.set_title('Attitude Error [deg]')
if uavatt_log is not None:
    for c_name, c_col, lbl in [('e_rx', 'red', 'e_roll'), ('e_ry', 'green', 'e_pitch'), ('e_rz', 'blue', 'e_yaw')]:
        if c_name in uavatt_log.columns:
            ax.plot(uavatt_log['t'], uavatt_log[c_name] * RAD2DEG, color=c_col, lw=0.8, label=f'{lbl} uavatt')
if COMPARE and nhfc_log is not None:
    for c_name, c_col, lbl in [('e_rx', 'red', 'e_roll'), ('e_ry', 'green', 'e_pitch'), ('e_rz', 'blue', 'e_yaw')]:
        if c_name in nhfc_log.columns:
            ax.plot(nhfc_log['t'], nhfc_log[c_name] * RAD2DEG, color=c_col, ls='--', lw=0.7,
                    alpha=0.7, label=f'{lbl} nhfc')
ax.set_xlabel('t [s]')
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

fig1.tight_layout()

# ============================================================
#  FIGURE 2: Wrench in Body Frame (2x1)
# ============================================================
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 7))
fig2.suptitle('Wrench in Body Frame — uavpos/uavatt', fontsize=13)

if w_05a is not None:
    ax = axes2[0]
    ax.set_title('Body Forces [N]')
    for j, (lbl, c) in enumerate([('fx', 'red'), ('fy', 'green'), ('fz', 'blue')]):
        ax.plot(t_05a, w_05a[:, j], color=c, lw=0.8, label=lbl)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('[N]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.set_title('Body Torques [Nm]')
    for j, (lbl, c) in enumerate([('τx', 'red'), ('τy', 'green'), ('τz', 'blue')]):
        ax.plot(t_05a, w_05a[:, 3+j], color=c, lw=0.8, label=lbl)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('[Nm]')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    axes2[0].text(0.5, 0.5, 'No wrench data (w) in simulation_data.npz',
                  ha='center', va='center', transform=axes2[0].transAxes)

fig2.tight_layout()

# ============================================================
#  SAVE
# ============================================================
os.makedirs(PLOT_DIR, exist_ok=True)

suffix = '_compare' if COMPARE else ''
fig1.savefig(os.path.join(PLOT_DIR, f'tracking{suffix}.png'), dpi=150, bbox_inches='tight')
fig2.savefig(os.path.join(PLOT_DIR, f'wrench{suffix}.png'), dpi=150, bbox_inches='tight')
print(f"\nPlots saved to: {PLOT_DIR}/")

plt.show()
