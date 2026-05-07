"""
Assignment 05b: Python Feedback Linearization Controller
Plots trajectory tracking + wrench, with optional comparison to nhfc (03a) and uavpos/uavatt (05a).

Usage:
  python3 plot_05b.py             # plot 05b results only
  python3 plot_05b.py compare     # overlay nhfc (03a) + uavpos/uavatt (05a) + Python FB-lin (05b)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, 'plots_05b')

LOG_DIR_05B = os.path.expanduser('~/tk3lab-ws/logs/05b-motion-control')
LOG_DIR_05A = os.path.expanduser('~/tk3lab-ws/logs/05a-motion-control')
LOG_DIR_03A = os.path.expanduser('~/tk3lab-ws/logs/03a-trajectory/hexa-fa')

RAD2DEG = 180.0 / math.pi

COMPARE = len(sys.argv) > 1 and sys.argv[1] == 'compare'


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
    ep = sim['ep'] if 'ep' in sim else None
    eR = sim['eR'] if 'eR' in sim else None
    euler = np.array([quat_to_euler(x[i, 3:7]) for i in range(len(t))])
    return t, x, u, w, ep, eR, euler


# ============================================================
#  LOAD DATA
# ============================================================
print(f"Loading 05b data from: {LOG_DIR_05B}")
t_05b, x_05b, u_05b, w_05b, ep_05b, eR_05b, euler_05b = load_sim_data(LOG_DIR_05B)
sim_duration = t_05b[-1]

t_05a, x_05a, w_05a, euler_05a = None, None, None, None
t_03a, x_03a, euler_03a = None, None, None

if COMPARE:
    if os.path.exists(os.path.join(LOG_DIR_05A, 'simulation_data.npz')):
        print(f"Loading 05a data from: {LOG_DIR_05A}")
        t_05a, x_05a, _, w_05a, _, _, euler_05a = load_sim_data(LOG_DIR_05A)
    if os.path.exists(os.path.join(LOG_DIR_03A, 'simulation_data.npz')):
        print(f"Loading 03a data from: {LOG_DIR_03A}")
        t_03a, x_03a, _, _, _, _, euler_03a = load_sim_data(LOG_DIR_03A)

print(f"Duration: {sim_duration:.1f}s")

# ============================================================
#  FIGURE 1: Trajectory Tracking (3x2)
# ============================================================
fig1, axes1 = plt.subplots(3, 2, figsize=(14, 10))
title = 'Trajectory Tracking — Python FB Linearization'
if COMPARE:
    title += ' vs nhfc vs uavpos/uavatt'
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
            ax.plot(t_05b, x_05b[:, sim_idx[j]] * scale, color=c, lw=1.0,
                    label=f'{lbl} python')
        else:
            ax.plot(t_05b, euler_05b[:, j] * scale, color=c, lw=1.0,
                    label=f'{lbl} python')

        if COMPARE:
            if x_05a is not None:
                if sim_idx is not None:
                    ax.plot(t_05a, x_05a[:, sim_idx[j]] * scale, color=c, ls='--', lw=0.7,
                            alpha=0.7, label=f'{lbl} uavpos')
                else:
                    ax.plot(t_05a, euler_05a[:, j] * scale, color=c, ls='--', lw=0.7,
                            alpha=0.7, label=f'{lbl} uavpos')
            if x_03a is not None:
                if sim_idx is not None:
                    ax.plot(t_03a, x_03a[:, sim_idx[j]] * scale, color=c, ls=':', lw=0.7,
                            alpha=0.5, label=f'{lbl} nhfc')
                else:
                    ax.plot(t_03a, euler_03a[:, j] * scale, color=c, ls=':', lw=0.7,
                            alpha=0.5, label=f'{lbl} nhfc')

    ax.set_xlabel('t [s]')
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

# Position error
ax = axes1[2, 0]
ax.set_title('Position Error [m]')
ref_pos = None
if ep_05b is not None:
    ref_pos = x_05b[:, 0:3] + ep_05b
    for j, (lbl, c) in enumerate([('e_x', 'red'), ('e_y', 'green'), ('e_z', 'blue')]):
        ax.plot(t_05b, ep_05b[:, j], color=c, lw=0.8, label=f'{lbl} python')

    if COMPARE and ref_pos is not None:
        if x_05a is not None:
            for j, (lbl, c) in enumerate([('e_x', 'red'), ('e_y', 'green'), ('e_z', 'blue')]):
                pos_interp = np.interp(t_05b, t_05a, x_05a[:, j])
                ax.plot(t_05b, ref_pos[:, j] - pos_interp, color=c, ls='--', lw=0.7,
                        alpha=0.7, label=f'{lbl} uavpos')
        if x_03a is not None:
            for j, (lbl, c) in enumerate([('e_x', 'red'), ('e_y', 'green'), ('e_z', 'blue')]):
                pos_interp = np.interp(t_05b, t_03a, x_03a[:, j])
                ax.plot(t_05b, ref_pos[:, j] - pos_interp, color=c, ls=':', lw=0.7,
                        alpha=0.5, label=f'{lbl} nhfc')

ax.set_xlabel('t [s]')
ax.legend(fontsize=5)
ax.grid(True, alpha=0.3)

# Attitude error
ax = axes1[2, 1]
ax.set_title('Attitude Error [deg]')
ref_euler = None
if eR_05b is not None:
    ref_euler = euler_05b + eR_05b
    for j, (lbl, c) in enumerate([('e_roll', 'red'), ('e_pitch', 'green'), ('e_yaw', 'blue')]):
        ax.plot(t_05b, eR_05b[:, j] * RAD2DEG, color=c, lw=0.8, label=f'{lbl} python')

    if COMPARE and ref_euler is not None:
        if euler_05a is not None:
            for j, (lbl, c) in enumerate([('e_roll', 'red'), ('e_pitch', 'green'), ('e_yaw', 'blue')]):
                eu_interp = np.interp(t_05b, t_05a, euler_05a[:, j])
                ref_interp = ref_euler[:, j]
                ax.plot(t_05b, (ref_interp - eu_interp) * RAD2DEG, color=c, ls='--', lw=0.7,
                        alpha=0.7, label=f'{lbl} uavpos')
        if euler_03a is not None:
            for j, (lbl, c) in enumerate([('e_roll', 'red'), ('e_pitch', 'green'), ('e_yaw', 'blue')]):
                eu_interp = np.interp(t_05b, t_03a, euler_03a[:, j])
                ref_interp = ref_euler[:, j]
                ax.plot(t_05b, (ref_interp - eu_interp) * RAD2DEG, color=c, ls=':', lw=0.7,
                        alpha=0.5, label=f'{lbl} nhfc')

ax.set_xlabel('t [s]')
ax.legend(fontsize=5)
ax.grid(True, alpha=0.3)

fig1.tight_layout()

# ============================================================
#  FIGURE 2: Wrench in Body Frame (2x1)
# ============================================================
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 7))
fig2.suptitle('Wrench in Body Frame — Python FB Linearization', fontsize=13)

if w_05b is not None:
    ax = axes2[0]
    ax.set_title('Body Forces [N]')
    for j, (lbl, c) in enumerate([('fx', 'red'), ('fy', 'green'), ('fz', 'blue')]):
        ax.plot(t_05b, w_05b[:, j], color=c, lw=0.8, label=lbl)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('[N]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.set_title('Body Torques [Nm]')
    for j, (lbl, c) in enumerate([('tx', 'red'), ('ty', 'green'), ('tz', 'blue')]):
        ax.plot(t_05b, w_05b[:, 3+j], color=c, lw=0.8, label=lbl)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('[Nm]')
    ax.legend()
    ax.grid(True, alpha=0.3)

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
