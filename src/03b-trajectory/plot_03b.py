import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import sys

# ============================================================
#  DIRECTORIES
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, 'plots_03b')

ROBOT = sys.argv[1] if len(sys.argv) > 1 else 'quad'
LOG_DIR = os.path.expanduser(f'~/tk3lab-ws/logs/03b-trajectory/{ROBOT}')

# ============================================================
#  PARSE LOG FILES
# ============================================================

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


# ============================================================
#  LOAD DATA
# ============================================================
print(f"Loading log files from: {LOG_DIR}")

nhfc = parse_log(os.path.join(LOG_DIR, 'nhfc.log'))

sim = np.load(os.path.join(LOG_DIR, 'simulation_data.npz'))
t_sim = sim['t']
x_sim = sim['x']
u_sim = sim['u']
ref = sim['ref']  # x,y,z, vx,vy,vz, ax,ay,az, yaw

sim_duration = t_sim[-1]

# nhfc uses wall-clock timestamps; remap to simulation time
dt_nhfc = nhfc['ts'].diff()
gap_idx = dt_nhfc[dt_nhfc > 1.0].index
if len(gap_idx) > 0:
    sim_start_idx = gap_idx[-1]
    nhfc = nhfc.iloc[sim_start_idx:].reset_index(drop=True)
    print(f"Skipped {sim_start_idx} pre-simulation samples")

nhfc['t'] = np.linspace(0, sim_duration, len(nhfc))

euler_sim = np.array([quat_to_euler(x_sim[i, 3:7]) for i in range(len(t_sim))])

RAD2DEG = 180.0 / math.pi

print(f"nhfc samples: {len(nhfc)}")
print(f"sim samples:  {len(t_sim)}")
print(f"Duration:     {sim_duration:.1f}s")

# ============================================================
#  PLOT — 2x5 layout
# ============================================================
plt.rcParams.update({'font.size': 9, 'figure.dpi': 120})

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
fig.suptitle(f'Python Simulator + Polynomial Trajectory + nhfc — {ROBOT} (Assignment 3b)', fontsize=14)

# ---- Row 1: Tracking (desired=dashed, measured=solid) ----

# Position [m] — use ref_log for desired
ax = axes[0, 0]
ax.set_title('Position [m]')
for c, col, idx, ridx in [('x', 'red', 0, 0), ('y', 'green', 1, 1), ('z', 'blue', 2, 2)]:
    ax.plot(t_sim, x_sim[:, idx], color=col, lw=0.8, label=f'{c} meas')
    ax.plot(t_sim, ref[:, ridx], color=col, ls='--', lw=0.8, label=f'{c} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[m]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Attitude [deg]
ax = axes[0, 1]
ax.set_title('Attitude [deg]')
for c, col, idx in [('roll', 'red', 0), ('pitch', 'green', 1), ('yaw', 'blue', 2)]:
    ax.plot(t_sim, euler_sim[:, idx] * RAD2DEG, color=col, lw=0.8, label=f'{c} meas')
    if f'{c}d' in nhfc.columns:
        ax.plot(nhfc['t'], nhfc[f'{c}d'] * RAD2DEG, color=col, ls='--', lw=0.8, label=f'{c} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[deg]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Linear Velocity [m/s]
ax = axes[0, 2]
ax.set_title('Linear Velocity [m/s]')
for c, col, idx, ridx in [('vx', 'red', 7, 3), ('vy', 'green', 8, 4), ('vz', 'blue', 9, 5)]:
    ax.plot(t_sim, x_sim[:, idx], color=col, lw=0.8, label=f'{c} meas')
    ax.plot(t_sim, ref[:, ridx], color=col, ls='--', lw=0.8, label=f'{c} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[m/s]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Angular Velocity [deg/s]
ax = axes[0, 3]
ax.set_title('Angular Velocity [deg/s]')
for c, col, idx in [('wx', 'red', 10), ('wy', 'green', 11), ('wz', 'blue', 12)]:
    ax.plot(t_sim, x_sim[:, idx] * RAD2DEG, color=col, lw=0.8, label=f'{c} meas')
    if f'{c}d' in nhfc.columns:
        ax.plot(nhfc['t'], nhfc[f'{c}d'] * RAD2DEG, color=col, ls='--', lw=0.8, label=f'{c} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[deg/s]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Linear Acceleration [m/s²]
ax = axes[0, 4]
ax.set_title('Linear Acceleration [m/s²]')
dt_sim = t_sim[1] - t_sim[0]
acc_meas = np.gradient(x_sim[:, 7:10], dt_sim, axis=0)
for c, col, idx, ridx in [('ax', 'red', 0, 6), ('ay', 'green', 1, 7), ('az', 'blue', 2, 8)]:
    ax.plot(t_sim, acc_meas[:, idx], color=col, lw=0.8, label=f'{c} meas')
    ax.plot(t_sim, ref[:, ridx], color=col, ls='--', lw=0.8, label=f'{c} des')
ax.set_xlabel('t [s]'); ax.set_ylabel('[m/s²]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# ---- Row 2: Errors + Forces ----

# Position Error [m]
ax = axes[1, 0]
ax.set_title('Position Error [m]')
for c, col in [('x', 'red'), ('y', 'green'), ('z', 'blue')]:
    if f'e_{c}' in nhfc.columns:
        ax.plot(nhfc['t'], nhfc[f'e_{c}'], color=col, lw=0.8, label=f'e_{c}')
ax.set_xlabel('t [s]'); ax.set_ylabel('[m]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Attitude Error [deg]
ax = axes[1, 1]
ax.set_title('Attitude Error [deg]')
for c, col, n in [('rx', 'red', 'roll'), ('ry', 'green', 'pitch'), ('rz', 'blue', 'yaw')]:
    if f'e_{c}' in nhfc.columns:
        ax.plot(nhfc['t'], nhfc[f'e_{c}'] * RAD2DEG, color=col, lw=0.8, label=f'e_{n}')
ax.set_xlabel('t [s]'); ax.set_ylabel('[deg]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Velocity Error [m/s]
ax = axes[1, 2]
ax.set_title('Velocity Error [m/s]')
for c, col in [('vx', 'red'), ('vy', 'green'), ('vz', 'blue')]:
    if f'e_{c}' in nhfc.columns:
        ax.plot(nhfc['t'], nhfc[f'e_{c}'], color=col, lw=0.8, label=f'e_{c}')
ax.set_xlabel('t [s]'); ax.set_ylabel('[m/s]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Angular Velocity Error [deg/s]
ax = axes[1, 3]
ax.set_title('Ang. Velocity Error [deg/s]')
for c, col in [('wx', 'red'), ('wy', 'green'), ('wz', 'blue')]:
    if f'e_{c}' in nhfc.columns:
        ax.plot(nhfc['t'], nhfc[f'e_{c}'] * RAD2DEG, color=col, lw=0.8, label=f'e_{c}')
ax.set_xlabel('t [s]'); ax.set_ylabel('[deg/s]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Force [N] and Torque [Nm]
ax = axes[1, 4]
ax.set_title('Force [N] & Torque [Nm]')
for c, col in [('fx', 'red'), ('fy', 'green'), ('fz', 'blue')]:
    if c in nhfc.columns:
        ax.plot(nhfc['t'], nhfc[c], color=col, lw=0.8, label=f'{c} [N]')
for c, col in [('tx', 'red'), ('ty', 'green'), ('tz', 'blue')]:
    if c in nhfc.columns:
        ax.plot(nhfc['t'], nhfc[c], color=col, ls='--', lw=0.8, label=f'{c} [Nm]')
ax.set_xlabel('t [s]'); ax.set_ylabel('[N] / [Nm]'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

fig.tight_layout()

# Save
os.makedirs(PLOT_DIR, exist_ok=True)
fig.savefig(os.path.join(PLOT_DIR, f'plot_03b_{ROBOT}.png'), dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {PLOT_DIR}/plot_03b_{ROBOT}.png")

plt.show()
