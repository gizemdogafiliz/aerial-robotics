"""
Assignment 06a: Physical Interaction — Contact Inspection (Gazebo)
Plots: tracking, wrench, contact forces, nominal vs filtered trajectory, EE position.

Reads telekyb3 component logs (pom, uavpos, uavatt, maneuver) and the
simulation_data.npz produced by model_hexa_fa.py.

Usage:
  python3 plot_06a.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, 'plots_06a', 'gazebo')
LOG_DIR = os.path.expanduser('~/tk3lab-ws/logs/06a-physical-interaction/gazebo')

RAD2DEG = 180.0 / math.pi
X_WALL = 2.0
L_BAR = 0.6
TIP_RADIUS = 0.05  # sphere collision radius — adds to effective EE reach
L_EFF = L_BAR + TIP_RADIUS  # tip surface contacts wall, not tip center
K_WALL = 500.0


def quat_to_euler(q):
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])


print(f"Loading data from: {LOG_DIR}")


def parse_pom_log(filepath):
    """Parse pom.log column-format file → dict of arrays."""
    header = None
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            if header is None:
                header = line.split()
                continue
            parts = line.split()
            if len(parts) == len(header):
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    continue
    arr = np.array(rows)
    return {col: arr[:, i] for i, col in enumerate(header)}


# Reconstruct everything from telekyb3 component logs.
sim = np.load(os.path.join(LOG_DIR, 'simulation_data.npz')) \
    if os.path.exists(os.path.join(LOG_DIR, 'simulation_data.npz')) else {}

pom_log = os.path.join(LOG_DIR, 'pom.log')
print("Parsing pom.log (state) ...")
pom = parse_pom_log(pom_log)
ts = pom['ts']
t = ts - ts[0]
pos = np.column_stack([pom['x'], pom['y'], pom['z']])
roll, pitch, yaw = pom['roll'], pom['pitch'], pom['yaw']
# rebuild quaternion from euler (ZYX convention)
cy = np.cos(yaw*0.5);   sy = np.sin(yaw*0.5)
cp = np.cos(pitch*0.5); sp = np.sin(pitch*0.5)
cr = np.cos(roll*0.5);  sr = np.sin(roll*0.5)
att = np.column_stack([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                        cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])
vel = np.column_stack([pom['vx'], pom['vy'], pom['vz']])
avel = np.column_stack([pom['wx'], pom['wy'], pom['wz']])
print(f"  {len(t)} pom samples, duration {t[-1]:.1f}s")

x = np.hstack([pos, att, vel, avel])

# EE tip surface position: pos + R(quat) @ [L_EFF, 0, 0]
qw, qx, qy, qz = att[:, 0], att[:, 1], att[:, 2], att[:, 3]
r11 = 1 - 2*(qy*qy + qz*qz)
r21 = 2*(qx*qy + qz*qw)
r31 = 2*(qx*qz - qy*qw)
p_ee = np.column_stack([
    pos[:, 0] + L_EFF * r11,
    pos[:, 1] + L_EFF * r21,
    pos[:, 2] + L_EFF * r31,
])


def _interp_log_to_pom(log_dict, cols, dim=3):
    if log_dict is None:
        return None
    log_t = log_dict['ts'] - ts[0]
    return np.column_stack([np.interp(t, log_t, log_dict[c]) for c in cols[:dim]])


# uavpos.log → pos_d, vel_d, position error, controller force
uavpos_log = os.path.join(LOG_DIR, 'uavpos.log')
if os.path.exists(uavpos_log):
    print("Parsing uavpos.log ...")
    up = parse_pom_log(uavpos_log)
    pos_d = _interp_log_to_pom(up, ['xd', 'yd', 'zd'])
    vel_d = _interp_log_to_pom(up, ['vxd', 'vyd', 'vzd'])
    ep = _interp_log_to_pom(up, ['e_x', 'e_y', 'e_z'])
    f_ctrl = _interp_log_to_pom(up, ['fx', 'fy', 'fz'])
else:
    pos_d = vel_d = ep = f_ctrl = None

# uavatt.log → attitude desired/error, controller torque
uavatt_log = os.path.join(LOG_DIR, 'uavatt.log')
if os.path.exists(uavatt_log):
    print("Parsing uavatt.log ...")
    ua = parse_pom_log(uavatt_log)
    rolld_arr = _interp_log_to_pom(ua, ['roll', 'pitch', 'yaw'])
    rd, pd_, yd = rolld_arr[:, 0], rolld_arr[:, 1], rolld_arr[:, 2]
    cy_ = np.cos(yd*0.5);  sy_ = np.sin(yd*0.5)
    cp_ = np.cos(pd_*0.5); sp_ = np.sin(pd_*0.5)
    cr_ = np.cos(rd*0.5);  sr_ = np.sin(rd*0.5)
    q_d = np.column_stack([cr_*cp_*cy_ + sr_*sp_*sy_, sr_*cp_*cy_ - cr_*sp_*sy_,
                            cr_*sp_*cy_ + sr_*cp_*sy_, cr_*cp_*sy_ - sr_*sp_*cy_])
    omega_d = _interp_log_to_pom(ua, ['wx', 'wy', 'wz'])
    eR = _interp_log_to_pom(ua, ['e_rx', 'e_ry', 'e_rz'])
    tau_ctrl = _interp_log_to_pom(ua, ['tx', 'ty', 'tz'])
else:
    q_d = omega_d = eR = tau_ctrl = None

# combine controller force + torque into 6D wrench (figure 4)
w = np.hstack([f_ctrl, tau_ctrl]) if (f_ctrl is not None and tau_ctrl is not None) else None

# maneuver.log → nominal trajectory (before AF filter)
maneuver_log = os.path.join(LOG_DIR, 'maneuver.log')
if os.path.exists(maneuver_log):
    print("Parsing maneuver.log ...")
    mv = parse_pom_log(maneuver_log)
    p_nom = _interp_log_to_pom(mv, ['x', 'y', 'z'])
else:
    p_nom = None

# Contact force fallback (phynt.log empty): idealised spring from penetration
penetration = np.maximum(p_ee[:, 0] - X_WALL, 0.0)
f_normal = K_WALL * penetration
f_contact = np.column_stack([-f_normal, np.zeros_like(f_normal), np.zeros_like(f_normal)])

u = None

euler = np.array([quat_to_euler(x[i, 3:7]) for i in range(len(t))])
euler_d = np.array([quat_to_euler(q_d[i]) for i in range(len(t))]) if q_d is not None else None
print(f"Duration: {t[-1]:.1f}s, {len(t)} samples")

# ============================================================
#  FIGURE 1: Trajectory Tracking (3x2)
# ============================================================
fig1, axes1 = plt.subplots(3, 2, figsize=(14, 10))
fig1.suptitle('06a Physical Interaction — Trajectory Tracking', fontsize=13)

state_groups = [
    (0, 0, 'Position [m]',            ['x', 'y', 'z'],    [0, 1, 2],   1.0),
    (0, 1, 'Attitude [deg]',          ['roll', 'pitch', 'yaw'], None,   RAD2DEG),
    (1, 0, 'Linear Velocity [m/s]',   ['vx', 'vy', 'vz'], [7, 8, 9],   1.0),
    (1, 1, 'Angular Velocity [deg/s]', ['wx', 'wy', 'wz'], [10, 11, 12], RAD2DEG),
]

colors = {'x': 'red', 'y': 'green', 'z': 'blue',
          'roll': 'red', 'pitch': 'green', 'yaw': 'blue',
          'vx': 'red', 'vy': 'green', 'vz': 'blue',
          'wx': 'red', 'wy': 'green', 'wz': 'blue'}

des_pos_idx = {'x': 0, 'y': 1, 'z': 2}
des_vel_idx = {'vx': 0, 'vy': 1, 'vz': 2}
des_att_labels = ['roll', 'pitch', 'yaw']
des_avel_idx = {'wx': 0, 'wy': 1, 'wz': 2}

for row, col, title_str, labels, sim_idx, scale in state_groups:
    ax = axes1[row, col]
    ax.set_title(title_str)
    for j, lbl in enumerate(labels):
        c = colors[lbl]
        if sim_idx is not None:
            ax.plot(t, x[:, sim_idx[j]] * scale, color=c, lw=0.8, label=lbl)
        else:
            ax.plot(t, euler[:, j] * scale, color=c, lw=0.8, label=lbl)

        if row == 0 and col == 0 and pos_d is not None and lbl in des_pos_idx:
            ax.plot(t, pos_d[:, des_pos_idx[lbl]] * scale, color=c, ls='--', lw=0.5, alpha=0.6, label=f'{lbl}_d')
        if row == 0 and col == 1 and euler_d is not None and lbl in des_att_labels:
            ax.plot(t, euler_d[:, j] * scale, color=c, ls='--', lw=0.5, alpha=0.6, label=f'{lbl}_d')
        if row == 1 and col == 0 and vel_d is not None and lbl in des_vel_idx:
            ax.plot(t, vel_d[:, des_vel_idx[lbl]] * scale, color=c, ls='--', lw=0.5, alpha=0.6, label=f'{lbl}_d')
        if row == 1 and col == 1 and omega_d is not None and lbl in des_avel_idx:
            ax.plot(t, omega_d[:, des_avel_idx[lbl]] * scale, color=c, ls='--', lw=0.5, alpha=0.6, label=f'{lbl}_d')
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Position error: from logged ep, or fallback to pos_d - pos
ax = axes1[2, 0]
ax.set_title('Position Error [m]')
if ep is not None:
    err_p = ep
elif pos_d is not None:
    err_p = pos_d - x[:, 0:3]
else:
    err_p = None
if err_p is not None:
    for j, (lbl, c) in enumerate([('e_x', 'red'), ('e_y', 'green'), ('e_z', 'blue')]):
        ax.plot(t, err_p[:, j], color=c, lw=0.8, label=lbl)
ax.set_xlabel('t [s]')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Attitude error: from logged eR, or fallback to euler_d - euler
ax = axes1[2, 1]
ax.set_title('Attitude Error [deg]')
if eR is not None:
    err_R = eR * RAD2DEG
elif euler_d is not None:
    err_R = (euler_d - euler) * RAD2DEG
else:
    err_R = None
if err_R is not None:
    for j, (lbl, c) in enumerate([('e_roll', 'red'), ('e_pitch', 'green'), ('e_yaw', 'blue')]):
        ax.plot(t, err_R[:, j], color=c, lw=0.8, label=lbl)
ax.set_xlabel('t [s]')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

fig1.tight_layout()

# ============================================================
#  FIGURE 2: Contact Forces and End-Effector Position (3x1)
# ============================================================
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 9))
fig2.suptitle('06a Physical Interaction — Contact & End-Effector', fontsize=13)

F_REQ = 2.0
ax = axes2[0]
ax.set_title('Contact Force [N] (idealized spring from EE penetration)')
if f_contact is not None:
    ax.plot(t, -f_contact[:, 0], 'r', lw=0.8, label='|Fx| (normal)')
    ax.plot(t, np.abs(f_contact[:, 1]), 'g', lw=0.8, label='|Fy| (friction)')
    ax.plot(t, np.abs(f_contact[:, 2]), 'b', lw=0.8, label='|Fz| (friction)')
    ax.axhline(F_REQ, color='k', ls='--', lw=0.8, label=f'req ≥{F_REQ}N')
    ax.fill_between(t, F_REQ, -f_contact[:, 0],
                    where=(-f_contact[:, 0]) >= F_REQ, alpha=0.15, color='green', label='meets req')
ax.set_ylim(bottom=-0.5)
ax.set_xlabel('t [s]')
ax.set_ylabel('[N]')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes2[1]
ax.set_title('End-Effector Position [m]')
ax.plot(t, p_ee[:, 0], 'r', lw=0.8, label='EE_x')
ax.plot(t, p_ee[:, 1], 'g', lw=0.8, label='EE_y')
ax.plot(t, p_ee[:, 2], 'b', lw=0.8, label='EE_z')
if pos_d is not None:
    ax.plot(t, pos_d[:, 0] + L_BAR, 'r--', lw=0.5, alpha=0.6, label='EE_x_d')
    ax.plot(t, pos_d[:, 1], 'g--', lw=0.5, alpha=0.6, label='EE_y_d')
    ax.plot(t, pos_d[:, 2], 'b--', lw=0.5, alpha=0.6, label='EE_z_d')
ax.axhline(X_WALL, color='k', ls='--', lw=0.8, label=f'wall x={X_WALL}')
ax.set_xlabel('t [s]')
ax.set_ylabel('[m]')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes2[2]
penetration_mm = (p_ee[:, 0] - X_WALL) * 1000.0
penetration_mm = np.clip(penetration_mm, 0, None)
pen_req_mm = F_REQ / K_WALL * 1000.0
ax.set_title('EE Penetration into Wall [mm]')
ax.plot(t, penetration_mm, 'r', lw=0.8, label='penetration')
ax.axhline(pen_req_mm, color='k', ls='--', lw=0.8, label=f'≥{F_REQ}N → ≥{pen_req_mm:.1f}mm')
ax.fill_between(t, pen_req_mm, penetration_mm,
                where=penetration_mm >= pen_req_mm, alpha=0.15, color='green', label='meets req')
ax.set_xlabel('t [s]')
ax.set_ylabel('[mm]')
ax.legend()
ax.grid(True, alpha=0.3)

fig2.tight_layout()

# ============================================================
#  FIGURE 3: Nominal vs Filtered Trajectory (2x2)
# ============================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
fig3.suptitle('06a Nominal vs Admittance-Filtered Trajectory', fontsize=13)

has_nominal = np.any(np.abs(p_nom) > 1e-6)

has_desired = pos_d is not None and np.any(np.abs(pos_d) > 1e-6)

ax = axes3[0, 0]
ax.set_title('X Position [m]')
ax.plot(t, x[:, 0], 'r', lw=0.8, label='actual')
if has_nominal:
    ax.plot(t, p_nom[:, 0], 'r--', lw=0.6, alpha=0.7, label='nominal')
if has_desired:
    ax.plot(t, pos_d[:, 0], 'r:', lw=0.6, alpha=0.5, label='desired (AF)')
ax.axhline(X_WALL - L_BAR, color='k', ls=':', lw=0.5, label=f'body at wall ({X_WALL-L_BAR})')
ax.set_xlabel('t [s]')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes3[0, 1]
ax.set_title('Y Position [m]')
ax.plot(t, x[:, 1], 'g', lw=0.8, label='actual')
if has_nominal:
    ax.plot(t, p_nom[:, 1], 'g--', lw=0.6, alpha=0.7, label='nominal')
if has_desired:
    ax.plot(t, pos_d[:, 1], 'g:', lw=0.6, alpha=0.5, label='desired (AF)')
ax.set_xlabel('t [s]')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes3[1, 0]
ax.set_title('Z Position [m]')
ax.plot(t, x[:, 2], 'b', lw=0.8, label='actual')
if has_nominal:
    ax.plot(t, p_nom[:, 2], 'b--', lw=0.6, alpha=0.7, label='nominal')
if has_desired:
    ax.plot(t, pos_d[:, 2], 'b:', lw=0.6, alpha=0.5, label='desired (AF)')
ax.set_xlabel('t [s]')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes3[1, 1]
ax.set_title('3D Trajectory (YZ wall-plane)')
ax.plot(x[:, 1], x[:, 2], 'k', lw=0.8, label='actual path')
if has_nominal:
    ax.plot(p_nom[:, 1], p_nom[:, 2], 'b--', lw=0.6, alpha=0.7, label='nominal path')
if has_desired:
    ax.plot(pos_d[:, 1], pos_d[:, 2], 'r:', lw=0.6, alpha=0.5, label='desired (AF)')
ax.set_xlabel('Y [m]')
ax.set_ylabel('Z [m]')
ax.set_aspect('equal')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

fig3.tight_layout()

# ============================================================
#  FIGURE 4: External Wrench Estimate
# ============================================================
fig4, axes4 = plt.subplots(2, 1, figsize=(12, 7))
fig4.suptitle('06a External Wrench Estimate (idealized r × F from EE penetration)', fontsize=13)

f_ext = f_contact
if f_contact is not None:
    # tau_world = r_world × F_world, r_world = R @ [L_EFF, 0, 0]
    r_w_x = L_EFF * r11
    r_w_y = L_EFF * r21
    r_w_z = L_EFF * r31
    F_w_x = f_contact[:, 0]
    F_w_y = f_contact[:, 1]
    F_w_z = f_contact[:, 2]
    tau_ext = np.column_stack([
        r_w_y * F_w_z - r_w_z * F_w_y,
        r_w_z * F_w_x - r_w_x * F_w_z,
        r_w_x * F_w_y - r_w_y * F_w_x,
    ])
else:
    tau_ext = None

ax = axes4[0]
ax.set_title('Estimated External Force [N]')
if f_ext is not None:
    for j, (lbl, c) in enumerate([('fx', 'red'), ('fy', 'green'), ('fz', 'blue')]):
        ax.plot(t, f_ext[:, j], color=c, lw=0.8, label=lbl)
ax.set_xlabel('t [s]'); ax.set_ylabel('[N]'); ax.legend(); ax.grid(True, alpha=0.3)
ax = axes4[1]
ax.set_title('Estimated External Torque [Nm]')
if tau_ext is not None:
    for j, (lbl, c) in enumerate([('tx', 'red'), ('ty', 'green'), ('tz', 'blue')]):
        ax.plot(t, tau_ext[:, j], color=c, lw=0.8, label=lbl)
ax.set_xlabel('t [s]'); ax.set_ylabel('[Nm]'); ax.legend(); ax.grid(True, alpha=0.3)

fig4.tight_layout()

# ============================================================
#  SAVE
# ============================================================
os.makedirs(PLOT_DIR, exist_ok=True)

fig1.savefig(os.path.join(PLOT_DIR, 'tracking.png'), dpi=150, bbox_inches='tight')
fig2.savefig(os.path.join(PLOT_DIR, 'contact.png'), dpi=150, bbox_inches='tight')
fig3.savefig(os.path.join(PLOT_DIR, 'nominal_vs_filtered.png'), dpi=150, bbox_inches='tight')
fig4.savefig(os.path.join(PLOT_DIR, 'wrench.png'), dpi=150, bbox_inches='tight')
print(f"\nPlots saved to: {PLOT_DIR}/")

plt.show()
