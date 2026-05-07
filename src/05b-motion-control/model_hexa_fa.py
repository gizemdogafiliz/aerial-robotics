import genomix
import math
import numpy as np
import os
import time
import shutil

# ############################################
#  FULLY-ACTUATED HEXAROTOR (TILTHEX) PARAMETERS
# ############################################

MASS = 2.3
G_ACC = 9.81
J = np.diag([0.0115, 0.0114, 0.0194])
J_INV = np.diag([1/0.0115, 1/0.0114, 1/0.0194])
N_ROTORS = 6
ARMLEN = 0.38998
CF = 9.9016e-4
CT = 1.9e-5
RZ = -1
RX_DEG = -21.2
RY_DEG = -18.7
GROUND_Z = 0.0

OMEGA_MIN = 16.0
OMEGA_MAX = 100.0


def compute_G(n_rotors, armlen, cf, ct, rz, rx_deg=0.0, ry_deg=0.0):
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    G = np.zeros((6, n_rotors))
    sign = 1
    for i in range(n_rotors):
        theta = 2 * np.pi * i / n_rotors

        Rz = np.array([[ np.cos(theta), -np.sin(theta), 0],
                        [ np.sin(theta),  np.cos(theta), 0],
                        [ 0,              0,             1]])
        srx = sign * rx
        Rx = np.array([[1, 0,            0           ],
                        [0, np.cos(srx), -np.sin(srx)],
                        [0, np.sin(srx),  np.cos(srx)]])
        Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                        [ 0,          1, 0          ],
                        [-np.sin(ry), 0, np.cos(ry)]])

        z = (Rz @ Rx @ Ry)[:, 2]
        p = armlen * np.array([np.cos(theta), np.sin(theta), 0.0])

        G[0:3, i] = cf * z
        G[3:6, i] = cf * np.cross(p, z) - sign * rz * ct * z
        sign = -sign
    return G


G_ALLOC = compute_G(N_ROTORS, ARMLEN, CF, CT, RZ, RX_DEG, RY_DEG)
G_ALLOC_INV = np.linalg.inv(G_ALLOC)


def quat_multiply(q, p):
    a1, b1, c1, d1 = q
    a2, b2, c2, d2 = p
    return np.array([
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2,
    ])


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def dynamics(x, u_lambda, G, mass, J, J_inv):
    q = x[3:7]
    v = x[7:10]
    omega = x[10:13]

    R = quat_to_rotmat(q)

    wrench = G @ u_lambda
    f_B = wrench[0:3]
    tau_B = wrench[3:6]

    p_dot = v
    q_dot = 0.5 * quat_multiply(q, np.array([0, omega[0], omega[1], omega[2]]))
    v_dot = np.array([0, 0, -G_ACC]) + (1.0 / mass) * R @ f_B
    omega_dot = J_inv @ (-np.cross(omega, J @ omega) + tau_B)

    return np.concatenate([p_dot, q_dot, v_dot, omega_dot])


def rk4_step(x, u_lambda, dt, G, mass, J, J_inv):
    k1 = dynamics(x, u_lambda, G, mass, J, J_inv)
    k2 = dynamics(x + dt/2 * k1, u_lambda, G, mass, J, J_inv)
    k3 = dynamics(x + dt/2 * k2, u_lambda, G, mass, J, J_inv)
    k4 = dynamics(x + dt * k3, u_lambda, G, mass, J, J_inv)
    x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    x_new[3:7] = quat_normalize(x_new[3:7])
    return x_new, k1


def apply_ground_reaction(x):
    if x[2] <= GROUND_Z:
        x[2] = GROUND_Z
        x[7:10] = 0.0
        x[10:13] = 0.0
    return x


# ############################################
#  FEEDBACK LINEARIZATION CONTROLLER
# ############################################

Kpxy = 3.0;  Kpz = 6.0
Kvxy = 5.0;  Kvz = 8.0
Kqxy = 4.0;  Kqz = 3.0
Kwxy = 5.0;  Kwz = 4.0

K_p = np.diag([Kpxy, Kpxy, Kpz])
K_v = np.diag([Kvxy, Kvxy, Kvz])
K_q = np.diag([Kqxy, Kqxy, Kqz])
K_w = np.diag([Kwxy, Kwxy, Kwz])


_hold_pos = np.zeros(3)
_hold_q = np.array([1.0, 0.0, 0.0, 0.0])


def read_desired():
    global _hold_pos, _hold_q
    raw = maneuver.desired()["desired"]
    if raw is None or raw.get("pos") is None:
        return (_hold_pos.copy(), np.zeros(3), np.zeros(3), _hold_q.copy(), np.zeros(3))
    data = raw
    pos_d = np.array([data["pos"]["x"], data["pos"]["y"], data["pos"]["z"]])
    vel_d = np.array([data["vel"]["vx"], data["vel"]["vy"], data["vel"]["vz"]])
    acc_d = np.array([data["acc"]["ax"], data["acc"]["ay"], data["acc"]["az"]])
    q_d = np.array([data["att"]["qw"], data["att"]["qx"],
                     data["att"]["qy"], data["att"]["qz"]])
    omega_d = np.array([data["avel"]["wx"], data["avel"]["wy"], data["avel"]["wz"]])

    jump = np.linalg.norm(pos_d - _hold_pos)
    if jump > 0.5 and np.linalg.norm(vel_d) < 1e-4 and np.linalg.norm(acc_d) < 1e-4:
        return (_hold_pos.copy(), np.zeros(3), np.zeros(3), _hold_q.copy(), np.zeros(3))

    _hold_pos = pos_d.copy()
    _hold_q = q_d.copy()
    return pos_d, vel_d, acc_d, q_d, omega_d


def fb_linearization(x, pos_d, vel_d, acc_d, q_d, omega_d):
    pos = x[0:3]
    q = x[3:7]
    vel = x[7:10]
    omega = x[10:13]
    R = quat_to_rotmat(q)
    R_d = quat_to_rotmat(q_d)

    # --- position control (world frame) ---
    e_p = pos_d - pos
    e_v = vel_d - vel
    a_des = acc_d + K_v @ e_v + K_p @ e_p

    # desired force in world frame (gravity cancellation)
    f_d_world = MASS * (a_des + np.array([0, 0, G_ACC]))

    # transform to body frame
    f_B = R.T @ f_d_world

    # --- attitude control (body frame) ---
    e_R = 0.5 * vee(R.T @ R_d - R_d.T @ R)
    e_omega = R.T @ R_d @ omega_d - omega

    alpha_des = K_q @ e_R + K_w @ e_omega

    # desired torque (Coriolis cancellation)
    tau_B = J @ alpha_des + np.cross(omega, J @ omega)

    # --- allocation ---
    wrench = np.concatenate([f_B, tau_B])
    u_lambda = G_ALLOC_INV @ wrench

    # --- convert to speeds and saturate ---
    speeds = np.sign(u_lambda) * np.sqrt(np.abs(u_lambda))
    speed_abs = np.abs(speeds)
    speed_abs = np.clip(speed_abs, OMEGA_MIN, OMEGA_MAX)
    speeds = speed_abs * np.sign(speeds)
    u_lambda_sat = speeds * np.abs(speeds)

    return u_lambda_sat, wrench, e_p, e_R


# ############################################
#  GENOMIX SETUP
# ############################################

def setup():
    state_port = nhfc.state('my_state')
    maneuver.connect_port({'local': 'state', 'remote': 'my_state'})
    return state_port


def start():
    maneuver.set_bounds({
        'xmin': -100, 'xmax': 100,
        'ymin': -100, 'ymax': 100,
        'zmin': -100, 'zmax': 100,
        'yawmin': -2*math.pi, 'yawmax': 2*math.pi
    })
    maneuver.set_velocity_limit({'v': 1, 'w': 0.5})
    maneuver.set_acceleration_limit({'a': 0.8, 'dw': 0.5})
    maneuver.set_jerk_limit({'j': 5, 'ddw': 3})
    maneuver.set_snap_limit({'s': 25, 'dddw': 15})

    maneuver.log('/tmp/maneuver.log')
    maneuver.set_current_state()


def stop():
    maneuver.stop()
    maneuver.log_stop()


def state_to_port(state_port, state: np.array):
    def _get_time():
        now = math.modf(time.clock_gettime(time.CLOCK_REALTIME))
        return (int(now[1]), int(now[0]*1e9))

    pstddev = 1e-3
    qstddev = 1e-3
    vstddev = 1e-3
    wstddev = 3e-3
    astddev = 2e-2

    pos_cov = [(pstddev)**2, 0, (pstddev)**2, 0, 0, (pstddev)**2]

    att_cov = [0 for i in range(10)]
    qw = state[3]
    qx = state[4]
    qy = state[5]
    qz = state[6]
    att_cov[0] = (qstddev**2) * (1 - qw*qw)
    att_cov[1] = (qstddev**2) * -qw*qx
    att_cov[2] = (qstddev**2) * (1 - qx*qx)
    att_cov[3] = (qstddev**2) * qw*qy
    att_cov[4] = (qstddev**2) * -qx*qy
    att_cov[5] = (qstddev**2) * (1 - qy*qy)
    att_cov[6] = (qstddev**2) * -qw*qz
    att_cov[7] = (qstddev**2) * -qx*qz
    att_cov[8] = (qstddev**2) * -qy*qz
    att_cov[9] = (qstddev**2) * (1 - qz*qz)

    att_pos_cov = [0 for i in range(4*3)]

    vel_cov = [(vstddev)**2, 0, (vstddev)**2, 0, 0, (vstddev)**2]
    avel_cov = [(wstddev)**2, 0, (wstddev)**2, 0, 0, (wstddev)**2]
    acc_cov = [(astddev)**2, 0, (astddev)**2, 0, 0, (astddev)**2]

    aacc_cov = [0 for i in range(6)]

    now = _get_time()

    data = { "state": {
            "ts" : {"sec": now[0], "nsec": now[1]},
            "intrinsic": False,
            "pos": ({"x": state[0], "y": state[1], "z": state[2]}),
            "att": ({"qw": state[3], "qx": state[4], "qy": state[5], "qz": state[6]}),
            "vel": ({"vx": state[7], "vy": state[8], "vz": state[9]}),
            "avel": ({"wx": state[10], "wy": state[11], "wz": state[12]}),
            "acc": ({"ax": state[13], "ay": state[14], "az": state[15]}),
            "aacc": ({"awx": state[16], "awy": state[17], "awz": state[18]}),
            "pos_cov": ({"cov": pos_cov}),
            "att_cov": ({"cov": att_cov}),
            "att_pos_cov": ({"cov": att_pos_cov}),
            "vel_cov": ({"cov": vel_cov}),
            "avel_cov": ({"cov": avel_cov}),
            "acc_cov": ({"cov": acc_cov}),
            "aacc_cov": ({"cov": aacc_cov})
            }
        }

    if not state_port:
        print("port 'state_port' is not set")
        return
    state_port(data)


def get_time_now_ms():
    return time.clock_gettime_ns(time.CLOCK_REALTIME)*1e-6


################################################################################
g = genomix.connect()

g.rpath(os.environ['HOME'] + '/openrobots/lib/genom/pocolibs/plugins')

nhfc = g.load('nhfc')
maneuver = g.load('maneuver')

state_port = setup()

input("start simulation?")

# ############################
#  INITIALIZE SIMULATION
# ############################

x0 = np.zeros(13)
x0[3] = 1.0
u0 = np.zeros(N_ROTORS)

dt = 1e-3
tf = 40.0

N = math.ceil(tf / dt)
tt = np.linspace(0, tf, N)
x_log = np.zeros((N, x0.shape[0]))
u_log = np.zeros((N, u0.shape[0]))
w_log = np.zeros((N, 6))
ep_log = np.zeros((N, 3))
eR_log = np.zeros((N, 3))
t_log = np.zeros(N)
tc_log = np.zeros(N)

x = x0.copy()
xdot = dynamics(x, u0, G_ALLOC, MASS, J, J_INV)
init_state = np.hstack((x, xdot[7:10], xdot[10:13]))
state_to_port(state_port, init_state)
time.sleep(0.5)

start()

time.sleep(0.5)

# ############################
#  WAYPOINT SEQUENCE
# ############################

wp_idx = 0
waypoints = [
    (2.0,  'goto', (1, 1, 1, 0, 8)),
    (10.0, 'goto', (1, -1, 1, math.pi/2, 8)),
    (18.0, 'goto', (0, 0, 1, 0, 8)),
    (26.0, 'goto', (0, 0, 0, 0, 8)),
]

for i, ts in enumerate(tt):
    if wp_idx < len(waypoints) and ts >= waypoints[wp_idx][0]:
        t_wp, cmd, args = waypoints[wp_idx]
        print(f"[t={ts:.1f}s] maneuver.{cmd}{args}")
        maneuver.goto(*args, ack=True)
        wp_idx += 1

    pos_d, vel_d, acc_d, q_d, omega_d = read_desired()
    u_lambda, wrench, e_p, e_R = fb_linearization(
        x, pos_d, vel_d, acc_d, q_d, omega_d)

    t1 = get_time_now_ms()

    x_new, xdot = rk4_step(x, u_lambda, dt, G_ALLOC, MASS, J, J_INV)
    x_new = apply_ground_reaction(x_new)
    x = x_new

    t_log[i] = ts
    x_log[i, :] = x.reshape(-1)
    u_log[i, :] = u_lambda.reshape(-1)
    w_log[i, :] = wrench.reshape(-1)
    ep_log[i, :] = e_p.reshape(-1)
    eR_log[i, :] = e_R.reshape(-1)

    if (int(ts/dt) % 1000) == 0:
        print(f"t: {ts:.1f}")

    t2 = get_time_now_ms()
    elapsed_ms = t2 - t1
    tc_log[i] = elapsed_ms
    if elapsed_ms > 0:
        time.sleep(elapsed_ms * 1e-3)
    elif elapsed_ms < 0:
        print(f"delay of: {dt - elapsed_ms}ms")

    state_vec = np.hstack((x, xdot[7:10], xdot[10:13]))
    state_to_port(state_port, state_vec)

stop()

# ############################################
#  SAVE DATA
# ############################################
LOG_DIR = '/shared-workspace/logs/05b-motion-control'
os.makedirs(LOG_DIR, exist_ok=True)

np.savez(os.path.join(LOG_DIR, 'simulation_data.npz'),
         t=t_log, x=x_log, u=u_log, w=w_log,
         ep=ep_log, eR=eR_log, tc=tc_log)
print(f"Saved: {LOG_DIR}/simulation_data.npz")

for f in ['maneuver.log']:
    src = '/tmp/' + f
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(LOG_DIR, f))
        print(f"Copied: {src} -> {LOG_DIR}/{f}")

print(f"\nAll data saved to: {LOG_DIR}")
