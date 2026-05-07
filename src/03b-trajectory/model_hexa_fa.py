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
#  QUINTIC POLYNOMIAL TRAJECTORY GENERATOR
# ############################################

def quintic_coeffs(q0, qf, T, v0=0, vf=0, a0=0, af=0):
    dq = qf - (q0 + v0*T + 0.5*a0*T**2)
    dv = vf - (v0 + a0*T)
    da = af - a0

    c0 = q0
    c1 = v0
    c2 = a0 / 2.0
    c3 = 10*dq/T**3 - 4*dv/T**2 + da/(2*T)
    c4 = -15*dq/T**4 + 7*dv/T**3 - da/T**2
    c5 = 6*dq/T**5 - 3*dv/T**4 + da/(2*T**3)

    return np.array([c0, c1, c2, c3, c4, c5])


def eval_quintic(coeffs, t):
    c = coeffs
    pos = c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5
    vel = c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4
    acc = 2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3
    jrk = 6*c[3] + 24*c[4]*t + 60*c[5]*t**2
    snp = 24*c[4] + 120*c[5]*t
    return pos, vel, acc, jrk, snp


class TrajectoryGenerator:
    def __init__(self, waypoints):
        self.segments = []
        for t_start, dur, x, y, z, yaw in waypoints:
            self.segments.append({
                't_start': t_start,
                't_end': t_start + dur,
                'duration': dur,
                'target': np.array([x, y, z, yaw]),
            })

        current = np.array([0.0, 0.0, 0.0, 0.0])
        for seg in self.segments:
            seg['coeffs'] = []
            for i in range(4):
                c = quintic_coeffs(current[i], seg['target'][i], seg['duration'])
                seg['coeffs'].append(c)
            current = seg['target'].copy()

    def evaluate(self, t):
        active = None
        for seg in self.segments:
            if seg['t_start'] <= t < seg['t_end']:
                active = seg
                break

        if active is None:
            if t < self.segments[0]['t_start']:
                target = np.array([0.0, 0.0, 0.0, 0.0])
            else:
                target = self.segments[-1]['target']
            z3 = np.zeros(3)
            return target[:3], z3, z3, z3, z3, target[3], 0.0, 0.0, 0.0, 0.0

        tau = t - active['t_start']
        pos = np.zeros(4)
        vel = np.zeros(4)
        acc = np.zeros(4)
        jrk = np.zeros(4)
        snp = np.zeros(4)
        for i in range(4):
            pos[i], vel[i], acc[i], jrk[i], snp[i] = eval_quintic(active['coeffs'][i], tau)

        return pos[:3], vel[:3], acc[:3], jrk[:3], snp[:3], pos[3], vel[3], acc[3], jrk[3], snp[3]


def yaw_to_quat(yaw):
    return np.array([math.cos(yaw/2), 0.0, 0.0, math.sin(yaw/2)])


# --- setup ----------------------------------------------------------------
def setup():
    my_state_port = nhfc.state('my_state')
    my_ref_port = nhfc.reference('my_ref')

    nhfc.connect_port({'local': 'state', 'remote': 'my_state'})
    nhfc.connect_port({'local': 'reference', 'remote': 'my_ref'})

    return my_state_port, my_ref_port


# --- start ----------------------------------------------------------------
def start():
    nhfc.set_gtmrp_geom({
        'rotors': N_ROTORS, 'cx': 0, 'cy': 0, 'cz': 0,
        'armlen': ARMLEN, 'mass': MASS,
        'rx': RX_DEG, 'ry': RY_DEG, 'rz': RZ, 'cf': CF, 'ct': CT
    })
    nhfc.set_emerg({'emerg': {
        'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3
    }})
    nhfc.set_saturation({'sat': {'x': 2, 'v': 2, 'ix': 0}})
    nhfc.set_servo_gain({'gain': {
        'Kpxy': 10, 'Kpz': 15, 'Kqxy': 4, 'Kqz': 0.25,
        'Kvxy': 20, 'Kvz': 10, 'Kwxy': 2, 'Kwz': 0.25,
        'Kixy': 0.01, 'Kiz': 0.03
    }})
    nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'})

    nhfc.log('/tmp/nhfc.log')

    nhfc.set_current_position()
    nhfc.servo(ack=True)


# --- stop -----------------------------------------------------------------
def stop():
    nhfc.stop()
    nhfc.log_stop()


# --- state_to_nhfc --------------------------------------------------------
def state_to_nhfc(state_port, state: np.array):
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
        print("port 'sim_state_port' is not set")
        return
    state_port(data)


# --- ref_to_nhfc ----------------------------------------------------------
def ref_to_nhfc(ref_port, pos_xyz, vel_xyz, acc_xyz, jrk_xyz, snp_xyz,
                yaw, yaw_dot, yaw_ddot, yaw_jrk, yaw_snp):
    def _get_time():
        now = math.modf(time.clock_gettime(time.CLOCK_REALTIME))
        return (int(now[1]), int(now[0]*1e9))

    now = _get_time()
    q = yaw_to_quat(yaw)

    data = { "reference": {
        "ts": {"sec": now[0], "nsec": now[1]},
        "intrinsic": False,
        "pos": {"x": float(pos_xyz[0]), "y": float(pos_xyz[1]), "z": float(pos_xyz[2])},
        "att": {"qw": float(q[0]), "qx": float(q[1]), "qy": float(q[2]), "qz": float(q[3])},
        "vel": {"vx": float(vel_xyz[0]), "vy": float(vel_xyz[1]), "vz": float(vel_xyz[2])},
        "avel": {"wx": 0.0, "wy": 0.0, "wz": float(yaw_dot)},
        "acc": {"ax": float(acc_xyz[0]), "ay": float(acc_xyz[1]), "az": float(acc_xyz[2])},
        "aacc": {"awx": 0.0, "awy": 0.0, "awz": float(yaw_ddot)},
        "jerk": {"jx": float(jrk_xyz[0]), "jy": float(jrk_xyz[1]), "jz": float(jrk_xyz[2])},
        "snap": {"sx": float(snp_xyz[0]), "sy": float(snp_xyz[1]), "sz": float(snp_xyz[2])},
    }}

    ref_port(data)


# --- rotor_speeds_from_nhfc -----------------------------------------------
def rotor_speeds_from_nhfc(c_f, n_act=6):
    desired_speeds = np.zeros(n_act)

    data = nhfc.rotor_input()["rotor_input"]
    for i, s in enumerate(data["desired"]):
        if s:
            desired_speeds[i] = data["desired"][i]
    return desired_speeds


# --- get_time_now_ms ------------------------------------------------------
def get_time_now_ms():
    return time.clock_gettime_ns(time.CLOCK_REALTIME)*1e-6


################################################################################
g = genomix.connect()

g.rpath(os.environ['HOME'] + '/openrobots/lib/genom/pocolibs/plugins')

nhfc = g.load('nhfc')

state_port, ref_port = setup()

input("start simulation?")

# ############################
#  INITIALIZE SIMULATION
# ############################

x0 = np.zeros(13)
x0[3] = 1.0
u0 = np.zeros(N_ROTORS)

dt = 1e-3
tf = 27.0

N = math.ceil(tf / dt)
tt = np.linspace(0, tf, N)
x_log = np.zeros((N, x0.shape[0]))
u_log = np.zeros((N, u0.shape[0]))
t_log = np.zeros(N)
tc_log = np.zeros(N)

ref_log = np.zeros((N, 10))  # x,y,z, vx,vy,vz, ax,ay,az, yaw

# ############################
#  TRAJECTORY DEFINITION
# ############################
# Same waypoints as 03a hexa-fa:
#   (t_start, duration, x, y, z, yaw)
traj = TrajectoryGenerator([
    (2.0,  5.0,  1,  1,  1, 0),
    (7.0,  5.0,  1, -1,  1, math.pi/2),
    (12.0, 5.0,  0,  0,  1, 0),
    (17.0, 5.0,  0,  0,  0, 0),
])

x = x0.copy()
xdot = dynamics(x, u0, G_ALLOC, MASS, J, J_INV)
init_state = np.hstack((x, xdot[7:10], xdot[10:13]))
state_to_nhfc(state_port, init_state)

pos0, vel0, acc0, jrk0, snp0, yaw0, yd0, ydd0, yj0, ys0 = traj.evaluate(0)
ref_to_nhfc(ref_port, pos0, vel0, acc0, jrk0, snp0, yaw0, yd0, ydd0, yj0, ys0)

start()

time.sleep(0.5)

# ############################
#  SIMULATION LOOP
# ############################
for i, ts in enumerate(tt):
    pos_d, vel_d, acc_d, jrk_d, snp_d, yaw_d, yaw_dot_d, yaw_ddot_d, yaw_jrk_d, yaw_snp_d = traj.evaluate(ts)

    ref_to_nhfc(ref_port, pos_d, vel_d, acc_d, jrk_d, snp_d, yaw_d, yaw_dot_d, yaw_ddot_d, yaw_jrk_d, yaw_snp_d)

    speeds = rotor_speeds_from_nhfc(CF)
    u_lambda = speeds * np.abs(speeds)
    t1 = get_time_now_ms()

    x_new, xdot = rk4_step(x, u_lambda, dt, G_ALLOC, MASS, J, J_INV)
    x_new = apply_ground_reaction(x_new)
    x = x_new

    t_log[i] = ts
    x_log[i, :] = x.reshape(-1)
    u_log[i, :] = u_lambda.reshape(-1)
    ref_log[i, :] = np.array([
        pos_d[0], pos_d[1], pos_d[2],
        vel_d[0], vel_d[1], vel_d[2],
        acc_d[0], acc_d[1], acc_d[2],
        yaw_d
    ])

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
    state_to_nhfc(state_port, state_vec)

stop()

# ############################################
#  SAVE DATA
# ############################################
LOG_DIR = '/shared-workspace/logs/03b-trajectory/hexa-fa'
os.makedirs(LOG_DIR, exist_ok=True)

np.savez(os.path.join(LOG_DIR, 'simulation_data.npz'),
         t=t_log, x=x_log, u=u_log, tc=tc_log, ref=ref_log)
print(f"Saved: {LOG_DIR}/simulation_data.npz")

for f in ['nhfc.log']:
    src = '/tmp/' + f
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(LOG_DIR, f))
        print(f"Copied: {src} -> {LOG_DIR}/{f}")

ref_logfile = os.path.join(LOG_DIR, 'trajectory.log')
with open(ref_logfile, 'w') as fout:
    fout.write("ts x y z yaw vx vy vz wz ax ay az\n")
    for i in range(N):
        fout.write(f"{t_log[i]:.6f}  "
                   f"{ref_log[i,0]:.6f}  {ref_log[i,1]:.6f}  {ref_log[i,2]:.6f}  "
                   f"{ref_log[i,9]:.6f}  "
                   f"{ref_log[i,3]:.6f}  {ref_log[i,4]:.6f}  {ref_log[i,5]:.6f}  "
                   f"0  "
                   f"{ref_log[i,6]:.6f}  {ref_log[i,7]:.6f}  {ref_log[i,8]:.6f}\n")
print(f"Saved: {ref_logfile}")

print(f"\nAll data saved to: {LOG_DIR}")
