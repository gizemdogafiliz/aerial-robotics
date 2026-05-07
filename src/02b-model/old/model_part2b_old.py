# Copyright (c) 2026 IRISA/CNRS-INRIA
# All rights reserved.
#
# Redistribution  and  use  in  source  and binary  forms,  with  or  without
# modification, are permitted provided that the following conditions are met:
#
#   1. Redistributions of  source  code must retain the  above copyright
#      notice and this list of conditions.
#   2. Redistributions in binary form must reproduce the above copyright
#      notice and  this list of  conditions in the  documentation and/or
#      other materials provided with the distribution.
#
# THE SOFTWARE  IS PROVIDED "AS IS"  AND THE AUTHOR  DISCLAIMS ALL WARRANTIES
# WITH  REGARD   TO  THIS  SOFTWARE  INCLUDING  ALL   IMPLIED  WARRANTIES  OF
# MERCHANTABILITY AND  FITNESS.  IN NO EVENT  SHALL THE AUTHOR  BE LIABLE FOR
# ANY  SPECIAL, DIRECT,  INDIRECT, OR  CONSEQUENTIAL DAMAGES  OR  ANY DAMAGES
# WHATSOEVER  RESULTING FROM  LOSS OF  USE, DATA  OR PROFITS,  WHETHER  IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR  OTHER TORTIOUS ACTION, ARISING OUT OF OR
# IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
#                                         Gianluca Corsini on Thu Apr 16 2026

# ############################################
#  PLACE HERE YOUR NECESSARY FUNCTION IMPORTS
# ############################################

import genomix
import math
import numpy as np
import os
import time

# --- Quadrotor physical parameters ---
MASS = 1.28
G_ACC = 9.81
J = np.diag([0.015, 0.015, 0.007])
J_INV = np.diag([1/0.015, 1/0.015, 1/0.007])
N_ROTORS = 4
ARMLEN = 0.23
CF = 6.5e-4
CT = 1e-5
RZ = -1
GROUND_Z = 0.0

# --- Allocation matrix ---
def compute_G(n_rotors, armlen, cf, ct, rz):
    G = np.zeros((6, n_rotors))
    sign = 1
    for i in range(n_rotors):
        theta = 2 * np.pi * i / n_rotors
        z = np.array([0.0, 0.0, 1.0])
        p = armlen * np.array([np.cos(theta), np.sin(theta), 0.0])
        G[0:3, i] = cf * z
        G[3:6, i] = cf * np.cross(p, z) - sign * rz * ct * z
        sign = -sign
    return G

# --- Quaternion utilities ---
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

# --- Dynamics ---
def dynamics(x, w_squared, G, mass, J, J_inv):
    q = x[3:7]
    v = x[7:10]
    omega = x[10:13]
    R = quat_to_rotmat(q)
    wrench = G @ w_squared
    f_B = wrench[0:3]
    tau_B = wrench[3:6]
    p_dot = v
    q_dot = 0.5 * quat_multiply(q, np.array([0, omega[0], omega[1], omega[2]]))
    v_dot = np.array([0, 0, -G_ACC]) + (1.0 / mass) * R @ f_B
    omega_dot = J_inv @ (-np.cross(omega, J @ omega) + tau_B)
    return np.concatenate([p_dot, q_dot, v_dot, omega_dot])

# --- RK4 integrator ---
def rk4_step(x, w_squared, dt, G, mass, J, J_inv):
    k1 = dynamics(x, w_squared, G, mass, J, J_inv)
    k2 = dynamics(x + dt/2 * k1, w_squared, G, mass, J, J_inv)
    k3 = dynamics(x + dt/2 * k2, w_squared, G, mass, J, J_inv)
    k4 = dynamics(x + dt * k3, w_squared, G, mass, J, J_inv)
    x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    x_new[3:7] = quat_normalize(x_new[3:7])
    return x_new, k1

# --- Ground reaction ---
def apply_ground_reaction(x, ground_z=GROUND_Z):
    if x[2] <= ground_z:
        x[2] = ground_z
        x[7:10] = 0.0
        x[10:13] = 0.0
    return x

G_ALLOC = compute_G(N_ROTORS, ARMLEN, CF, CT, RZ)

LOG_DIR = '/home/gizemd/tk3lab-ws/src/02b-model/logs_part2b_old'

# --- setup ----------------------------------------------------------------
def setup():
    # ###############################################
    #  PLACE HERE YOUR NECESSARY SETUP LINES OF CODE
    # ###############################################

    # configure quadrotor geometry (same as Gazebo simulation)
    nhfc.set_gtmrp_geom({
        'rotors': N_ROTORS, 'cx': 0, 'cy': 0, 'cz': 0,
        'armlen': ARMLEN, 'mass': MASS,
        'rx': 0, 'ry': 0, 'rz': RZ,
        'cf': CF, 'ct': CT
    })

    # emergency descent parameters
    nhfc.set_emerg({'emerg': {
        'descent': 0.1, 'dx': 4, 'dq': 4, 'dv': 10, 'dw': 10
    }})

    # PID gains
    nhfc.set_saturation({'sat': {'x': 1, 'v': 1, 'ix': 0}})
    nhfc.set_servo_gain({'gain': {
        'Kpxy': 6, 'Kpz': 5, 'Kqxy': 4, 'Kqz': 0.1,
        'Kvxy': 6, 'Kvz': 6, 'Kwxy': 1, 'Kwz': 0.4,
        'Kixy': 0, 'Kiz': 0
    }})

    # tilt-prioritized controller
    nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'})

    my_state_port = nhfc.state('my_state')

    # connect nhfc to state port which will be updated by our simulator
    nhfc.connect_port({ 'local': 'state', 'remote': 'my_state' })

    return my_state_port

# --- start ----------------------------------------------------------------
def start():
    # ########################################
    #  FILL THIS FUNCTION, THEN REMOVE 'pass'
    # ########################################
    os.makedirs(LOG_DIR, exist_ok=True)
    nhfc.log(LOG_DIR + '/nhfc.log')


# --- stop -----------------------------------------------------------------
def stop():
    # ########################################
    #  FILL THIS FUNCTION, THEN REMOVE 'pass'
    # ########################################
    nhfc.stop()
    nhfc.log_stop()

# --- state_to_nhfc --------------------------------------------------------
def state_to_nhfc(state_port, state: np.array):
    def _get_time():
        # returns a tuple of the type (<sec>,<nsec>)
        now = math.modf(time.clock_gettime(time.CLOCK_REALTIME))
        return (int(now[1]), int(now[0]*1e9))

    # https://git.openrobots.org/projects/libmrsim/repository/libmrsim/revisions/master/entry/src/sim.c#L66
    pstddev = 1e-3
    qstddev = 1e-3
    vstddev = 1e-3
    wstddev = 3e-3
    astddev = 2e-2

    """ covariances """
    # Covariances documentation at
    # https://git.openrobots.org/projects/openrobots-idl/repository/openrobots-idl/revisions/master/entry/pose/t3d.idl#L41

    pos_cov = [(pstddev)**2, 0, (pstddev)**2, 0, 0, (pstddev)**2]

    att_cov = [0 for i in range(10)]
    # https://git.openrobots.org/projects/mrsim-genom3/repository/mrsim-genom3/revisions/master/entry/codels/sim.c#L52
    qw = state[3]
    qx = state[4]
    qy = state[5]
    qz = state[6]
    att_cov[0] = (qstddev**2) * (1 - qw*qw);
    att_cov[1] = (qstddev**2) * -qw*qx;
    att_cov[2] = (qstddev**2) * (1 - qx*qx);
    att_cov[3] = (qstddev**2) * qw*qy;
    att_cov[4] = (qstddev**2) * -qx*qy;
    att_cov[5] = (qstddev**2) * (1 - qy*qy);
    att_cov[6] = (qstddev**2) * -qw*qz;
    att_cov[7] = (qstddev**2) * -qx*qz;
    att_cov[8] = (qstddev**2) * -qy*qz;
    att_cov[9] = (qstddev**2) * (1 - qz*qz);

    att_pos_cov = [0 for i in range(4*3)]

    vel_cov = [(vstddev)**2, 0, (vstddev)**2, 0, 0, (vstddev)**2]
    avel_cov = [(wstddev)**2, 0, (wstddev)**2, 0, 0, (wstddev)**2]
    acc_cov = [(astddev)**2, 0, (astddev)**2, 0, 0, (astddev)**2]

    aacc_cov = [0 for i in range(6)]

    now = _get_time()

    # port and message descriptions at
    # https://git.openrobots.org/projects/openrobots-idl/repository/openrobots-idl/revisions/master/entry/pose/pose_estimator.gen
    # https://git.openrobots.org/projects/openrobots-idl/repository/openrobots-idl/revisions/master/entry/pose/t3d.idl
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

# --- rotor_speeds_from_nhfc -----------------------------------------------
def rotor_speeds_from_nhfc(c_f, n_act=4):
    desired_speeds = np.zeros(n_act)

    data = nhfc.rotor_input()["rotor_input"]
    # sometimes None may appear
    # if that happens for i-th rotor, then skip its speed
    for i, s in enumerate(data["desired"]):
        if s:
            desired_speeds[i] = data["desired"][i]
    return desired_speeds

# --- speed_to_thrust ------------------------------------------------------
def speed_to_thrust(speed: np.array, c_f):
    return np.square(speed) * c_f

# --- get_time_now_ms ------------------------------------------------------
def get_time_now_ms():
    return time.clock_gettime_ns(time.CLOCK_REALTIME)*1e-6 # return ms

################################################################################
g = genomix.connect()

g.rpath(os.environ['HOME'] + '/openrobots/lib/genom/pocolibs/plugins')

nhfc = g.load('nhfc')

state_port = setup()

input("start simulation?")

# ############################
#  INITIALIZE SIMULATION HERE
# ############################

x0 = np.zeros(13)
x0[3] = 1.0  # qw = 1 (identity quaternion)
u0 = np.zeros(N_ROTORS)

t0 = 0
tf = 17
dt = 1e-3
SUBSTEPS = 2
sub_dt = dt / SUBSTEPS

# preallocate arrays to store all simulation data
# more efficient than dynamic allocation
N = math.ceil((tf-t0)/dt)
tt = np.linspace(t0, tf, N)
x_log = np.zeros((N, x0.shape[0]))
u_log = np.zeros((N, u0.shape[0]))
t_log = np.zeros(N) # (N,)
tc_log = np.zeros(N) # (N,)

start()

# send initial state to nhfc before set_current_position
x = x0.copy()
xdot_init = dynamics(x, u0, G_ALLOC, MASS, J, J_INV)
state = np.hstack((x, xdot_init[-6:])).reshape(-1)
state_to_nhfc(state_port, state)
time.sleep(0.5)

nhfc.set_current_position()
time.sleep(0.1)

wp_flags = [True, True, True, True]
for i, ts in enumerate(tt):
    # wp0: t=0s -> (0,0,0, yaw=0)
    if wp_flags[0] and ts >= 0:
        nhfc.set_position(0, 0, 0, 0)
        wp_flags[0] = False
    # wp1: t=2s -> (1,1,1, yaw=0)
    if wp_flags[1] and ts >= 2:
        nhfc.set_position(1, 1, 1, 0)
        wp_flags[1] = False
    # wp2: t=7s -> (1,-1,1, yaw=pi/2)
    if wp_flags[2] and ts >= 7:
        nhfc.set_position(1, -1, 1, math.pi/2)
        wp_flags[2] = False
    # wp3: t=12s -> (0,0,0, yaw=0)
    if wp_flags[3] and ts >= 12:
        nhfc.set_position(0, 0, 0, 0)
        wp_flags[3] = False

    t1 = get_time_now_ms()

    c_f = CF

    speeds = rotor_speeds_from_nhfc(c_f)
    w_squared = speeds * np.abs(speeds)

    # ################################
    #  UPDATE SIMULATION: make 1 step
    #  (multiple substeps for accuracy, single nhfc comm for speed)
    # ################################
    for _ in range(SUBSTEPS):
        x, xdot = rk4_step(x, w_squared, sub_dt, G_ALLOC, MASS, J, J_INV)
        x = apply_ground_reaction(x)
    u = w_squared

    # update nhfc state
    state = np.hstack((x, xdot[-6:])).reshape(-1)
    state_to_nhfc(state_port, state)

    # save data
    t_log[i] = ts
    x_log[i, :] = x.reshape(-1)
    u_log[i, :] = u.reshape(-1)

    # print simulation time but every 1k iterations
    if (int(ts/dt) % 1000) == 0:
        print(f"t: {ts}")

    # if necessary, wait to match dt
    t2 = get_time_now_ms()
    elapsed_ms = t2 - t1
    tc_log[i] = elapsed_ms
    dt_ms = dt * 1000.0
    remaining_ms = dt_ms - elapsed_ms
    if remaining_ms > 0:
        time.sleep(remaining_ms * 1e-3)
    elif remaining_ms < 0:
        print(f"delay of: {-remaining_ms:.2f}ms")

stop()

# ############################################
#  SAVE DATA TO DISK:
#  generate log files as GenoM3 components do
# ############################################

os.makedirs(LOG_DIR, exist_ok=True)
np.savez(
    LOG_DIR + '/simulation_data.npz',
    t=t_log, x=x_log, u=u_log, tc=tc_log
)
print(f"Data saved to {LOG_DIR}/simulation_data.npz")
