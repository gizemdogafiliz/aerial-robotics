import numpy as np
import matplotlib.pyplot as plt
import math
import os

# --- Quadrotor physical parameters (from mrsim-quadrotor SDF) ---
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
# Wrench w(u) = G @ u_lambda, where u_lambda_i = |w_i| * w_i
# For fixed-axis collinear quadrotor: v_i = [0, 0, 1] for all rotors
# f_i = cf * v_i * u_lambda_i
# m_i = k_i * ct * v_i * u_lambda_i + p_i x f_i
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


G_ALLOC = compute_G(N_ROTORS, ARMLEN, CF, CT, RZ)


# --- Quaternion utilities ---
def quat_multiply(q, p):
    """Hamilton product (scalar first: w, x, y, z)."""
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


def quat_to_euler(q):
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])


# --- Dynamics: xdot = f(x, u) ---
# State:  x = [p(3), q(4), v(3), omega(3)]  (13 elements)
#   p:     position in world frame W
#   q:     quaternion (scalar first), orientation of B w.r.t. W
#   v:     linear velocity of B origin, expressed in W
#   omega: angular velocity of B w.r.t. W, expressed in B
#
# Input:  u_lambda = [u_lambda_1 ... u_lambda_N], u_lambda_i = |w_i|*w_i
#
# Equations of motion (Newton-Euler):
#   m * v_dot     = -m*g*z_W + R * f_B
#   J * omega_dot = -omega x (J * omega) + tau_B
#   p_dot         = v
#   q_dot         = 1/2 * q (x) (0, omega)
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


# --- RK4 integrator ---
def rk4_step(x, u_lambda, dt, G, mass, J, J_inv):
    k1 = dynamics(x, u_lambda, G, mass, J, J_inv)
    k2 = dynamics(x + dt/2 * k1, u_lambda, G, mass, J, J_inv)
    k3 = dynamics(x + dt/2 * k2, u_lambda, G, mass, J, J_inv)
    k4 = dynamics(x + dt * k3, u_lambda, G, mass, J, J_inv)
    x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    x_new[3:7] = quat_normalize(x_new[3:7])
    return x_new, k1


# --- Ground reaction ---
# If robot height is below the ground:
#   keep constant position and attitude (z = ground level)
#   reset all velocities and accelerations to 0
def apply_ground_reaction(x):
    if x[2] <= GROUND_Z:
        x[2] = GROUND_Z
        x[7:10] = 0.0
        x[10:13] = 0.0
    return x


# --- Simulation ---
def simulate(total_thrust, t_final=5.0, dt=1e-3, label=""):
    # Equal rotor inputs for desired total vertical thrust
    # total_thrust = N * cf * u_lambda  =>  u_lambda = total_thrust / (N * cf)
    u_lambda_val = total_thrust / (N_ROTORS * CF)
    u_lambda = np.full(N_ROTORS, u_lambda_val)

    x0 = np.zeros(13)
    x0[3] = 1.0  # identity quaternion

    N = math.ceil(t_final / dt)
    t_log = np.zeros(N)
    x_log = np.zeros((N, 13))
    acc_log = np.zeros((N, 3))

    x = x0.copy()
    for k in range(N):
        t_log[k] = k * dt
        x_log[k] = x

        x_new, xdot = rk4_step(x, u_lambda, dt, G_ALLOC, MASS, J, J_INV)
        acc_log[k] = xdot[7:10]

        x_new = apply_ground_reaction(x_new)
        x = x_new

    print(f"[{label}] Done: {t_final}s, {N} steps")
    return {
        't': t_log, 'x': x_log, 'acc': acc_log,
        'label': label, 'u_lambda': u_lambda,
    }


# --- Plotting ---
# Layout from lab slides (slide 5): measured state only, no controller setpoints.
# RGB for XYZ axes.
def plot_results(results, save_dir=None):
    RAD2DEG = 180.0 / np.pi
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 120})

    for res in results:
        t = res['t']
        x = res['x']
        acc = res['acc']
        label = res['label']
        u = res['u_lambda']

        euler = np.array([quat_to_euler(x[i, 3:7]) for i in range(len(t))])

        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Standalone Simulator — {label}', fontsize=14)

        # Row 1, Col 1: Position [m]
        ax = axes[0, 0]
        ax.set_title('Position [m]')
        ax.plot(t, x[:, 0], 'r', lw=0.8, label='x')
        ax.plot(t, x[:, 1], 'g', lw=0.8, label='y')
        ax.plot(t, x[:, 2], 'b', lw=0.8, label='z')
        ax.set_xlabel('t [s]'); ax.set_ylabel('[m]')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Row 1, Col 2: Attitude [deg]
        ax = axes[0, 1]
        ax.set_title('Attitude [deg]')
        ax.plot(t, euler[:, 0] * RAD2DEG, 'r', lw=0.8, label='roll')
        ax.plot(t, euler[:, 1] * RAD2DEG, 'g', lw=0.8, label='pitch')
        ax.plot(t, euler[:, 2] * RAD2DEG, 'b', lw=0.8, label='yaw')
        ax.set_xlabel('t [s]'); ax.set_ylabel('[deg]')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Row 2, Col 1: Linear Velocity [m/s]
        ax = axes[1, 0]
        ax.set_title('Linear Velocity [m/s]')
        ax.plot(t, x[:, 7], 'r', lw=0.8, label='vx')
        ax.plot(t, x[:, 8], 'g', lw=0.8, label='vy')
        ax.plot(t, x[:, 9], 'b', lw=0.8, label='vz')
        ax.set_xlabel('t [s]'); ax.set_ylabel('[m/s]')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Row 2, Col 2: Angular Velocity [deg/s]
        ax = axes[1, 1]
        ax.set_title('Angular Velocity [deg/s]')
        ax.plot(t, x[:, 10] * RAD2DEG, 'r', lw=0.8, label='wx')
        ax.plot(t, x[:, 11] * RAD2DEG, 'g', lw=0.8, label='wy')
        ax.plot(t, x[:, 12] * RAD2DEG, 'b', lw=0.8, label='wz')
        ax.set_xlabel('t [s]'); ax.set_ylabel('[deg/s]')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Row 3, Col 1: Linear Acceleration [m/s²]
        ax = axes[2, 0]
        ax.set_title('Linear Acceleration [m/s²]')
        ax.plot(t, acc[:, 0], 'r', lw=0.8, label='ax')
        ax.plot(t, acc[:, 1], 'g', lw=0.8, label='ay')
        ax.plot(t, acc[:, 2], 'b', lw=0.8, label='az')
        ax.set_xlabel('t [s]'); ax.set_ylabel('[m/s²]')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Row 3, Col 2: Propeller forces [N]
        ax = axes[2, 1]
        ax.set_title('Propeller Forces [N]')
        for i in range(len(u)):
            f_prop = CF * u[i]
            ax.plot(t, np.full_like(t, f_prop), lw=0.8,
                    label=f'prop {i+1}: {f_prop:.3f} N')
        ax.set_xlabel('t [s]'); ax.set_ylabel('[N]')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = label.replace(' ', '_').replace('=', '').replace('+', 'plus')
            fig.savefig(os.path.join(save_dir, f'{fname}.png'),
                        dpi=150, bbox_inches='tight')
            print(f"Saved: {save_dir}/{fname}.png")

    plt.show()


# ============================================================
#  MAIN
# ============================================================
if __name__ == '__main__':
    weight = MASS * G_ACC

    print("Test 1: thrust = weight (hover)...")
    r1 = simulate(weight, t_final=5.0, dt=1e-3, label="thrust = mg")

    print("Test 2: thrust = weight + 1N (ascent)...")
    r2 = simulate(weight + 1.0, t_final=5.0, dt=1e-3, label="thrust = mg + 1N")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_DIR = os.path.join(SCRIPT_DIR, 'plots_part2b')
    plot_results([r1, r2], save_dir=PLOT_DIR)
