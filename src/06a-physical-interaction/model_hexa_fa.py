"""
Assignment 06a: Physical Interaction — Gazebo Simulation
Fully-actuated hexarotor (TiltHex) with end-effector contacting a thin wall.

Architecture:
  Gazebo (mrsim plugin) → dynamics + contact via SDF wall
  optitrack → motion capture state
  rotorcraft → motor interface (/tmp/pty-hr6)
  pom → state estimation (IMU + mocap fusion)
  uavpos → fully-actuated position controller
  uavatt → fully-actuated attitude controller
  maneuver → trajectory generator
  phynt → admittance filter + wrench observer

Data flow:
  maneuver/desired → phynt/reference (nominal)
  phynt/desired → uavpos/reference (AF filtered)
  uavpos/uav_input → uavatt/uav_input
  uavatt/rotor_input → rotorcraft/rotor_input → Gazebo motors
  pom/frame/robot → uavpos/state, uavatt/state, phynt/state
  rotorcraft/wrench → phynt/wrench_measure (for WO)

Usage:
  Terminal 1: ./simulation.sh   (starts Gazebo + components)
  Terminal 2: python3 -i model_hexa_fa.py
              >>> simulation()
"""

import genomix
import math
import numpy as np
import os
import time
import shutil

# ############################################
#  PARAMETERS
# ############################################

MASS = 2.72  # base 2.3 + 6 rotors × 0.07 (from mrsim-rotor SDF)
J = [0.0115, 0, 0, 0, 0.0114, 0, 0, 0, 0.0194]
ARMLEN = 0.38998
CF = 9.9016e-4
CT = 1.9e-5

# Wall (set in SDF, must match)
X_WALL = 2.0
L_BAR = 0.6

# Body x for contact: EE = body + L_BAR. Want EE 0.15m past wall (theoretical 1.5N).
# Lighter penetration to avoid drone slamming into the wall under aggressive gains.
BODY_X_CONTACT = X_WALL - L_BAR + 0.15

LOG_DIR = '/shared-workspace/logs/06a-physical-interaction/gazebo'
os.makedirs(LOG_DIR, exist_ok=True)

# ############################################
#  GENOMIX SETUP
# ############################################

g = genomix.connect()
g.rpath(os.environ['HOME'] + '/openrobots/lib/genom/pocolibs/plugins')

optitrack = g.load('optitrack')
rotorcraft = g.load('rotorcraft')
pom = g.load('pom')
uavpos = g.load('uavpos')
uavatt = g.load('uavatt')
maneuver = g.load('maneuver')
phynt = g.load('phynt')


def setup():
    print("Waiting 5s for Gazebo plugins to load...")
    time.sleep(5)
    print("Connecting components...")

    # optitrack — motion capture from Gazebo
    optitrack.connect({
        'host': 'localhost', 'host_port': '1509', 'mcast': '', 'mcast_port': '0'
    })

    # rotorcraft — motor interface (Gazebo simulates serial port)
    rotorcraft.connect({'serial': '/tmp/pty-hr6', 'baud': 0})
    rotorcraft.set_sensor_rate({'rate': {
        'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1
    }})
    rotorcraft.set_imu_filter({
        'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]
    })

    # pom — state estimation
    pom.set_prediction_model('::pom::constant_acceleration')
    pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})
    pom.set_history_length({'history_length': 0.25})
    pom.set_mag_field({'magdir': {
        'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06
    }})
    pom.connect_port({'local': 'measure/imu', 'remote': 'rotorcraft/imu'})
    pom.add_measurement('imu')
    pom.connect_port({'local': 'measure/mag', 'remote': 'rotorcraft/mag'})
    pom.add_measurement('mag')
    pom.connect_port({
        'local': 'measure/mocap', 'remote': 'optitrack/bodies/HR_6'
    })
    pom.add_measurement('mocap')

    # uavpos — fully-actuated position controller
    uavpos.set_mass({'mass': MASS})
    uavpos.set_xyradius({'rxy': 5.0})
    uavpos.set_saturation({'sat': {'x': 3.0, 'v': 3.0, 'ix': 0.5}})
    uavpos.set_servo_gain({'gain': {
        'Kpxy': 5.0,  'Kpz': 25.0,
        'Kvxy': 7.0,  'Kvz': 18.0,
        'Kixy': 0.0,  'Kiz': 1.0
    }})
    uavpos.set_emerg({'emerg': {
        'descent': 0.1, 'dx': 50.0, 'dv': 50.0
    }})
    uavpos.connect_port({'local': 'state', 'remote': 'pom/frame/robot'})
    uavpos.connect_port({'local': 'reference', 'remote': 'phynt/desired'})

    # uavatt — fully-actuated attitude controller
    uavatt.set_gtmrp_geom({
        'rotors': 6, 'cx': 0, 'cy': 0, 'cz': 0,
        'armlen': ARMLEN, 'mass': MASS,
        'rx': -21.2, 'ry': -18.7, 'rz': -1, 'cf': CF, 'ct': CT
    })
    uavatt.set_wlimit({'wmin': 16.0, 'wmax': 100.0})
    uavatt.set_servo_gain({'gain': {
        'Kqxy': 4.0, 'Kqz': 3.0,
        'Kwxy': 0.8, 'Kwz': 0.8
    }})
    uavatt.connect_port({'local': 'state', 'remote': 'pom/frame/robot'})
    uavatt.connect_port({'local': 'uav_input', 'remote': 'uavpos/uav_input'})

    # rotorcraft reads from uavatt
    rotorcraft.connect_port({
        'local': 'rotor_input', 'remote': 'uavatt/rotor_input'
    })

    # maneuver — trajectory generator
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
    maneuver.connect_port({'local': 'state', 'remote': 'pom/frame/robot'})

    # phynt — AF + WO
    phynt.set_mass(mass=MASS)
    phynt.set_geom(J=J)

    af_mass = 5.0
    af_K = [10.0, 100.0, 100.0, 50.0, 50.0, 50.0]
    af_B = [2*math.sqrt(af_mass*af_K[i]) for i in range(6)]
    af_J = [0.05, 0, 0, 0, 0.05, 0, 0, 0, 0.05]
    phynt.set_af_parameters(mass=af_mass, B=af_B, K=af_K, J=af_J)

    # WO: torque observer disabled to avoid drift in simulation
    phynt.set_wo_gains(K=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    phynt.set_wo_fc(fc=[20.0, 20.0, 20.0, 1.0, 1.0, 1.0])

    phynt.connect_port({'local': 'state', 'remote': 'pom/frame/robot'})
    phynt.connect_port({
        'local': 'wrench_measure', 'remote': 'uavatt/wrench_measure'
    })
    phynt.connect_port({
        'local': 'reference', 'remote': 'maneuver/desired'
    })

    print("All components connected.")


def start():
    pom.log_state('/tmp/pom.log')
    pom.log_measurements('/tmp/pom-measurements.log')
    optitrack.set_logfile('/tmp/opti.log')
    rotorcraft.log('/tmp/rotorcraft.log')
    uavpos.log('/tmp/uavpos.log')
    uavatt.log('/tmp/uavatt.log')
    maneuver.log('/tmp/maneuver.log')

    rotorcraft.start()
    rotorcraft.servo(ack=True)

    uavpos.set_current_position()
    maneuver.set_current_state()

    phynt.enable(enable={'wo': True, 'af': True})
    phynt.set_current_position()
    phynt.servo(ack=True)

    uavatt.servo(ack=True)
    uavpos.servo(ack=True)

    # phynt.log() requires a valid reference input first; servo above feeds it
    time.sleep(0.5)
    phynt.log('/tmp/phynt.log')
    print("Started: rotors armed, controllers running, AF/WO enabled.")


def stop():
    rotorcraft.stop()
    uavpos.stop()
    uavatt.stop()
    phynt.stop()

    rotorcraft.log_stop()
    uavpos.log_stop()
    uavatt.log_stop()
    maneuver.log_stop()
    phynt.log_stop()
    pom.log_stop()
    optitrack.unset_logfile()

    for f in ['rotorcraft.log', 'pom.log', 'pom-measurements.log',
              'uavpos.log', 'uavatt.log', 'maneuver.log', 'phynt.log',
              'opti.log']:
        src = '/tmp/' + f
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(LOG_DIR, f))
    print(f"Logs saved to: {LOG_DIR}")


# ############################################
#  DATA LOGGING
# ############################################

def read_state():
    """Read current state from pom."""
    try:
        s = pom.frame()['frame']
        pos = np.array([s['pos']['x'], s['pos']['y'], s['pos']['z']])
        att = np.array([s['att']['qw'], s['att']['qx'],
                        s['att']['qy'], s['att']['qz']])
        vel = np.array([s['vel']['vx'], s['vel']['vy'], s['vel']['vz']])
        avel = np.array([s['avel']['wx'], s['avel']['wy'], s['avel']['wz']])
        return pos, att, vel, avel
    except Exception as e:
        return np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(3), np.zeros(3)


def read_external_wrench():
    """Read estimated external wrench from phynt WO."""
    try:
        w = phynt.get_wo_data()
        ew = w.get('external_wrench', w)
        f = np.array([ew['force']['x'], ew['force']['y'], ew['force']['z']])
        t = np.array([ew['torque']['x'], ew['torque']['y'], ew['torque']['z']])
        return f, t
    except Exception:
        return np.zeros(3), np.zeros(3)


def read_desired_phynt():
    """Read AF-filtered reference from phynt."""
    try:
        d = phynt.desired()['desired']
        pos = np.array([d['pos']['x'], d['pos']['y'], d['pos']['z']])
        return pos
    except Exception:
        return np.zeros(3)


def read_nominal():
    """Read maneuver's original (non-filtered) reference."""
    try:
        d = maneuver.desired()['desired']
        pos = np.array([d['pos']['x'], d['pos']['y'], d['pos']['z']])
        return pos
    except Exception:
        return np.zeros(3)


# ############################################
#  WAYPOINT SEQUENCE — square pattern on wall
# ############################################

waypoints = [
    (2.0,   (0, 0, 1.0, 0, 5)),                  # hover
    (10.0,  (1.2, 0, 1.0, 0, 4)),                # approach
    (22.0,  (BODY_X_CONTACT, 0, 1.0, 0, 2)),     # contact bottom-left
    (30.0,  (BODY_X_CONTACT, 1, 1.0, 0, 5)),     # bottom-right
    (38.0,  (BODY_X_CONTACT, 1, 2.0, 0, 5)),     # top-right
    (46.0,  (BODY_X_CONTACT, 0, 2.0, 0, 5)),     # top-left
    (54.0,  (BODY_X_CONTACT, 0, 1.0, 0, 5)),     # close square
    (62.0,  (1.0, 0, 1.0, 0, 4)),                # retract — same Z as last waypoint, no needless climb
    (69.0,  (0, 0, 0, 0, 5)),                    # land
]


def simulation():
    print("=== 06a Physical Interaction — Gazebo simulation ===")
    setup()
    start()

    tf = 80.0
    dt_log = 0.02   # 50Hz logging
    N = int(tf / dt_log)

    t_log = np.zeros(N)
    pos_log = np.zeros((N, 3))
    att_log = np.zeros((N, 4))
    vel_log = np.zeros((N, 3))
    avel_log = np.zeros((N, 3))
    f_ext_log = np.zeros((N, 3))
    t_ext_log = np.zeros((N, 3))
    pos_d_log = np.zeros((N, 3))
    p_nom_log = np.zeros((N, 3))

    wo_calibrated = False
    wp_idx = 0
    t0 = time.time()

    for i in range(N):
        ts = i * dt_log

        # WO calibration during hover
        if not wo_calibrated and ts >= 20.0:
            print(f"[t={ts:.1f}s] calibrating WO...")
            phynt.set_wo_zero(duration=2.0)
            wo_calibrated = True

        # waypoint trigger
        if wp_idx < len(waypoints) and ts >= waypoints[wp_idx][0]:
            t_wp, args = waypoints[wp_idx]
            print(f"[t={ts:.1f}s] maneuver.goto{args}")
            maneuver.goto(*args)
            wp_idx += 1

        # log
        pos, att, vel, avel = read_state()
        f_ext, tau_ext = read_external_wrench()
        pos_d = read_desired_phynt()
        p_nom = read_nominal()

        t_log[i] = ts
        pos_log[i] = pos
        att_log[i] = att
        vel_log[i] = vel
        avel_log[i] = avel
        f_ext_log[i] = f_ext
        t_ext_log[i] = tau_ext
        pos_d_log[i] = pos_d
        p_nom_log[i] = p_nom

        if i % 50 == 0:
            print(f"t: {ts:5.1f}  pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]  "
                  f"f_ext: [{f_ext[0]:.2f}, {f_ext[1]:.2f}, {f_ext[2]:.2f}]")

        # real-time pacing
        elapsed = time.time() - t0
        target = (i + 1) * dt_log
        if target > elapsed:
            time.sleep(target - elapsed)

    print("Stopping...")
    stop()

    np.savez(os.path.join(LOG_DIR, 'simulation_data.npz'),
             t=t_log, pos=pos_log, att=att_log, vel=vel_log, avel=avel_log,
             f_ext=f_ext_log, tau_ext=t_ext_log,
             pos_d=pos_d_log, p_nom=p_nom_log)
    print(f"Data saved: {LOG_DIR}/simulation_data.npz")
    print("=== Simulation complete ===")


# To run:
#   python3 -i model_hexa_fa.py
#   >>> simulation()
