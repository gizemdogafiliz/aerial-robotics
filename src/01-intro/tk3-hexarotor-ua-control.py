#
# Under-Actuated Hexarotor Control Script
# Part 1 - Assignment 1b
#
import genomix
import math
import os
import shutil
import time

# connect to genomixd server
g = genomix.connect()
g.rpath(os.environ['HOME'] + '/openrobots/lib/genom/pocolibs/plugins')

# load GenoM3 component clients
optitrack = g.load('optitrack')
rotorcraft = g.load('rotorcraft')
pom = g.load('pom')
nhfc = g.load('nhfc')

# log directory for under-actuated hexarotor
LOG_DIR = '/shared-workspace/logs/01-intro/hexa-ua'
os.makedirs(LOG_DIR, exist_ok=True)

# --- setup ----------------------------------------------------------------
def setup():
    # optitrack
    optitrack.connect({
        'host': 'localhost', 'host_port': '1509', 'mcast': '', 'mcast_port': '0'
    })

    # rotorcraft - connect to UA hexarotor serial port
    rotorcraft.connect({'serial': '/tmp/pty-hu6', 'baud': 0})
    rotorcraft.set_sensor_rate({'rate': {
        'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1
    }})
    rotorcraft.set_imu_filter({
        'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]
    })
    rotorcraft.connect_port({
        'local': 'rotor_input', 'remote': 'nhfc/rotor_input'
    })

    # nhfc - UA HEXAROTOR geometry
    #   Same as tilthex but NO TILT: rx=0, ry=0
    #   rotors: 6
    #   armlen: 0.39m (same frame as tilthex)
    #   mass: 2.3kg (same frame as tilthex)
    #   rx: 0, ry: 0 (NO TILT - this makes it under-actuated)
    #   rz: -1 (first rotor CW)
    #   cf: 9.9016e-4, ct: 1.9e-5 (same propellers as tilthex)
    nhfc.set_gtmrp_geom({
        'rotors': 6, 'cx': 0, 'cy': 0, 'cz': 0, 'armlen': 0.38998, 'mass': 2.72,
        'rx': 0, 'ry': 0, 'rz': -1, 'cf': 9.9016e-4, 'ct': 1.9e-5
    })

    nhfc.set_emerg({'emerg': {
        'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3
    }})
    nhfc.set_saturation({'sat': {'x': 1, 'v': 1, 'ix': 0}})
    nhfc.set_servo_gain({'gain': {
        'Kpxy': 10, 'Kpz': 38, 'Kqxy': 4, 'Kqz': 0.25,
        'Kvxy': 13, 'Kvz': 33, 'Kwxy': 2, 'Kwz': 0.25,
        'Kixy': 0.01, 'Kiz': 0.03
    }})
    nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'})
    nhfc.connect_port({
        'local': 'rotor_measure', 'remote': 'rotorcraft/rotor_measure'
    })
    nhfc.connect_port({
        'local': 'state', 'remote': 'pom/frame/robot'
    })

    # pom - UA hexarotor optitrack body: HU_6
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
        'local': 'measure/mocap', 'remote': 'optitrack/bodies/HU_6'
    })
    pom.add_measurement('mocap')


# --- start ----------------------------------------------------------------
def start():
    pom.log_state('/tmp/pom.log')
    pom.log_measurements('/tmp/pom-measurements.log')
    optitrack.set_logfile('/tmp/opti.log')
    rotorcraft.log('/tmp/rotorcraft.log')
    nhfc.log('/tmp/nhfc.log')

    rotorcraft.start()
    rotorcraft.servo(ack=True)
    nhfc.set_current_position()


# --- stop -----------------------------------------------------------------
def stop():
    rotorcraft.stop()
    rotorcraft.log_stop()
    nhfc.stop()
    nhfc.log_stop()
    pom.log_stop()
    optitrack.unset_logfile()

    # copy logs from /tmp/ to shared workspace
    os.makedirs(LOG_DIR, exist_ok=True)
    for f in ['nhfc.log', 'pom.log', 'pom-measurements.log', 'rotorcraft.log', 'opti.log']:
        src = '/tmp/' + f
        if os.path.exists(src):
            shutil.copy2(src, LOG_DIR + '/' + f)
    print(f"Logs copied to: {LOG_DIR}")


# --- simulation -----------------------------------------------------------
def simulation():
    print("=== Starting UA Hexarotor simulation ===")
    print(f"Log directory: {LOG_DIR}")

    setup()
    start()

    print("[t=0s] wp0: x=0, y=0, z=0, yaw=0 (start)")
    nhfc.set_position(0, 0, 0, 0)

    time.sleep(2)
    print("[t=2s] wp1: x=1, y=1, z=1, yaw=0")
    nhfc.set_position(1, 1, 1, 0)

    time.sleep(5)
    print("[t=7s] wp2: x=1, y=-1, z=1, yaw=pi/2")
    nhfc.set_position(1, -1, 1, math.pi/2)

    time.sleep(5)
    print("[t=9s] wp3: x=0, y=0, z=1, yaw=0")
    nhfc.set_position(0, 0, 1, 0)

    time.sleep(2)
    print("[t=14s] wp4: x=0, y=0, z=0, yaw=0 (end)")
    nhfc.set_position(0, 0, 0, 0)

    time.sleep(5)
    print("[t=19s] Stopping simulation...")
    stop()

    print("=== Simulation complete ===")


## To run:
# simulation()
