"""
Assignment 04: State Estimation
Compare Kalman filter (pom-genom3) under different sensor covariance configs.

Usage:
  Terminal 1 (Docker):  ./simulation.sh
  Terminal 2 (Docker):  python3 state_estimation.py <config>

Configs: baseline, gps_mag_x10, gps_mag_x100, imu_only_x10, imu_only_x100

Restart simulation.sh between configs for clean Kalman filter state.
"""

import genomix
import os
import sys
import time
import shutil

HOVER_DURATION = 20
SETTLE_TIME = 5
LOG_BASE = '/shared-workspace/logs/04-state-estimation'

DEFAULT_GSTDDEV = [0.01, 0.01, 0.01]
DEFAULT_ASTDDEV = [0.05, 0.05, 0.05]

CONFIGS = {
    'baseline': {
        'optitrack_pstddev': 0,
        'optitrack_qstddev': 0,
        'imu_gstddev': DEFAULT_GSTDDEV,
        'imu_astddev': DEFAULT_ASTDDEV,
    },
    # GPS+Mag: position OK, roll/pitch unreliable, IMU noisier
    # x10: qstddev ~1.7deg, gyro x3.16, accel x3.16
    'gps_mag_x10': {
        'optitrack_pstddev': 0,
        'optitrack_qstddev': 0.03,
        'imu_gstddev': [0.032, 0.032, 0.032],
        'imu_astddev': [0.16, 0.16, 0.16],
    },
    # x100: qstddev ~5.7deg, gyro x10, accel x10
    'gps_mag_x100': {
        'optitrack_pstddev': 0,
        'optitrack_qstddev': 0.1,
        'imu_gstddev': [0.1, 0.1, 0.1],
        'imu_astddev': [0.5, 0.5, 0.5],
    },
    # IMU-only: mocap degraded for both position and attitude
    # x10: 3cm position noise, ~1.7deg attitude noise
    'imu_only_x10': {
        'optitrack_pstddev': 0.03,
        'optitrack_qstddev': 0.03,
        'imu_gstddev': DEFAULT_GSTDDEV,
        'imu_astddev': DEFAULT_ASTDDEV,
    },
    # x100: 10cm position noise, ~5.7deg attitude noise
    'imu_only_x100': {
        'optitrack_pstddev': 0.1,
        'optitrack_qstddev': 0.1,
        'imu_gstddev': DEFAULT_GSTDDEV,
        'imu_astddev': DEFAULT_ASTDDEV,
    },
}


def run_config(config_name):
    cfg = CONFIGS[config_name]
    log_dir = os.path.join(LOG_BASE, config_name)

    print(f"=== Config: {config_name} ===")
    print(f"  optitrack  pstddev={cfg['optitrack_pstddev']}, qstddev={cfg['optitrack_qstddev']}")
    print(f"  IMU        gstddev={cfg['imu_gstddev']}, astddev={cfg['imu_astddev']}")
    print(f"  Log dir:   {log_dir}")

    g = genomix.connect()
    g.rpath(os.environ['HOME'] + '/openrobots/lib/genom/pocolibs/plugins')

    optitrack = g.load('optitrack')
    rotorcraft = g.load('rotorcraft')
    pom = g.load('pom')
    nhfc = g.load('nhfc')

    # --- Optitrack ---
    optitrack.connect({
        'host': 'localhost', 'host_port': '1509', 'mcast': '', 'mcast_port': '0'
    })
    optitrack.set_noise({'noise': {
        'pstddev': cfg['optitrack_pstddev'],
        'qstddev': cfg['optitrack_qstddev'],
    }})

    # --- Rotorcraft ---
    rotorcraft.connect({'serial': '/tmp/pty-qr4', 'baud': 0})
    rotorcraft.set_sensor_rate({'rate': {
        'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1
    }})
    rotorcraft.set_imu_filter({
        'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]
    })
    rotorcraft.connect_port({
        'local': 'rotor_input', 'remote': 'nhfc/rotor_input'
    })

    result = rotorcraft.get_imu_calibration()
    result['imu_calibration']['gstddev'] = list(cfg['imu_gstddev'])
    result['imu_calibration']['astddev'] = list(cfg['imu_astddev'])
    rotorcraft.set_imu_calibration(result)

    # --- nhfc ---
    nhfc.set_gtmrp_geom({
        'rotors': 4, 'cx': 0, 'cy': 0, 'cz': 0, 'armlen': 0.23, 'mass': 1.28,
        'rx': 0, 'ry': 0, 'rz': -1, 'cf': 6.5e-4, 'ct': 1e-5
    })
    nhfc.set_emerg({'emerg': {
        'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3
    }})
    nhfc.set_saturation({'sat': {'x': 1, 'v': 1, 'ix': 0}})
    nhfc.set_servo_gain({'gain': {
        'Kpxy': 5, 'Kpz': 5, 'Kqxy': 4, 'Kqz': 0.1,
        'Kvxy': 6, 'Kvz': 6, 'Kwxy': 1, 'Kwz': 0.1,
        'Kixy': 0, 'Kiz': 0
    }})
    nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'})
    nhfc.connect_port({'local': 'rotor_measure', 'remote': 'rotorcraft/rotor_measure'})
    nhfc.connect_port({'local': 'state', 'remote': 'pom/frame/robot'})

    # --- pom ---
    pom.set_prediction_model('::pom::constant_acceleration')
    pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})
    pom.set_history_length({'history_length': 0.25})
    pom.set_mag_field({'magdir': {
        'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06
    }})

    pom.connect_port({'local': 'measure/imu', 'remote': 'rotorcraft/imu'})
    pom.add_measurement('imu')
    pom.connect_port({'local': 'measure/mocap', 'remote': 'optitrack/bodies/QR_4'})
    pom.add_measurement('mocap')

    # --- Start ---
    pom.log_state('/tmp/pom.log')
    pom.log_measurements('/tmp/pom-measurements.log')
    rotorcraft.log('/tmp/rotorcraft.log')
    nhfc.log('/tmp/nhfc.log')

    rotorcraft.start()
    rotorcraft.servo(ack=True)

    time.sleep(2)

    nhfc.set_current_position()

    print("Taking off to z=1m...")
    nhfc.set_position(-1, 0, 1, 0)

    print(f"Settling {SETTLE_TIME}s...")
    time.sleep(SETTLE_TIME)

    print(f"Recording {HOVER_DURATION}s...")
    time.sleep(HOVER_DURATION)

    # --- Descend ---
    print("Descending to ground...")
    nhfc.set_position(-1, 0, 0, 0)
    time.sleep(5)

    # --- Stop ---
    print("Stopping...")
    nhfc.stop()
    nhfc.log_stop()
    rotorcraft.stop()
    rotorcraft.log_stop()
    pom.log_stop()

    os.makedirs(log_dir, exist_ok=True)
    for f in ['pom.log', 'pom-measurements.log', 'nhfc.log', 'rotorcraft.log']:
        src = '/tmp/' + f
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(log_dir, f))

    print(f"=== Done: {config_name} ===")
    print(f"Logs: {log_dir}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CONFIGS:
        print(f"Usage: python3 {sys.argv[0]} <config>")
        print(f"Configs: {', '.join(CONFIGS.keys())}")
        sys.exit(1)

    run_config(sys.argv[1])


if __name__ == '__main__':
    main()
