#
import genomix
import os
import math
import time

# this connects to components running on the same host (localhost)
g = genomix.connect()
# to instead control components running on the remote computer "hostname" use
# g = genomix.connect('hostname')

# adapt path to your setup
g.rpath(os.environ['HOME'] + '/openrobots/lib/genom/pocolibs/plugins')

# load components clients
optitrack = g.load('optitrack')
rotorcraft = g.load('rotorcraft')
pom = g.load('pom')
nhfc = g.load('nhfc')

#In order the files not to be lost after docer is closed
# log directory: change this for different robots/simulations
LOG_DIR = '/shared-workspace/logs/01-intro/quad'
os.makedirs(LOG_DIR, exist_ok=True)

# --- setup ----------------------------------------------------------------
#
# configure components, to be called interactively
def setup():
  # optitrack
  #
  # connect to the simulated optitrack system on localhost
  optitrack.connect({
    'host': 'localhost', 'host_port': '1509', 'mcast': '', 'mcast_port': '0'
  })


  # rotorcraft
  #
  # connect to the simulated quadrotor
  rotorcraft.connect({'serial': '/tmp/pty-qr4', 'baud': 0})

  # get IMU at 1kHz and motor data at 20Hz
  rotorcraft.set_sensor_rate({'rate': {
    'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1
  }})

  # Filter IMU: 20Hz cut-off frequency for gyroscopes and 5Hz for
  # accelerometers. This is important for cancelling vibrations.
  rotorcraft.set_imu_filter({
    'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]
  })

  # read propellers velocities from nhfc controller
  rotorcraft.connect_port({
    'local': 'rotor_input', 'remote': 'nhfc/rotor_input'
  })


  # nhfc
  #
  # configure quadrotor geometry: 4 rotors, not tilted, 23cm arms
  nhfc.set_gtmrp_geom({
    'rotors': 4, 'cx': 0, 'cy': 0, 'cz': 0, 'armlen': 0.23, 'mass': 1.28,
    'rx':0, 'ry': 0, 'rz': -1, 'cf': 6.5e-4, 'ct': 1e-5
  })

  # emergency descent parameters
  nhfc.set_emerg({'emerg': {
    'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3
  }})

  # PID tuning
  nhfc.set_saturation({'sat': {'x': 1, 'v': 1, 'ix': 0}})
  nhfc.set_servo_gain({ 'gain': {
    'Kpxy': 5, 'Kpz': 5, 'Kqxy': 4, 'Kqz': 0.1,
    'Kvxy': 6, 'Kvz': 6, 'Kwxy': 1, 'Kwz': 0.1,
    'Kixy': 0, 'Kiz': 0
  }})

  # use tilt-prioritized controller
  nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'})

  # read measured propeller velocities from rotorcraft
  nhfc.connect_port({
    'local': 'rotor_measure', 'remote': 'rotorcraft/rotor_measure'
  })

  # read current state from pom
  nhfc.connect_port({
    'local': 'state', 'remote': 'pom/frame/robot'
  })


  # pom
  #
  # configure kalman filter
  pom.set_prediction_model('::pom::constant_acceleration')
  pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})

  # allow sensor data up to 250ms old
  pom.set_history_length({'history_length': 0.25})

  # configure magnetic field
  pom.set_mag_field({'magdir': {
    'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06
  }})

  # read IMU and magnetometers from rotorcraft
  pom.connect_port({'local': 'measure/imu', 'remote': 'rotorcraft/imu'})
  pom.add_measurement('imu')
  pom.connect_port({'local': 'measure/mag', 'remote': 'rotorcraft/mag'})
  pom.add_measurement('mag')

  # read position and orientation from optitrack
  pom.connect_port({
    'local': 'measure/mocap', 'remote': 'optitrack/bodies/QR_4'
  })
  pom.add_measurement('mocap')


# --- start ----------------------------------------------------------------
#
# Spin the motors and servo on current position. To be called interactively
def start():
  pom.log_state('/tmp/pom.log')
  pom.log_measurements('/tmp/pom-measurements.log')

  optitrack.set_logfile('/tmp/opti.log')

  rotorcraft.log('/tmp/rotorcraft.log')
  rotorcraft.start()
  rotorcraft.servo(ack=True) # this runs until stopped or input error

  nhfc.log('/tmp/nhfc.log')
  nhfc.set_current_position() # hover on current position


# --- stop -----------------------------------------------------------------
#
# Stop motors. To be called interactively
def stop():
  rotorcraft.stop()
  rotorcraft.log_stop()

  nhfc.stop()
  nhfc.log_stop()

  pom.log_stop()

  optitrack.unset_logfile()

  # copy logs to shared workspace
  import shutil
  os.makedirs(LOG_DIR, exist_ok=True)
  for f in ['nhfc.log', 'pom.log', 'pom-measurements.log', 'rotorcraft.log', 'opti.log']:
    src = '/tmp/' + f
    if os.path.exists(src):
      shutil.copy2(src, LOG_DIR + '/' + f)
  print(f"Logs copied to: {LOG_DIR}")


# --- simulation -----------------------------------------------------------
# Assignment function: runs the full waypoint sequence automatically
def simulation():
    print("=== Starting simulation ===")
    print(f"Log directory: {LOG_DIR}")
 
    # 1. Setup and start
    setup()
    start()
 
    # wp0: robot starts at (0, 0, 0, yaw=0) - already there
    print("[t=0s] wp0: x=0, y=0, z=0, yaw=0 (start)")
 
    # 2. Wait 2s, then go to wp1
    time.sleep(2)
    print("[t=2s] wp1: x=1, y=1, z=1, yaw=0")
    nhfc.set_position(1, 1, 1, 0)
 
    # 3. Wait 5s, then go to wp2
    time.sleep(5)
    print("[t=7s] wp2: x=1, y=-1, z=1, yaw=pi/2")
    nhfc.set_position(1, -1, 1, math.pi/2)
 
    # 4. Wait 5s, then go to wp3 (end)
    time.sleep(5)
    print("[t=12s] wp3: x=0, y=0, z=0, yaw=0 (end)")
    nhfc.set_position(0, 0, 0, 0)
 
    # 5. Wait for robot to stabilize, then stop
    time.sleep(5)
    print("[t=17s] Stopping simulation...")
    stop()
 
    print("=== Simulation complete ===")
    print(f"Log files saved in: {LOG_DIR}")



## interactively, one can start the simulation with
# setup()
# start()
## and then for instance set a desired position with
# nhfc.set_position(0, 0, 1, 0)
## to stop, use
# stop
