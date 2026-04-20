"""Vertical takeoff/hover: drones hold their initial X-Y and climb to a target height.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid-vertical.py

Notes
-----
Each drone starts on the ground (small Z) at a spaced X offset and tracks the same
X-Y while ascending to ``target_height``.

Kinematic time series (same layout as ``MultiModelLearn.py`` rollouts) are saved under
``output_folder`` as ``kinematic_rollout_<drone_model>.npz`` for a single drone, or
``kinematic_rollout_<drone_model>_drone<j>.npz`` when ``num_drones`` > 1. Each archive
contains ``timestamps``, ``states``, ``controls``, ``t``, and ``logging_freq_hz``.
When plotting is enabled, the figure is also written to
``kinematic_rollout_<drone_model>.png`` in the same folder.

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 30
DEFAULT_DURATION_SEC = 10
DEFAULT_TARGET_HEIGHT_M = 1
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        target_height_m=DEFAULT_TARGET_HEIGHT_M,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    
    filename = f'save-pid-{str(drone)}-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        
    #### Initialize the simulation #############################
    # Spaced along X so multiple drones do not start collocated.

    np_random = np.random.default_rng()
    
    # INIT_XYZS = np.array([[0.25 * (i - (num_drones - 1) / 2), 0, 0.02] for i in range(num_drones)])
    INIT_XYZS = np.array([[0, 0, 0.02] for i in range(num_drones)]);
    INIT_RPYS = np_random.uniform(-np.pi/30, np.pi/30, size=(num_drones, 3))
    # INIT_RPYS = np.array([[0, 0, 0] for i in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=os.path.join(output_folder, filename),
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int((duration_sec+2)*env.CTRL_FREQ)):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(num_drones):
            target_pos = np.hstack([INIT_XYZS[j, 0:2], target_height_m])
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=target_pos,
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([INIT_XYZS[j, 0:2], target_height_m, INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    drone_label = drone.value
    for j in range(num_drones):
        n = int(logger.counters[j])
        freq = int(logger.LOGGING_FREQ_HZ)
        t_axis = np.arange(0, n / freq, 1.0 / freq)
        base = f'kinematic_rollout_{drone_label}'
        if num_drones > 1:
            base = f'{base}_drone{j}'
        rollout_npz_path = os.path.join(logger.OUTPUT_FOLDER, f'{base}.npz')
        np.savez_compressed(
            rollout_npz_path,
            timestamps=logger.timestamps[j, :n],
            states=logger.states[j, :, :n],
            controls=logger.controls[j, :, :n],
            t=t_axis,
            logging_freq_hz=freq,
        )
        print('[INFO] Saved rollout kinematic data:', rollout_npz_path)

    #### Plot the simulation results ###########################
    if plot:
        base = os.path.join(logger.OUTPUT_FOLDER, f'kinematic_rollout_{drone_label}')
        kinematic_plot_paths = [f'{base}.png', f'{base}.svg']
        logger.plot(save_path=kinematic_plot_paths)
        print('[INFO] Saved rollout kinematic plots:', ', '.join(kinematic_plot_paths))

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Vertical ascent to target height using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--target_height_m',    default=DEFAULT_TARGET_HEIGHT_M,      type=float,         help='Target altitude (m) above ground (default: 1.0)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
