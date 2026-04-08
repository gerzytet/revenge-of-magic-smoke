#!/bin/bash


/opt/miniconda3/bin/python rollout_rpy_error.py ./results/save-pid-DroneModel.CF2X-04.07.2026_21.53.39-useme/kinematic_rollout_cf2x.npz
/opt/miniconda3/bin/python rollout_rpy_error.py ./results/save-pid-DroneModel.CF2P-04.07.2026_21.53.12-useme/kinematic_rollout_cf2p.npz
/opt/miniconda3/bin/python rollout_rpy_error.py ./results/save-multimodel-04.07.2026_20.36.11-useme/rollouts/cf2p/kinematic_rollout_cf2p.npz 
/opt/miniconda3/bin/python rollout_rpy_error.py ./results/save-multimodel-04.07.2026_20.36.11-useme/rollouts/cf2x/kinematic_rollout_cf2x.npz 
