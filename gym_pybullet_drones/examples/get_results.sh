#!/bin/bash
/opt/miniconda3/bin/python rollout_rpy_error.py ./results/hover_v3_new/rollouts/cf2p/kinematic_rollout_cf2p.npz
/opt/miniconda3/bin/python rollout_rpy_error.py ./results/hover_v3_new/rollouts/cf2x/kinematic_rollout_cf2x.npz
/opt/miniconda3/bin/python rollout_rpy_error.py ./results/pid_CF2P/kinematic_rollout_cf2p.npz
/opt/miniconda3/bin/python rollout_rpy_error.py ./results/pid_CF2X/kinematic_rollout_cf2x.npz

/opt/miniconda3/bin/python compare_error.py ./results/hover_v3_new/rollouts/cf2p/kinematic_rollout_cf2p.npz ./results/pid_CF2P/kinematic_rollout_cf2p.npz
/opt/miniconda3/bin/python compare_error.py ./results/hover_v3_new/rollouts/cf2x/kinematic_rollout_cf2x.npz ./results/pid_CF2X/kinematic_rollout_cf2x.npz
