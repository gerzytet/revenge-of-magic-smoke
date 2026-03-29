"""Script demonstrating PPO on `MultiModelHoverAviary` (domain randomization over drone models).

Based on `learn.py`, but uses `MultiModelHoverAviary` with multiple candidate URDFs; one model
is active per episode according to `ModelResamplePolicy`.

Example
-------
In a terminal, from the repo root (with `PYTHONPATH=.` or editable install)::

    $ python gym_pybullet_drones/examples/MultiModelLearn.py
    $ python gym_pybullet_drones/examples/MultiModelLearn.py --first_reset_only true

Notes
-----
This is a minimal working example integrating multi-model hover with stable-baselines3.

After training, the script runs a separate PyBullet rollout and saves ``Logger.plot()`` output
for each entry in ``DEFAULT_DRONE_MODELS`` as ``<run>/rollouts/<model_name>/kinematic_rollout.png``.
The training eval curve is saved as ``<run>/training_eval_reward.png`` under the same run folder
(typically under ``results/``). Use
``reset(..., options={'active_model_idx': k})`` to pin the active URDF while keeping the
same observation shape as training.

"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.MultiModelHoverAviary import MultiModelHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel, ModelResamplePolicy

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('one_d_rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

DEFAULT_DRONE_MODELS = (
    DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE,
    DroneModel.A, DroneModel.B, DroneModel.C, DroneModel.D, DroneModel.E,
)
DEFAULT_INCLUDE_MODEL_INDEX_IN_OBS = True
DEFAULT_FIRST_RESET_ONLY = False

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True,
        first_reset_only=DEFAULT_FIRST_RESET_ONLY, include_model_index_in_obs=DEFAULT_INCLUDE_MODEL_INDEX_IN_OBS):

    filename = os.path.join(output_folder, 'save-multimodel-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    policy = ModelResamplePolicy.FIRST_RESET_ONLY if first_reset_only else ModelResamplePolicy.EACH_RESET

    multimodel_kwargs = dict(
        drone_models=list(DEFAULT_DRONE_MODELS),
        model_resample_policy=policy,
        include_model_index_in_obs=include_model_index_in_obs,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
    )

    train_env = make_vec_env(MultiModelHoverAviary,
                             env_kwargs=multimodel_kwargs,
                             n_envs=1,
                             seed=0
                             )
    eval_env = MultiModelHoverAviary(**multimodel_kwargs)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    model = PPO('MlpPolicy',
                train_env,
                verbose=1)

    #### Target cumulative rewards (problem-dependent) ##########
    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 474.
    else:
        target_reward = 467.
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(1e7) if local else int(1e2),
                callback=eval_callback,
                log_interval=100)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        timesteps = data['timesteps']
        results = data['results'][:, 0]
        print("Data from evaluations.npz")
        for j in range(timesteps.shape[0]):
            print(f"{timesteps[j]},{results[j]}")
        if local:
            fig, ax = plt.subplots(figsize=(11, 6.5))
            ax.plot(timesteps, results, marker='o', linestyle='-', markersize=5)
            ax.set_xlabel('Training Steps', fontsize=12)
            ax.set_ylabel('Episode Reward', fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            ax.grid(True, alpha=0.6)
            fig.tight_layout()
            train_plot_path = os.path.join(filename, 'training_eval_reward.png')
            fig.savefig(train_plot_path, dpi=200, bbox_inches='tight')
            print('[INFO] Saved training eval plot:', train_plot_path)
            plt.show()
            plt.close(fig)

    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Aggregate evaluation (random model each episode) #######
    test_env_nogui = MultiModelHoverAviary(**multimodel_kwargs)
    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    test_env_nogui.close()

    #### One GUI rollout + kinematic plot per candidate model ####
    rollout_root = os.path.join(filename, 'rollouts')
    if not os.path.exists(rollout_root):
        os.makedirs(rollout_root)

    for model_idx, dm in enumerate(DEFAULT_DRONE_MODELS):
        print('[INFO] Rollout for drone model:', dm.value, '(index', model_idx, ')')
        test_env = MultiModelHoverAviary(gui=gui,
                                         record=record_video,
                                         **multimodel_kwargs)
        logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=1,
                    output_folder=os.path.join(rollout_root, dm.value),
                    colab=colab
                    )

        reset_opts = {'active_model_idx': model_idx}
        obs, info = test_env.reset(seed=42, options=reset_opts)
        start = time.time()
        for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
            action, _states = model.predict(obs,
                                            deterministic=True
                                            )
            obs, reward, terminated, truncated, info = test_env.step(action)
            obs2 = obs.squeeze()
            act2 = action.squeeze()
            print("Model", dm.value, "| Obs:", obs, "\tAction", action, "\tReward:", reward,
                  "\tTerminated:", terminated, "\tTruncated:", truncated)
            if DEFAULT_OBS == ObservationType.KIN:
                logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                    control=np.zeros(12)
                    )
            test_env.render()
            print(terminated)
            sync(i, start, test_env.CTRL_TIMESTEP)
            if terminated:
                obs, _ = test_env.reset(seed=42, options=reset_opts)
        test_env.close()

        if plot and DEFAULT_OBS == ObservationType.KIN:
            kinematic_plot_path = os.path.join(logger.OUTPUT_FOLDER, f'kinematic_rollout_{dm.value}.png')
            logger.plot(title='Rollout: {}'.format(dm.value), save_path=kinematic_plot_path)
            print('[INFO] Saved rollout kinematic plot:', kinematic_plot_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-model hover reinforcement learning example')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: False)', metavar='')
    parser.add_argument('--first_reset_only',   default=DEFAULT_FIRST_RESET_ONLY, type=str2bool, help='If true, ModelResamplePolicy.FIRST_RESET_ONLY; else EACH_RESET (default: False)', metavar='')
    parser.add_argument('--no_model_index_in_obs', action='store_true',        help='Set include_model_index_in_obs=False (default: include index)')
    ARGS = parser.parse_args()
    include_idx = not ARGS.no_model_index_in_obs
    run(output_folder=ARGS.output_folder, gui=ARGS.gui, plot=True, colab=ARGS.colab, record_video=ARGS.record_video, local=True,
        first_reset_only=ARGS.first_reset_only, include_model_index_in_obs=include_idx)
