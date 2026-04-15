import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.MultiModelRLAviary import MultiModelRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ModelResamplePolicy

class MultiModelHoverAviary(MultiModelRLAviary):
    """Single agent RL problem: hover at position, with multiple candidate drone models."""

    # Penalize angular rate and tilt from vertical (body z vs world +z); yaw-free.
    REWARD_W_OMEGA = 1.0
    REWARD_W_TILT = 12.0

    ################################################################################

    def __init__(self,
                 drone_models: list,
                 model_resample_policy: ModelResamplePolicy = ModelResamplePolicy.EACH_RESET,
                 include_model_index_in_obs: bool = True,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent multi-model RL environment.

        Parameters
        ----------
        drone_models : list of DroneModel
            Candidate drone types; one active per episode.
        model_resample_policy : ModelResamplePolicy
            When to resample the active model.
        include_model_index_in_obs : bool
            If True (default), append normalized active model index to kinematic observations.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 10
        super().__init__(drone_models=drone_models,
                         model_resample_policy=model_resample_policy,
                         include_model_index_in_obs=include_model_index_in_obs,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         act=act
                         )

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Shaped as negative penalties: small angular velocity and body z aligned with
        world +z (via rotation matrix from quaternion), without penalizing yaw.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        quat = state[3:7]
        ang_vel = state[13:16]
        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        uz_world_z = float(np.clip(rot[2, 2], -1.0, 1.0))
        tilt_gap = 1.0 - uz_world_z
        omega_sq = float(np.dot(ang_vel, ang_vel))
        dist = abs(np.linalg.norm(self.TARGET_POS-state[0:3]))
        hover_reward = max(0, 2 - dist**1.5 - dist*0.05 - state[7]*0.02 - state[8]*0.02)
        return hover_reward
        #return -self.REWARD_W_OMEGA * omega_sq - self.REWARD_W_TILT * tilt_gap

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(state[7:9]) <= .001:
            return True
        else:
            return False
        # if np.linalg.norm(self.TARGET_POS - state[0:3]) < .1:
        #     return True
        # else:
        #     return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0
             or abs(state[7]) > 0.4 or abs(state[8]) > 0.4
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s)."""
        info = super()._computeInfo()
        info['answer'] = 42
        return info
