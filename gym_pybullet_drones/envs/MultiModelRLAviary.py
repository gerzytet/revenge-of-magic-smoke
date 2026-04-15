import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.MultiModelBaseAviary import MultiModelBaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType, ModelResamplePolicy
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class MultiModelRLAviary(MultiModelBaseAviary):
    """Base RL environment with multiple candidate drone models (one active per episode)."""

    ################################################################################

    def __init__(self,
                 drone_models: list,
                 model_resample_policy: ModelResamplePolicy = ModelResamplePolicy.EACH_RESET,
                 include_model_index_in_obs: bool = True,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a generic single-agent multi-model RL environment.

        Parameters
        ----------
        drone_models : list of DroneModel
            Candidate models; one is active per episode according to `model_resample_policy`.
        model_resample_policy : ModelResamplePolicy
            When to resample the active model.
        include_model_index_in_obs : bool
            If True (default), append a one-hot of length ``len(drone_models)`` for the active model.
        num_drones : int, optional
            Must be 1.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
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
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        self.include_model_index_in_obs = include_model_index_in_obs
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        self.ACT_TYPE = act
        #### Create integrated controllers (one per candidate model for PID-style actions) ##
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            print(f"[Error] in MultiModelRLAviary.__init__(), You cannot use an action type of {act}")
            exit()
        super().__init__(drone_models=drone_models,
                         model_resample_policy=model_resample_policy,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         user_debug_gui=False,
                         )
    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size 4 or 1, depending on the action type.

        """
        if self.ACT_TYPE == ActionType.RPM:
            size = 4
        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            size = 1
        else:
            print("[ERROR] in MultiModelRLAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs."""
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            else:
                print("[ERROR] in MultiModelRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment."""
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo,lo,lo,
                                     lo,lo,lo,
                                     lo,lo,lo,
                                     lo,lo,lo
                                     ] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[hi,hi,hi,
                                     hi,hi,hi,
                                     hi,hi,hi,
                                     hi,hi,hi
                                     ] for i in range(self.NUM_DRONES)])
        #### Add action buffer to observation space ################
        act_lo = -1
        act_hi = +1
        for i in range(self.ACTION_BUFFER_SIZE):
            if self.ACT_TYPE == ActionType.RPM:
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
        if self.include_model_index_in_obs:
            #### One column per candidate model (one-hot); size follows NUM_MODELS, not NUM_DRONES
            model_low = np.zeros((self.NUM_DRONES, self.NUM_MODELS), dtype=np.float64)
            model_high = np.ones((self.NUM_DRONES, self.NUM_MODELS), dtype=np.float64)
            obs_lower_bound = np.hstack([obs_lower_bound, model_low])
            obs_upper_bound = np.hstack([obs_upper_bound, model_high])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _active_model_one_hot(self):
        """Shape (NUM_DRONES, NUM_MODELS); each row is the same one-hot for the active model."""
        oh = np.zeros((self.NUM_DRONES, self.NUM_MODELS), dtype=np.float32)
        oh[:, self.active_model_idx] = 1.0
        return oh

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment."""
        obs_12 = np.zeros((self.NUM_DRONES,12))
        for i in range(self.NUM_DRONES):
            obs = self._getDroneStateVector(i)
            obs_12[i, :] = np.hstack([obs[0:3],
                                      obs[7:10],
                                      obs[10:13],
                                      obs[13:16]
                                      ]).reshape(12,)
        ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
        for i in range(self.ACTION_BUFFER_SIZE):
            ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
        if self.include_model_index_in_obs:
            ret = np.hstack([ret, self._active_model_one_hot()])
        return ret

    ################################################################################

    def _computeInfo(self):
        """Adds active model metadata for debugging and logging."""
        return {
            'active_model_idx': int(self.active_model_idx),
            'active_drone_model': self.DRONE_MODELS_LIST[self.active_model_idx].value,
        }
