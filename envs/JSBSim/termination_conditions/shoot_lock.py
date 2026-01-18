import numpy as np

from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c


class ShootLock(BaseTerminationCondition):
    """
    LowAltitude
    End up the simulation if altitude are too low.
    """

    def __init__(self, config):
        super().__init__(config)
        self.radius_limit = getattr(config, 'radius_limit', 11.5)  # unit: m

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if altitude are too low.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
        ego_x = env.agents[agent_id].get_position()[0] / 1000  # unit: km
        ego_y = env.agents[agent_id].get_position()[1] / 1000  # unit: km
        distance = np.sqrt(ego_x ** 2 + ego_y ** 2)
        done = distance >= self.radius_limit
        if done:
            env.agents[agent_id].crash()
            self.log(f'{agent_id} exceeds the boundary. Total Steps={env.current_step}')
            print(f'{agent_id} exceeds the boundary. Total Steps={env.current_step}')
        success = False
        done = False
        return done, success, info
