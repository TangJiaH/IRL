import numpy as np

from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c


class OutBoundary(BaseTerminationCondition):
    """
    LowAltitude
    End up the simulation if altitude are too low.
    """

    def __init__(self, config):
        super().__init__(config)
        self.danger_radius = getattr(self.config, f'{self.__class__.__name__}_danger_radius', 16)  # km

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
        ego_x = env.agents[agent_id].get_position()[0] / 1000  # unit: km
        ego_y = env.agents[agent_id].get_position()[1] / 1000  # unit: km
        distance = np.sqrt(ego_x ** 2 + ego_y ** 2)
        done = distance >= self.danger_radius
        if done:
            env.agents[agent_id].crash()
            self.log(f'{agent_id} exceeds the boundary {self.danger_radius}km. Total Steps={env.current_step}')
            # print(f'{agent_id} exceeds the boundary. Total Steps={env.current_step}')
        success = False
        return done, success, info
