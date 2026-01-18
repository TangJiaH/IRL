import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import cup_reward


class BoundaryReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.safe_radius = getattr(self.config, f'{self.__class__.__name__}_safe_radius', 9.5)  # km
        self.danger_radius = getattr(self.config, f'{self.__class__.__name__}_danger_radius', 11)  # km
        self.Kr = getattr(self.config, f'{self.__class__.__name__}_Kr', 0.3)  # mh

        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        ego_x = env.agents[agent_id].get_position()[0] / 1000  # unit: km
        ego_vx = env.agents[agent_id].get_velocity()[0] / 340  # unit: mh
        ego_y = env.agents[agent_id].get_position()[1] / 1000  # unit: km
        ego_vy = env.agents[agent_id].get_velocity()[1] / 340  # unit: mh

        distance = np.sqrt(ego_x ** 2 + ego_y ** 2)

        # 坐标惩罚
        min_xy = -18000
        max_xy = 18000
        k_xy = 0.005
        xy = np.linalg.norm([ego_x, ego_y])
        r_xy = cup_reward(xy, min_xy, max_xy, k_xy)

        # print(f"agent_id:{agent_id}:  distance:{distance},  radial_speed:{radial_speed},  Pr:{Pr}")
        return self._process(r_xy, agent_id)
