import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R, cup_reward
from ..core.catalog import Catalog as c


class BoundaryReward(BaseRewardFunction):
    """
    BoundaryReward
    速度边界、高度边界、半径边界（惩戒性奖励）

    """

    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)  # km
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)  # km
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)  # mh

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
        reward = 0
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
        ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])

        # 高度惩罚
        min_h = 3500
        max_h = 8000
        k_h = 0.03
        h = env.agents[agent_id].get_position()[-1]
        r_h = cup_reward(h, min_h, max_h, k_h)

        # 速度惩罚
        min_v = 100
        max_v = 500
        k_v = 0.2
        v = ego_v
        r_v = cup_reward(v, min_v, max_v, k_v)

        # 距离惩罚
        min_R = 0
        max_R = 20000
        k_R = 0.01
        r_R = 0
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            _, _, R = get_AO_TA_R(ego_feature, enm_feature)
            r_R += cup_reward(R, min_R, max_R, k_R)

        # 加速度惩罚
        min_a = -9
        max_a = 9
        k_a = 10
        accelerations = [
            c.accelerations_n_pilot_x_norm,  # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,  # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,  # 15. a_down    (unit: G)
        ]
        a_x, a_y, a_z = env.agents[agent_id].get_property_values(accelerations)
        a = np.linalg.norm([a_x, a_y, a_z])
        r_a = cup_reward(a, min_a, max_a, k_a)

        # # 坐标惩罚
        # min_xy = -12000
        # max_xy = 12000
        # k_xy = 0.01
        # r_xy = cup_reward(ego_x, min_xy, max_xy, k_xy) + cup_reward(ego_y, min_xy, max_xy, k_xy)

        # 坐标惩罚
        min_xy = -12000
        max_xy = 12000
        k_xy = 0.005
        xy = np.linalg.norm([ego_x, ego_y])
        r_xy = cup_reward(xy, min_xy, max_xy, k_xy)

        reward = 0.4 * r_h + 0.4 * r_v + 0.2 * r_R + 0.2 * r_a + 0.4 * r_xy
        return self._process(reward, agent_id, (r_h, r_v, r_R, r_a, r_xy))
