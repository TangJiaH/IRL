import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import cup_reward


class AltitudeReward(BaseRewardFunction):
    """
    AltitudeReward
    Punish if current fighter doesn't satisfy some constraints. Typically negative.
    - Punishment of velocity when lower than safe altitude   (range: [-1, 0])
    - Punishment of altitude when lower than danger altitude (range: [-1, 0])
    """
    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)         # km
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)     # km
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)     # mh

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
        # 高度惩罚
        min_h = 4000
        max_h = 8000
        k_h = 0.03
        h = env.agents[agent_id].get_position()[-1]
        r_h = cup_reward(h, min_h, max_h, k_h)
        return self._process(r_h, agent_id)
