import numpy as np
from .reward_function_base import BaseRewardFunction


class AltitudeReward(BaseRewardFunction):
    """
    AltitudeReward
    Provides rewards/penalties based on altitude constraints, similar to BoundaryReward.
    - Safe Zone: 3500m to 14000m (Reward = 0)
    - Warning Zones: 2500m-3500m and 14000m-15000m (Quadratic Penalty)
    - Danger Zones: < 2500m or > 15000m (Crash Penalty = -200)
    """

    def __init__(self, config):
        super().__init__(config)
        # 定义高度阈值 (单位: 米)
        self.danger_altitude_low = getattr(self.config, f'{self.__class__.__name__}_danger_altitude_low', 2500.0)
        self.safe_altitude_low = getattr(self.config, f'{self.__class__.__name__}_safe_altitude_low', 3500.0)
        self.safe_altitude_high = getattr(self.config, f'{self.__class__.__name__}_safe_altitude_high', 14000.0)
        self.danger_altitude_high = getattr(self.config, f'{self.__class__.__name__}_danger_altitude_high', 15000.0)

        # 惩罚系数
        self.penalty_scale = getattr(self.config, f'{self.__class__.__name__}_penalty_scale', 8.0)
        # 坠毁惩罚值 (必须与边界惩罚保持一致的重要性)
        self.crash_penalty = getattr(self.config, f'{self.__class__.__name__}_crash_penalty', 0)

        # 速度惩罚系数
        self.velocity_penalty_scale = getattr(self.config, f'{self.__class__.__name__}_velocity_penalty_scale',
                                              0.05)  # 需要调整

        # 确保高度设置合理
        assert self.danger_altitude_low < self.safe_altitude_low < self.safe_altitude_high < self.danger_altitude_high

        self.reward_item_names = [self.__class__.__name__]

    def get_reward(self, task, env, agent_id):
        """
        根据飞机的高度计算奖励。

        Args:
            task: task 实例
            env: environment 实例
            agent_id: agent ID

        Returns:
            (float): 奖励值
        """
        agent = env.agents[agent_id]
        h = agent.get_position()[-1]  # 获取当前高度 (单位: 米)
        vz = agent.get_velocity()[-1]  # 获取垂直速度 (单位: 米/秒)

        reward = 0.0

        if h < self.danger_altitude_low or h > self.danger_altitude_high:
            # 危险区/坠毁区
            reward = self.crash_penalty
        elif h < self.safe_altitude_low:
            # 低空警告区
            normalized_penetration = (self.safe_altitude_low - h) / (self.safe_altitude_low - self.danger_altitude_low)
            reward = -self.penalty_scale * (normalized_penetration ** 2)
            # 低空速度惩罚: 如果在低空警告区且仍在下降 (vz < 0)
            if vz < 0:
                reward += vz * self.velocity_penalty_scale  # vz是负数，所以是减法 (惩罚)
        elif h > self.safe_altitude_high:
            # 高空警告区
            normalized_penetration = (h - self.safe_altitude_high) / (
                        self.danger_altitude_high - self.safe_altitude_high)
            reward = -self.penalty_scale * (normalized_penetration ** 2)
            # 高空速度惩罚: 如果在高空警告区且仍在上升 (vz > 0)
            if vz > 0:
                reward -= vz * self.velocity_penalty_scale  # vz是正数，所以是减法 (惩罚)
        else:
            # 安全区
            reward = 0.0

        return self._process(reward, agent_id)
