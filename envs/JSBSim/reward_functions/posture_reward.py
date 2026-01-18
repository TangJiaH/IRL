import numpy as np
from wandb import agent
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class PostureReward(BaseRewardFunction):
    """
    PostureReward = Orientation * Range
    - Orientation: Encourage pointing at enemy fighter, punish when is pointed at.
    - Range: Encourage getting closer to enemy fighter, punish if too far away.

    NOTE:
    - Only support one-to-one environments.
    """
    def __init__(self, config):
        super().__init__(config)
        self.orientation_version = getattr(self.config, f'{self.__class__.__name__}_orientation_version', 'v2')
        self.range_version = getattr(self.config, f'{self.__class__.__name__}_range_version', 'v3')
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 3.0)

        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]

    # def get_reward(self, task, env, agent_id):
    #     """
    #     Reward is a complex function of AO, TA and R in the last timestep.
    #
    #     Args:
    #         task: task instance
    #         env: environment instance
    #
    #     Returns:
    #         (float): reward
    #     """
    #     new_reward = 0
    #     # feature: (north, east, down, vn, ve, vd)
    #     ego_feature = np.hstack([env.agents[agent_id].get_position(),
    #                              env.agents[agent_id].get_velocity()])
    #
    #     for enm in env.agents[agent_id].enemies:
    #         if enm.is_alive:
    #             enm_feature = np.hstack([enm.get_position(),
    #                                     enm.get_velocity()])
    #             AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
    #             orientation_reward = self.orientation_fn(AO, TA)
    #             range_reward = self.range_fn(R / 1000)
    #             new_reward += orientation_reward * range_reward
    #         else:
    #             new_reward += 0
    #     return self._process(new_reward, agent_id)

    def get_reward(self, task, env, agent_id):
        max_reward = -np.inf  # 初始化为负无穷
        reward_details = [0, 0, 0]  # 存储最优奖励的细节

        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        has_alive_enemy = False
        for enm in env.agents[agent_id].enemies:
            if enm.is_alive:
                has_alive_enemy = True
                enm_feature = np.hstack([enm.get_position(),
                                         enm.get_velocity()])
                AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
                orientation_reward = self.orientation_fn(AO, TA)
                range_reward = self.range_fn(R / 1000)
                current_reward = orientation_reward * range_reward

                if current_reward > max_reward:
                    max_reward = current_reward
                    reward_details = [current_reward, orientation_reward, range_reward]

        # 如果没有存活的敌人，奖励为0；否则返回最大奖励
        final_reward = max_reward if has_alive_enemy else 0
        # 记录分项奖励时，需要确保 self._process 能处理列表或修改其接口
        # 这里为了简化，我们只返回总奖励，但实际应用中可能需要调整 _process
        # self.reward_items[agent_id] = reward_details
        return self._process(final_reward, agent_id)

    def get_orientation_function(self, version):
        if version == 'v0':
            return lambda AO, TA: (1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3. \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        elif version == 'v1':
            return lambda AO, TA: (1. - np.tanh(2 * (AO - np.pi / 2))) / 2. \
                * (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5
        elif version == 'v2':
            return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        else:
            raise NotImplementedError(f"Unknown orientation function version: {version}")

    def get_range_funtion(self, version):
        if version == 'v0':
            return lambda R: np.exp(-(R - self.target_dist) ** 2 * 0.004) / (1. + np.exp(-(R - self.target_dist + 2) * 2))
        elif version == 'v1':
            return lambda R: np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                     (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1)
        elif version == 'v2':
            return lambda R: max(np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                         (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1), np.sign(7 - R))
        elif version == 'v3':
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)
        elif version == 'v4':
            target_dist = self.target_dist
            optimal_sigma = 1.5  # 高斯函数的标准差，控制峰值的宽度
            max_dist_penalty_start = 18.0  # 开始施加远距离惩罚的距离 (km)
            penalty_factor = 0.1  # 远距离惩罚的强度

            def range_v4(R):
                # 在目标距离附近给予高斯奖励，峰值为 1
                gaussian_reward = np.exp(-((R - target_dist) ** 2) / (2 * optimal_sigma ** 2))
                # 对超过最大距离的情况施加线性惩罚
                far_penalty = penalty_factor * max(0, R - max_dist_penalty_start)
                # 最终奖励 = 高斯奖励 - 远距离惩罚，并限制在 [-0.5, 1] 范围内
                return np.clip(gaussian_reward - far_penalty, -0.5, 1.0)

            return range_v4
        else:
            raise NotImplementedError(f"Unknown range function version: {version}")
