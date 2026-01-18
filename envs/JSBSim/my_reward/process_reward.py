import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R, get_HR


class ProcessReward(BaseRewardFunction):
    """
    ProcessReward
    角度优势、能量优势
    与敌方飞机的距离差奖励和高度差奖励

    """
    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)         # km
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)     # km
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)     # mh

        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id):
        reward = 0
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        # 计算与存活敌我飞机间的态势
        for enm in env.agents[agent_id].enemies:
            if enm.is_alive:
                enm_feature = np.hstack([enm.get_position(),
                                        enm.get_velocity()])
                # 获取进入角和天线偏置角和距离差
                AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
                # 获取敌我能量高度和高度差
                HO, HT, H = get_HR(ego_feature, enm_feature)

                # 角度优势（-1，1）
                r_a = 1 - (AO + TA)/np.pi
                # 能量优势（-1，1）
                r_e = (HO - HT) / (HO + HT)
                # 距离差（0,1)
                r_R = (10000 - R) / 10000
                # 高度差(0,1)
                r_H = (1200 - H) / 1200

                # 暂时先取固定权重(固定0.1为生存奖励)
                reward += 0.05 * r_a + 0.05 * r_e + 0.001 * r_R + 0.001 * r_H
                # print(f"{agent_id}---{0.005 * r_a}---{0.005 * r_e}---{0.0001 * r_R}---{0.0001 * r_H}")
        return self._process(reward, agent_id)
