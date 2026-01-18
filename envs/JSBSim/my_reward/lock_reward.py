import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R, get_HR, cup_reward, cup_reward_2


class LockReward(BaseRewardFunction):
    """
    LockReward
    敌方进入导弹锁定范围

    """
    def __init__(self, config):
        super().__init__(config)
        self.max_lock_distance = getattr(self.config, 'max_lock_distance', np.inf)
        self.min_lock_distance = getattr(self.config, 'min_lock_distance', 0)
        self.max_lock_angle = getattr(self.config, 'max_lock_angle', 180)
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)     # mh

        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id):
        reward = 0
        r_1 = 0
        r_2 = 0
        r_3 = 0
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        # 计算与存活敌我飞机间的态势
        for enm in env.agents[agent_id].enemies:
            if enm.is_alive:
                enm_feature = np.hstack([enm.get_position(),
                                         enm.get_velocity()])
                AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
                T_AO, _, _ = get_AO_TA_R(enm_feature, ego_feature)
                attack_angle = np.rad2deg(AO)
                T_attack_angle = np.rad2deg(T_AO)

                lock_flag = (
                        attack_angle <= self.max_lock_angle and
                        self.min_lock_distance <= R <= self.max_lock_distance
                )

                T_lock_flag = (
                        T_attack_angle <= self.max_lock_angle and
                        self.min_lock_distance <= R <= self.max_lock_distance
                )

                # # 锁定判断
                # if lock_flag:
                #     r_1 += 0.5
                #     # print(f"current_step:{env.current_step}--lock_flag:{lock_flag}")
                # if T_lock_flag:
                #     r_2 -= 0.5
                #     # print(f"current_step:{env.current_step}--lock_flag:{T_lock_flag}")
                #
                # if lock_flag and T_lock_flag:
                #     r_3 -= 2


                # 角度范围
                lock_ao = cup_reward_2(np.rad2deg(AO), -20, 20, 1.3)

                # 距离范围
                lock_r = cup_reward_2(R / 1000, 2.5, 4.5, 2.3)

                # 角度范围
                threat_ao = cup_reward_2(np.rad2deg(T_AO), -20, 20, 1.3)

                # 距离范围
                threat_r = cup_reward_2(R / 1000, 2.5, 4.5, 2.3)

                reward += 0.4 * lock_ao * lock_r - 0.4 * threat_ao * threat_r
        return self._process(reward, agent_id)
