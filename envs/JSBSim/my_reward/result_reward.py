import numpy as np
from .reward_function_base import BaseRewardFunction


class ResultReward(BaseRewardFunction):
    """
    ResultReward
    杀敌、撞地、飞出边界（惩戒性奖励）

    """

    def __init__(self, config):
        super().__init__(config)
        self.last_status = {}

    def reset(self, task, env):
        self.last_status = {agent_id: agent.is_alive for agent_id, agent in env.agents.items()}
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0

        # 被击中或坠毁
        if env.agents[agent_id].is_shotdown:
            print(f"{agent_id}被击中")
            reward -= 50
        elif env.agents[agent_id].is_crash:
            print(f"{agent_id}坠毁")
            reward -= 50

        for enm in env.agents[agent_id].enemies:
            if not enm.is_dead:
                if enm.is_shotdown:
                    print(f"{agent_id}的敌机被击中")
                    reward += 50
                    enm.dead()
                elif enm.is_crash:
                    print(f"{agent_id}的敌机坠毁")
                    reward += 50
                    enm.dead()
        self.last_status[agent_id] = (env.agents[agent_id].is_alive or env.agents[agent_id].is_lockon)
        return self._process(reward, agent_id)
