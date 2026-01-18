from .reward_function_base import BaseRewardFunction


class ResultReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """

    def __init__(self, config):
        super().__init__(config)
        self.last_time_missiles = None
        self.num = {}  # 控制结果奖励只能累加一次

    def reset(self, task, env):
        self.num = {agent_id: 0 for agent_id, agent in env.agents.items()}
        self.last_time_missiles = {agent_id: 0 for agent_id, agent in env.agents.items()}
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
        ego = env.agents[agent_id]
        enm = env.agents[agent_id].enemies[0]
        reward = 0
        alive_missiles = 0

        # 攻击方
        if agent_id == 'A0100':
            # 超时未击毁敌机或自己坠机
            if env.current_step >= 1000 or ego.is_crash:
                reward = -200
            # 击毁敌机或敌方自己坠机
            elif enm.is_crash or enm.is_shotdown:
                reward = 200
            else:
                reward = -0.05
        elif agent_id == 'B0100':
            # 生存达到目标时间或敌方死亡
            if env.current_step >= 1000 or enm.is_crash:
                reward = 200
            elif ego.is_crash or ego.is_shotdown:
                reward = -200
            else:
                reward = 0.05

            for missile in ego.under_missiles:
                if missile.is_alive:
                    alive_missiles += 1

            # 成功规避导弹
            if ego.is_alive:
                if alive_missiles < self.last_time_missiles[agent_id]:
                    reward += 30
                # 保存上一个时间步存活的导弹数量
                self.last_time_missiles[agent_id] = alive_missiles
        return self._process(reward, agent_id)
