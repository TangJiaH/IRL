from .reward_function_base import BaseRewardFunction


class ShootPenaltyReward(BaseRewardFunction):
    """
    ShootPenaltyReward
    when launching a missile, give -10 reward for penalty,
    to avoid launching all missiles at once 
    """
    def __init__(self, config):
        super().__init__(config)
        self.last_missile_status = {}
        self.pre_remaining_missiles = {}

    def reset(self, task, env):
        # self.pre_remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        for agent_id, agent in env.agents.items():
            self.pre_remaining_missiles[agent_id] = agent.num_missiles
            for missile in env.agents[agent_id].launch_missiles:
                uid, status = missile.get_status()
                self.last_missile_status[uid] = status
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
        # 发射成功的奖励
        reward = 0
        if task.remaining_missiles[agent_id] == self.pre_remaining_missiles[agent_id] - 1:
            reward += 5
            # print("发射，加分")
        self.pre_remaining_missiles[agent_id] = task.remaining_missiles[agent_id]

        # 未命中的惩罚
        for missile in env.agents[agent_id].launch_missiles:
            uid, status = missile.get_status()
            if status == 2 and self.last_missile_status[uid] == 0:
                # print("未命中，扣分")
                reward -= 10
            self.last_missile_status[uid] = status
        return self._process(reward, agent_id)