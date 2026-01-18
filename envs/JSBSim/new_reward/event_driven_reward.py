from .reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """

    def __init__(self, config):
        super().__init__(config)
        self.last_status = None

    def reset(self, task, env):
        self.last_status = {agent_id: agent.is_alive for agent_id, agent in env.agents.items()}
        # print(self.last_status)
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
        if env.agents[agent_id].is_shotdown:
            print(f"{agent_id}被导弹击中")
            reward -= 200
        elif env.agents[agent_id].is_crash:
            print(f"{agent_id}坠毁")
            reward -= 200
        for missile in env.agents[agent_id].launch_missiles:
            if missile.is_hit():
                print(f"{agent_id}使用导弹击中成功")
                reward += 200
                missile.hit_reward()

        self.last_status[agent_id] = env.agents[agent_id].is_alive
        return self._process(reward, agent_id)
