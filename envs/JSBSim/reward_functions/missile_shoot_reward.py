from .reward_function_base import BaseRewardFunction


class MissileShootReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    - lauch a new missiles: +20
    - missiles is miss: -30
    """

    def __init__(self, config):
        super().__init__(config)
        self.last_status = None

    def reset(self, task, env):
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
        for missile in env.agents[agent_id].launch_missiles:
            if missile.is_hit():
                reward += 200
                missile.hit_reward()

        return self._process(reward, agent_id)
