import numpy as np
from .reward_function_base import BaseRewardFunction


class BoundaryReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        # 建议对这些半径进行仔细调整以适应您的特定场景
        # safe_radius: 智能体可以自由活动的安全区域半径 (km)
        self.safe_radius = getattr(self.config, f'{self.__class__.__name__}_safe_radius', 11)  # km
        # danger_radius: 绝对不应穿越的危险区域半径 (km)
        self.danger_radius = getattr(self.config, f'{self.__class__.__name__}_danger_radius', 16)  # km

        # 警告区惩罚系数，显著增大了默认值以提供更强的早期威慑
        # 这个值需要仔细调整，太小则效果不明显，太大可能导致智能体过于保守
        self.penalty_scale = getattr(self.config, f'{self.__class__.__name__}_penalty_scale', 9.0)

        # 飞出危险区域的巨大惩罚值。
        # 即便您在别处处理了坠毁，如果此处的 crash_penalty 预期被使用，
        # 强烈建议将其设置为一个远大于其他任何单步奖励绝对值的负数。
        # 例如: -200, -500, -1000 等，取决于您环境中奖励的整体尺度。
        # 如果完全由外部逻辑处理坠毁，此处的默认值可能不敏感，但设置一个警示性的默认值是好的实践。
        self.crash_penalty = getattr(self.config, f'{self.__class__.__name__}_crash_penalty', -20.0)

        # 确保 danger_radius > safe_radius
        assert self.danger_radius > self.safe_radius, "danger_radius must be greater than safe_radius"

        self.reward_item_names = [self.__class__.__name__]

    def get_reward(self, task, env, agent_id):
        """
        根据飞机与中心的距离计算奖励，
        惩罚接近或超过 danger_radius 的行为。

        Args:
            task: task 实例
            env: environment 实例
            agent_id: agent ID

        Returns:
            (float): 奖励值
        """
        # 获取飞机位置并转换为公里
        ego_pos = env.agents[agent_id].get_position() / 1000
        ego_x = ego_pos[0]
        ego_y = ego_pos[1]

        # 计算离中心的距离
        distance = np.sqrt(ego_x ** 2 + ego_y ** 2)

        reward = 0.0

        if distance < self.safe_radius:
            # 在安全区内，没有惩罚
            reward = 0.0
        elif distance >= self.danger_radius:
            # 飞出危险区，给予巨大惩罚 (如果此处的 crash_penalty 生效)
            reward = self.crash_penalty
            # print(f"CRASH: Agent {agent_id} flew out of bounds. Distance: {distance:.2f} km, Reward: {reward}")
        else:
            # 在警告区 (safe_radius <= distance < danger_radius)，惩罚随距离二次方增加
            # normalized_penetration 为0表示刚进入警告区，为1表示即将触碰危险边界
            normalized_penetration = (distance - self.safe_radius) / (self.danger_radius - self.safe_radius)
            reward = -self.penalty_scale * (normalized_penetration ** 2)
            # print(f"WARN: Agent {agent_id} in warning zone. Distance: {distance:.2f} km, Penetration: {normalized_penetration:.2f}, PosReward: {reward:.2f}")

        # (可选) 增加速度惩罚: 如果飞机在警告区边缘或更外侧，并且正向外飞，则增加惩罚
        # 将速度惩罚的触发条件稍微提前，并增加惩罚系数
        # 例如，当距离大于 safe_radius * 0.85 (原为 0.9) 时就开始检查
        # 速度惩罚系数从 0.8 增加到 1.5
        # 注意：此速度惩罚与位置惩罚是叠加的（如果在警告区内且向外飞）
        velocity_penalty_trigger_ratio = 0.85
        velocity_penalty_scale = 1.5

        # 仅在接近或已在警告区时考虑速度惩罚
        if distance > self.safe_radius * velocity_penalty_trigger_ratio:
            ego_vel_vector_km_s = env.agents[agent_id].get_velocity() / 1000  # 转换为 km/s
            # 将速度单位转换为马赫数 (假设声速约为 340 m/s = 0.34 km/s)
            # 或者，直接使用 km/s 进行计算，避免引入马赫数的复杂性，并调整系数
            # 这里我们继续使用马赫数作为示例，但请确认这是否适合您的系统
            # sound_speed_km_s = 0.340
            # ego_vel_mach_vector = ego_vel_vector_km_s / sound_speed_km_s
            # vel_x_mach = ego_vel_mach_vector[0]
            # vel_y_mach = ego_vel_mach_vector[1]

            # 直接使用 km/s 单位的速度向量进行径向速度计算，这样系数的物理意义更直接
            pos_vector_km = np.array([ego_x, ego_y])  # 已经是 km

            # 计算径向速度 (速度在位置向量上的投影，单位 km/s)
            # 如果 distance 为 0 (不太可能在这里发生，因为有safe_radius)，会导致除零错误
            if distance > 1e-6:  # 避免除以零
                # radial_velocity_km_s = np.dot(ego_vel_vector_km_s[:2], pos_vector_km) / distance
                # 使用速度向量的前两个分量，假设Z轴速度不参与边界判断
                current_velocity_vector_km_s = ego_vel_vector_km_s[:2]
                radial_velocity_km_s = np.dot(current_velocity_vector_km_s, pos_vector_km) / distance
            else:
                radial_velocity_km_s = 0

            if radial_velocity_km_s > 0:  # 如果正在向外飞 (径向速度为正)
                # 施加额外惩罚，惩罚大小与径向速度和设定的系数成正比
                # 这个惩罚值现在是基于 km/s 的速度，系数 velocity_penalty_scale 需要相应调整
                # 如果 radial_velocity_km_s 的典型值是 0.1 km/s (约 1/3 马赫), 那么惩罚是 -1.5 * 0.1 = -0.15
                # 您需要根据智能体的典型速度调整 velocity_penalty_scale
                additional_vel_penalty = -velocity_penalty_scale * radial_velocity_km_s
                reward += additional_vel_penalty  # 注意是叠加惩罚
                # print(f"VEL_WARN: Agent {agent_id} moving outwards near boundary. Radial Vel: {radial_velocity_km_s:.2f} km/s, VelPenalty: {additional_vel_penalty:.2f}, Total Reward: {reward:.2f}")

        return self._process(reward, agent_id)