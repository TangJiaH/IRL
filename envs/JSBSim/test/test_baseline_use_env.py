from abc import ABC
import sys
import os
from gymnasium import spaces

# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Literal
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import in_range_rad, get_root_dir, LLA2NEU, get2d_AO_TA_R
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv
from envs.JSBSim.model.baseline_actor import BaselineActor


class BaselineAgent(ABC):
    def __init__(self, agent_id) -> None:
        self.model_path = get_root_dir() + '/model/baseline_model.pt'
        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.actor.eval()
        self.agent_id = agent_id
        self.state_var = [
            c.delta_altitude,  # 0. delta_h   (unit: m)
            c.delta_heading,  # 1. delta_heading  (unit: °)
            c.delta_velocities_u,  # 2. delta_v   (unit: m/s)
            c.attitude_roll_rad,  # 3. roll      (unit: rad)
            c.attitude_pitch_rad,  # 4. pitch     (unit: rad)
            c.velocities_u_mps,  # 5. v_body_x   (unit: m/s)
            c.velocities_v_mps,  # 6. v_body_y   (unit: m/s)
            c.velocities_w_mps,  # 7. v_body_z   (unit: m/s)
            c.velocities_vc_mps,  # 8. vc        (unit: m/s)
            c.position_h_sl_m  # 9. altitude  (unit: m)
        ]
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))

    @abstractmethod
    def set_delta_value(self, env, task):
        raise NotImplementedError

    def get_observation(self, env, task, delta_value):
        uid = list(env.agents.keys())[self.agent_id]
        obs = env.agents[uid].get_property_values(self.state_var)
        norm_obs = np.zeros(12)
        norm_obs[0] = delta_value[0] / 1000  # 0. ego delta altitude  (unit: 1km)
        norm_obs[1] = in_range_rad(delta_value[1])  # 1. ego delta heading   (unit rad)
        norm_obs[2] = delta_value[2] / 340  # 2. ego delta velocities_u  (unit: mh)
        norm_obs[3] = obs[9] / 5000  # 3. ego_altitude (unit: km)
        norm_obs[4] = np.sin(obs[3])  # 4. ego_roll_sin
        norm_obs[5] = np.cos(obs[3])  # 5. ego_roll_cos
        norm_obs[6] = np.sin(obs[4])  # 6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[4])  # 7. ego_pitch_cos
        norm_obs[8] = obs[5] / 340  # 8. ego_v_x   (unit: mh)
        norm_obs[9] = obs[6] / 340  # 9. ego_v_y    (unit: mh)
        norm_obs[10] = obs[7] / 340  # 10. ego_v_z    (unit: mh)
        norm_obs[11] = obs[8] / 340  # 11. ego_vc        (unit: mh)
        norm_obs = np.expand_dims(norm_obs, axis=0)  # dim: (1,12)
        return norm_obs

    def get_action(self, env, task):
        delta_value = self.set_delta_value(env, task)
        observation = self.get_observation(env, task, delta_value)
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        return action


class PursueAgent(BaselineAgent):
    def __init__(self, agent_id) -> None:
        super().__init__(agent_id)

    def set_delta_value(self, env, task):
        # NOTE: only adapt for 1v1
        ego_uid, enm_uid = list(env.agents.keys())[self.agent_id], list(env.agents.keys())[(self.agent_id + 1) % 2]
        ego_x, ego_y, ego_z = env.agents[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.agents[ego_uid].get_velocity()
        enm_x, enm_y, enm_z = env.agents[enm_uid].get_position()

        # delta altitude
        delta_altitude = enm_z - ego_z
        # delta heading
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        # delta velocity
        delta_velocity = env.agents[enm_uid].get_property_value(c.velocities_u_mps) - \
                         env.agents[ego_uid].get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])


class HighYoYoAgent(BaselineAgent):
    def __init__(self, agent_id) -> None:
        super().__init__(agent_id)
        self.norm_altitude = np.array([0.1, 0, -0.1])
        self.norm_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_velocity = np.array([0.05, 0, -0.05])
        self.phase = 0  # High Yo Yo动作目前执行的阶段

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(15,))

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space
        """
        norm_obs = np.zeros(15)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # Extract feature
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])

        # Normalize ego info
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340

        # Relative info normalization
        ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag

        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def set_delta_value(self, env, task):
        """
        Calculate the values for High Yo-Yo based on current observation and task.
        """
        obs = self.get_obs(env, self.agent_id)
        ego_AO = obs[11]  # Extracted from get_obs
        R = obs[13] * 10000  # Convert back to meters
        side_flag = obs[14]
        relative_velocity = obs[9] * 340  # Relative speed in m/s
        relative_altitude = obs[10] * 1000  # Altitude difference in meters

        # Default values

        delta_altitude = self.norm_altitude[0]  # Maintain altitude
        delta_heading = self.norm_heading[2]  # Maintain heading
        delta_velocity = self.norm_velocity[1]  # 加速追击

        # Determine High Yo-Yo action
        # 阶段0： 初始化滚转和爬升
        if self.phase == 0:
            if np.pi / 6 <= ego_AO <= np.pi / 3 and R < 3000:
                self.phase = 1
                delta_altitude = self.norm_altitude[0]  # 爬升
                delta_heading = self.norm_heading[0] if side_flag > 0 else self.norm_heading[4]  # 根据目标所在侧进行滚转
                delta_velocity = self.norm_velocity[2]  # 减速
        # 阶段1：持续爬升并减速
        elif self.phase == 1:
            if relative_velocity < 10:
                self.phase = 2
            delta_altitude = self.norm_altitude[0]  # 爬升
            delta_heading = self.norm_heading[1]  # 保持
            delta_velocity = self.norm_velocity[2]  # 减速
        # 阶段2：高点转向
        elif self.phase == 2:
            if relative_altitude > 0:
                self.phase = 3
            delta_altitude = self.norm_altitude[1]  # 停止爬升
            delta_heading = self.norm_heading[1]  # 保持
            delta_velocity = self.norm_velocity[1]  # 匀速
        elif self.phase == 3:
            if R > 4000:
                self.phase = 0
            delta_altitude = self.norm_altitude[2]  # 俯冲
            delta_heading = self.norm_heading[1]  # 保持
            delta_velocity = self.norm_velocity[0]  # 加速追击
        print(np.array([delta_altitude, delta_heading, delta_velocity]))
        return np.array([delta_altitude, delta_heading, delta_velocity])


class ManeuverAgent(BaselineAgent):
    def __init__(self, agent_id, maneuver: Literal['l', 'r', 'n']) -> None:
        super().__init__(agent_id)
        self.turn_interval = 30
        self.dodge_missile = False  # if set true, start turn when missile is detected
        if maneuver == 'l':
            self.target_heading_list = [0]
        elif maneuver == 'r':
            self.target_heading_list = [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]
        elif maneuver == 'n':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi]
        elif maneuver == 'triangle':
            self.target_heading_list = [np.pi / 3, np.pi, -np.pi / 3] * 2
        self.target_altitude_list = [6000] * 6
        self.target_velocity_list = [243] * 6

    def reset(self):
        self.step = 0
        self.rnn_states = np.zeros((1, 1, 128))
        self.init_heading = None

    def set_delta_value(self, env, task):
        step_list = np.arange(1, len(self.target_heading_list) + 1) * self.turn_interval / env.time_interval
        uid = list(env.agents.keys())[self.agent_id]
        cur_heading = env.agents[uid].get_property_value(c.attitude_heading_true_rad)
        if self.init_heading is None:
            self.init_heading = cur_heading
        if not self.dodge_missile or task._check_missile_warning(env, self.agent_id) is not None:
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
            delta_heading = self.init_heading + self.target_heading_list[i] - cur_heading
            delta_altitude = self.target_altitude_list[i] - env.agents[uid].get_property_value(c.position_h_sl_m)
            delta_velocity = self.target_velocity_list[i] - env.agents[uid].get_property_value(c.velocities_u_mps)
            self.step += 1
        else:
            delta_heading = self.init_heading - cur_heading
            delta_altitude = 6000 - env.agents[uid].get_property_value(c.position_h_sl_m)
            delta_velocity = 243 - env.agents[uid].get_property_value(c.velocities_u_mps)

        return np.array([delta_altitude, delta_heading, delta_velocity])


def test_maneuver():
    env = SingleCombatEnv(config_name='1v1/NoWeapon/test/opposite')
    obs = env.reset()
    env.render(filepath="control.txt.acmi")
    agent0 = HighYoYoAgent(agent_id=0)
    agent1 = ManeuverAgent(agent_id=1, maneuver='n')
    reward_list = []
    while True:
        action0 = agent0.get_action(env, env.task)
        action1 = agent1.get_action(env, env.task)
        actions = [action0, action1]
        obs, reward, done, info = env.step(actions)
        env.render(filepath="control.txt.acmi")
        reward_list.append(reward[0])
        if np.array(done).all():
            print(info)
            break
    # plt.plot(reward_list)
    # plt.savefig('rewards.png')


if __name__ == '__main__':
    test_maneuver()
