from abc import ABC, abstractmethod
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

from algorithms.ppo.ppo_actor import PPOActor
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import in_range_rad, get_root_dir
from envs.JSBSim.envs import SingleCombatEnv
from envs.JSBSim.model.baseline_actor import BaselineActor
from renders.render_1v1 import ManeuverAgent

# Add the necessary import from your PPO implementation
# Assume that PPOActor is already defined as per the first code snippet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaselineAgent(ABC):
    def __init__(self, agent_id) -> None:
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

    @abstractmethod
    def get_action(self, env, task):
        pass


# Modify PursueAgent to use the PPO model
class PursueAgent(BaselineAgent):
    def __init__(self, agent_id, ppo_policy_path, args, env) -> None:
        super().__init__(agent_id)
        self.ppo_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
        self.ppo_policy.load_state_dict(torch.load(ppo_policy_path, map_location=device))
        self.ppo_policy.eval()

    def set_delta_value(self, env, task):
        # Calculate delta altitude, heading, and velocity
        ego_uid, enm_uid = list(env.agents.keys())[self.agent_id], list(env.agents.keys())[(self.agent_id + 1) % 2]
        ego_x, ego_y, ego_z = env.agents[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.agents[ego_uid].get_velocity()
        enm_x, enm_y, enm_z = env.agents[enm_uid].get_position()
        delta_altitude = enm_z - ego_z
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        delta_velocity = env.agents[enm_uid].get_property_value(c.velocities_u_mps) - env.agents[
            ego_uid].get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])

    def get_action(self, env, task):
        delta_value = self.set_delta_value(env, task)
        observation = self.get_observation(env, task, delta_value)
        # Use PPO model for action selection
        _action, self.rnn_states = self.ppo_policy(observation, self.rnn_states, deterministic=True)
        action = _action.detach().cpu().numpy().squeeze()
        return action


# No change for ManeuverAgent, it will still use the BaselineActor
class ManeuverAgent(BaselineAgent):
    def __init__(self, agent_id, maneuver: Literal['l', 'r', 'n']) -> None:
        super().__init__(agent_id)
        self.turn_interval = 30
        self.dodge_missile = False  # if set true, start turn when missile is detected
        print(maneuver)
        if maneuver == 'l':
            self.target_heading_list = [0]
        elif maneuver == 'r':
            self.target_heading_list = [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]
        elif maneuver == 'n':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi]
        elif maneuver == 'triangle':
            print('triangle')
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
            return np.array([delta_altitude, delta_heading, delta_velocity])

        def get_action(self, env, task):
            delta_value = self.set_delta_value(env, task)
            observation = self.get_observation(env, task, delta_value)
            _action, self.rnn_states = self.actor(observation, self.rnn_states)
            action = _action.detach().cpu().numpy().squeeze()
            return action

    # Assuming that PPOActor and the args variable are properly defined elsewhere in the code.
    # Example usage of both agents:

    # Environment and arguments initialization
    # Initialize SingleCombatEnv or other appropriate environment
    env = SingleCombatEnv()

    # Initialize the PPOAgent with PPO-trained model path
    ppo_model_path = get_root_dir() + '/model/ppo_policy.pt'
    args = None  # Define or pass necessary arguments for PPOActor

    # Create agents
    pursue_agent = PursueAgent(agent_id=0, ppo_policy_path=ppo_model_path, args=args, env=env)
    maneuver_agent = ManeuverAgent(agent_id=1, maneuver='r')

    # Simulation loop (example):
    for episode in range(10):  # Assume running 10 episodes
        env.reset()
        pursue_agent.reset()
        maneuver_agent.reset()

        for step in range(100):  # Run for 100 steps or till termination
            task = None  # Replace with an appropriate task object if required

            # Get actions from both agents
            pursue_action = pursue_agent.get_action(env, task)
            maneuver_action = maneuver_agent.get_action(env, task)

            # Take a step in the environment with both agents' actions
            env.step({0: pursue_action, 1: maneuver_action})

            # Render or log as needed
            env.render()
