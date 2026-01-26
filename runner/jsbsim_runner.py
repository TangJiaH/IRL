import os
import time
import csv
import torch
import logging
import numpy as np
from typing import List
from .base_runner import Runner, ReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class JSBSimRunner(Runner):

    def load(self):
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.use_selfplay = self.all_args.use_selfplay

        # policy & algorithm
        if self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, self.policy, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        self.warmup()

        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads
        log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        train_log_path = os.path.join(log_dir, "train_metrics.csv")
        if not os.path.exists(train_log_path):
            with open(train_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "total_steps",
                    "episode_return",
                    "episode_length",
                    "success_flag",
                    "termination_reason",
                    "action_change_count",
                    "action_delta_sum",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl",
                    "clip_fraction",
                    "average_heading_turns",
                ])

        for episode in range(episodes):

            self.current_episode = episode
            heading_turns_list = []
            episode_returns = np.zeros(self.n_rollout_threads, dtype=np.float32)
            episode_lengths = np.zeros(self.n_rollout_threads, dtype=np.int32)
            last_actions = None
            action_change_counts = np.zeros(self.n_rollout_threads, dtype=np.int32)
            action_delta_sum = np.zeros(self.n_rollout_threads, dtype=np.float32)

            for step in range(self.buffer_size):
                # Sample actions
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)

                # Extra recorded information
                for info in infos:
                    if 'heading_turn_counts' in info:
                        heading_turns_list.append(info['heading_turn_counts'])

                step_rewards = rewards.squeeze(-1).sum(axis=1)
                episode_returns += step_rewards
                episode_lengths += 1
                if last_actions is not None:
                    action_change_counts += (actions != last_actions).any(axis=(1, 2))
                    action_delta_sum += np.abs(actions - last_actions).sum(axis=(1, 2))
                last_actions = actions.copy()

                data = obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic

                # insert data into buffer
                self.insert(data)

                dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
                if dones_env.any():
                    for env_idx, done_flag in enumerate(dones_env):
                        if not done_flag:
                            continue
                        info = infos[env_idx] if env_idx < len(infos) else {}
                        termination_reason = "timeout" if info.get("current_step", 0) >= self.envs.envs[0].max_steps else "terminal"
                        success_flag = 1 if info.get("heading_turn_counts", 0) > 0 else 0
                        if self.use_wandb:
                            step_num = self.total_num_steps + (step + 1) * self.n_rollout_threads
                            episode_logs = {
                                "episode/return": float(episode_returns[env_idx]),
                                "episode/length": int(episode_lengths[env_idx]),
                                "episode/success": int(success_flag),
                                "episode/termination_reason": termination_reason,
                                "episode/action_change_count": int(action_change_counts[env_idx]),
                                "episode/action_delta_sum": float(action_delta_sum[env_idx]),
                                "episode/heading_turn_counts": float(info.get("heading_turn_counts", 0)),
                            }
                            self.log_info(episode_logs, step_num)
                        with open(train_log_path, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                episode,
                                self.total_num_steps,
                                float(episode_returns[env_idx]),
                                int(episode_lengths[env_idx]),
                                int(success_flag),
                                termination_reason,
                                int(action_change_counts[env_idx]),
                                float(action_delta_sum[env_idx]),
                                "",
                                "",
                                "",
                                "",
                                "",
                                info.get("heading_turn_counts", ""),
                            ])
                        episode_returns[env_idx] = 0.0
                        episode_lengths[env_idx] = 0
                        action_change_counts[env_idx] = 0
                        action_delta_sum[env_idx] = 0.0

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                logging.info("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                             .format(self.all_args.scenario_name,
                                     self.algorithm_name,
                                     self.experiment_name,
                                     episode,
                                     episodes,
                                     self.total_num_steps,
                                     self.num_env_steps,
                                     int(self.total_num_steps / (end - start))))

                train_infos["average_episode_rewards"] = self.buffer.rewards.sum() / (self.buffer.masks == False).sum()
                logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

                if len(heading_turns_list):
                    train_infos["average_heading_turns"] = np.mean(heading_turns_list)
                    logging.info("average heading turns is {}".format(train_infos["average_heading_turns"]))
                self.log_info(train_infos, self.total_num_steps)
                with open(train_log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode,
                        self.total_num_steps,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        train_infos.get("policy_loss", ""),
                        train_infos.get("value_loss", ""),
                        train_infos.get("policy_entropy_loss", ""),
                        train_infos.get("approx_kl", ""),
                        train_infos.get("clip_fraction", ""),
                        train_infos.get("average_heading_turns", ""),
                    ])

            # eval
            if episode % self.eval_interval == 0 and episode != 0 and self.use_eval:
                self.eval(self.total_num_steps)

            # save model
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

        self._plot_training_curves(train_log_path, log_dir)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        # split parallel data [N*M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data

        dones_env = np.all(dones.squeeze(axis=-1), axis=-1)

        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)

        self.timestamp = 0  # use for tacview real time render
        interval_timestamp = self.envs.envs[0].agent_interaction_steps / self.envs.envs[0].sim_freq
        if self.render_mode == "real_time" and self.tacview:  # 重新连接
            print("重新连接tacview.....")
            self.tacview.reconnect()
        #  Create a directory to save .acmi files only use for render mode is histroy_acmi
        save_dir = os.path.join(self.run_dir, 'acmi_files')
        os.makedirs(save_dir, exist_ok=True)
        acmi_filename = f"{save_dir}/eval_episode_{self.current_episode}.acmi"
        eval_log_dir = os.path.join(self.run_dir, "eval_logs")
        os.makedirs(eval_log_dir, exist_ok=True)
        timeseries_path = os.path.join(eval_log_dir, f"eval_timeseries_{total_num_steps}.csv")
        summary_path = os.path.join(eval_log_dir, f"eval_summary_{total_num_steps}.csv")
        with open(timeseries_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode_id",
                "env_idx",
                "step_idx",
                "delta_heading_rad",
                "delta_altitude_m",
                "delta_speed_mps",
                "roll_rad",
                "action_aileron_idx",
                "action_elevator_idx",
                "action_rudder_idx",
                "action_throttle_idx",
                "action_aileron",
                "action_elevator",
                "action_rudder",
                "action_throttle",
                "reward",
                "done",
                "termination_reason",
            ])

        episode_steps = np.zeros(self.n_eval_rollout_threads, dtype=np.int32)
        episode_ids = np.zeros(self.n_eval_rollout_threads, dtype=np.int32)
        last_actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.act_space.shape[0]), dtype=np.float32)
        delta_heading_sq = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)
        delta_altitude_sq = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)
        delta_speed_sq = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)
        max_abs_phi = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)
        action_change_counts = np.zeros(self.n_eval_rollout_threads, dtype=np.int32)
        action_delta_sum = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)

        while total_episodes < self.eval_episodes:

            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
                                                            np.concatenate(eval_rnn_states),
                                                            np.concatenate(eval_masks), deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)

            for env_idx in range(self.n_eval_rollout_threads):
                obs = eval_obs[env_idx, 0]
                action = eval_actions[env_idx, 0]
                delta_heading_rad = obs[1]
                delta_altitude_m = obs[0] * 1000.0
                delta_speed_mps = obs[2] * 340.0
                roll_rad = np.arctan2(obs[4], obs[5])
                max_abs_phi[env_idx] = max(max_abs_phi[env_idx], abs(roll_rad))
                delta_heading_sq[env_idx] += delta_heading_rad ** 2
                delta_altitude_sq[env_idx] += delta_altitude_m ** 2
                delta_speed_sq[env_idx] += delta_speed_mps ** 2
                if episode_steps[env_idx] > 0:
                    action_change_counts[env_idx] += int((action != last_actions[env_idx]).any())
                    action_delta_sum[env_idx] += np.abs(action - last_actions[env_idx]).sum()
                last_actions[env_idx] = action
                episode_steps[env_idx] += 1

                aileron = action[0] * 2.0 / (self.act_space.nvec[0] - 1.0) - 1.0
                elevator = action[1] * 2.0 / (self.act_space.nvec[1] - 1.0) - 1.0
                rudder = action[2] * 2.0 / (self.act_space.nvec[2] - 1.0) - 1.0
                throttle = action[3] * 0.5 / (self.act_space.nvec[3] - 1.0) + 0.4

                termination_reason = ""
                if eval_dones_env[env_idx]:
                    info = eval_infos[env_idx] if env_idx < len(eval_infos) else {}
                    termination_reason = "timeout" if info.get("current_step", 0) >= self.eval_envs.envs[0].max_steps else "terminal"

                with open(timeseries_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        int(episode_ids[env_idx]),
                        int(env_idx),
                        int(episode_steps[env_idx]),
                        float(delta_heading_rad),
                        float(delta_altitude_m),
                        float(delta_speed_mps),
                        float(roll_rad),
                        int(action[0]),
                        int(action[1]),
                        int(action[2]),
                        int(action[3]),
                        float(aileron),
                        float(elevator),
                        float(rudder),
                        float(throttle),
                        float(eval_rewards[env_idx].sum()),
                        int(eval_dones_env[env_idx]),
                        termination_reason,
                    ])

            # render with tacview
            self.eval_envs.envs[0].render_with_tacview(self.render_mode, self.tacview, acmi_filename,
                                                       self.eval_envs.envs[0], self.timestamp, self._should_save_acmi())

            self.timestamp += interval_timestamp  # step 0.2s

            eval_cumulative_rewards += eval_rewards
            total_episodes += np.sum(eval_dones_env)
            eval_episode_rewards.append(eval_cumulative_rewards[eval_dones_env == True])
            eval_cumulative_rewards[eval_dones_env == True] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]), dtype=np.float32)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)

            for env_idx, done_flag in enumerate(eval_dones_env):
                if not done_flag:
                    continue
                length = max(episode_steps[env_idx], 1)
                rmse_heading = float(np.sqrt(delta_heading_sq[env_idx] / length))
                rmse_altitude = float(np.sqrt(delta_altitude_sq[env_idx] / length))
                rmse_speed = float(np.sqrt(delta_speed_sq[env_idx] / length))
                info = eval_infos[env_idx] if env_idx < len(eval_infos) else {}
                termination_reason = "timeout" if info.get("current_step", 0) >= self.eval_envs.envs[0].max_steps else "terminal"
                write_header = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
                with open(summary_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow([
                            "episode_id",
                            "env_idx",
                            "episode_return",
                            "episode_length",
                            "rmse_heading_rad",
                            "rmse_altitude_m",
                            "rmse_speed_mps",
                            "max_abs_phi",
                            "action_change_count",
                            "action_delta_sum",
                            "termination_reason",
                        ])
                    writer.writerow([
                        int(episode_ids[env_idx]),
                        int(env_idx),
                        float(eval_episode_rewards[-1].sum()),
                        int(length),
                        rmse_heading,
                        rmse_altitude,
                        rmse_speed,
                        float(max_abs_phi[env_idx]),
                        int(action_change_counts[env_idx]),
                        float(action_delta_sum[env_idx]),
                        termination_reason,
                    ])
                episode_ids[env_idx] += 1
                episode_steps[env_idx] = 0
                delta_heading_sq[env_idx] = 0.0
                delta_altitude_sq[env_idx] = 0.0
                delta_speed_sq[env_idx] = 0.0
                max_abs_phi[env_idx] = 0.0
                action_change_counts[env_idx] = 0
                action_delta_sum[env_idx] = 0.0

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean(axis=1)  # shape: [num_agents, 1]
        logging.info(" eval average episode rewards: " + str(np.mean(eval_infos['eval_average_episode_rewards'])))
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

    def _plot_training_curves(self, train_log_path: str, log_dir: str) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            logging.warning("matplotlib 未安装，跳过训练曲线绘制。")
            return

        episodes = []
        returns = []
        policy_losses = []
        value_losses = []
        entropies = []
        with open(train_log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("episode_return"):
                    episodes.append(int(row["episode"]))
                    returns.append(float(row["episode_return"]))
                if row.get("policy_loss"):
                    policy_losses.append(float(row["policy_loss"]))
                if row.get("value_loss"):
                    value_losses.append(float(row["value_loss"]))
                if row.get("entropy"):
                    entropies.append(float(row["entropy"]))

        plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        if returns:
            plt.figure()
            plt.plot(episodes, returns, label="episode_return")
            plt.xlabel("episode")
            plt.ylabel("return")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "train_return.png"))
            plt.close()
        if policy_losses:
            plt.figure()
            plt.plot(policy_losses, label="policy_loss")
            plt.plot(value_losses, label="value_loss")
            plt.plot(entropies, label="entropy")
            plt.xlabel("update")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "train_losses.png"))
            plt.close()

    @torch.no_grad()
    def render(self):
        logging.info(f"\nStart render, render mode is {self.render_mode} ... ...")
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        if self.tacview is None:
            self.envs.render(mode=self.render_mode, filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
        else:
            self.envs.render(mode=self.render_mode, filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi',tacview=self.tacview)
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)

            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            if self.use_selfplay:
                render_rewards = render_rewards[:, :self.num_agents // 2, ...]
            render_episode_rewards += render_rewards
            # print(f"render_infos:{render_infos}---render_episode_rewards{render_episode_rewards}---render_rewards:{render_rewards}")
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    def save(self, episode):
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
