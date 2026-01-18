import json
import os
import shutil
import sys
import wandb
import torch
import logging
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.utils.buffer import ReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):
        print(config)
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.current_episode = 0
        self.render_mode = config['render_mode']
        self.timestamp = 0

        # Tacview render obj
        self.tacview = None
        if self.render_mode == "real_time":
            from runner.tacview import Tacview
            self.tacview = Tacview()

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = int(self.all_args.num_env_steps)
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.buffer_size = self.all_args.buffer_size
        self.use_wandb = self.all_args.use_wandb

        # interval
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.eval_episodes = self.all_args.eval_episodes

        # dir
        self.model_dir = self.all_args.model_dir
        self.run_dir = config["run_dir"]
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.save_dir = str(self.run_dir)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.load()

    def _should_save_acmi(self):
        """ 判断是否应该保存 ACMI 文件 """
        return (
                self.current_episode % self.eval_interval == 0
                and self.use_eval
        )

    def load(self):
        # algorithm
        if self.algorithm_name == "ppo":
            from ..algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from ..algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args,
                             self.envs.observation_space,
                             self.envs.action_space,
                             device=self.device)
        self.trainer = Trainer(self.all_args, self.policy, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args,
                                   self.envs.observation_space,
                                   self.envs.action_space)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def rollout(self):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        self.policy.prep_rollout()
        next_values = self.policy.get_values(np.concatenate(self.buffer.obs[-1]),
                                             np.concatenate(self.buffer.rnn_states_critic[-1]),
                                             np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def train(self):
        self.policy.prep_training()
        train_infos = self.trainer.train(self.policy, self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        policy_actor = self.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_latest.pt")
        policy_critic = self.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_latest.pt")

    def restore(self):
        # # 复制全部信息到当前文件夹下
        # try:
        #     # 复制所有内容（包括子目录）
        #     shutil.copytree(
        #         src=self.model_dir,
        #         dst=self.save_dir,
        #         dirs_exist_ok=True  # 允许目标文件夹已存在（Python 3.8+）
        #     )
        #     logging.info(f"Copied entire directory from {self.model_dir} to {self.save_dir}")
        # except Exception as e:
        #     logging.error(f"Failed to copy directory: {e}")
        #     raise
        #
        # # 读取elo信息文件
        # elo_history_path = os.path.join(self.model_dir, 'elo_history.json')
        # with open(elo_history_path, 'r') as f:
        #     elo_history = json.load(f)
        #     self.latest_elo = elo_history["latest_elo"]
        #     self.policy_pool = {str(k): float(v) for k, v in elo_history["policy_pool"].items()}
        #     self.current_episode = elo_history["current_episode"]
        #     logging.info(f"Loaded ELO history from {elo_history_path}")

        # # 读取当前策略网络
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_latest.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_latest.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_info(self, infos, total_num_steps):
        if self.use_wandb:
            for k, v in infos.items():
                wandb.log({k: v}, step=total_num_steps)
        else:
            pass

    def render_with_tacview(self, data):
        """
        Send data to Tacview for real-time rendering.
        :param data: The data to be rendered, which can be a string or a specific structure.
        """
        if self.tacview:
            try:
                self.tacview.send_data_to_client(data)
            except Exception as e:
                logging.error(f"Tacview rendering error: {e}")
