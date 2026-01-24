import torch
import torch.nn as nn
from typing import Union, List, Optional
from .ppo_policy import PPOPolicy
from ..utils.buffer import ReplayBuffer
from ..utils.utils import check, get_gard_norm


class PPOTrainer:
    def __init__(self, args, policy: Optional[PPOPolicy] = None, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        # PPO 配置
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # RNN 配置
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length
        self.bc_regularization = getattr(args, "bc_regularization", False)
        self.bc_coef = getattr(args, "bc_coef", 0.0)
        self.bc_policy = None
        if self.bc_regularization and policy is not None:
            bc_model_dir = getattr(args, "bc_model_dir", None)
            if bc_model_dir:
                self.bc_policy = policy.copy()
                actor_state = torch.load(f"{bc_model_dir}/actor_latest.pt", map_location=device)
                self.bc_policy.actor.load_state_dict(actor_state)
                self.bc_policy.actor.eval()

    def ppo_update(self, policy: PPOPolicy, sample):

        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        # 统一形状以便一次前向计算所有步
        values, action_log_probs, dist_entropy = policy.evaluate_actions(obs_batch,
                                                                         rnn_states_actor_batch,
                                                                         rnn_states_critic_batch,
                                                                         actions_batch,
                                                                         masks_batch)

        # 计算损失
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
        policy_loss = -policy_loss.mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2)
        value_loss = value_loss.mean()

        policy_entropy_loss = -dist_entropy.mean()

        loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef
        if self.bc_policy is not None and self.bc_coef > 0:
            with torch.no_grad():
                bc_action_log_probs, _ = self.bc_policy.actor.evaluate_actions(
                    obs_batch, rnn_states_actor_batch, actions_batch, masks_batch
                )
            bc_kl = (action_log_probs - bc_action_log_probs).mean()
            loss = loss + self.bc_coef * bc_kl

        # 反向传播与优化
        policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.optimizer.step()

        return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm

    def train(self, policy: PPOPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = ReplayBuffer.recurrent_generator(buffer, self.num_mini_batch, self.data_chunk_length)
            else:
                raise NotImplementedError

            for sample in data_generator:

                policy_loss, value_loss, policy_entropy_loss, ratio, \
                    actor_grad_norm, critic_grad_norm = self.ppo_update(policy, sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
