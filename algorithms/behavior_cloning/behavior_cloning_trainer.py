import os
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from algorithms.behavior_cloning.behavior_cloning_actor import BehaviorCloningActor


def behavior_cloning_loss(pred_actions, true_actions, model, lambda_reg=1e-5):
    mse_loss = F.mse_loss(pred_actions, true_actions)
    l2_reg = sum(p.norm(2).pow(2) for p in model.parameters())  # L2 正则化
    return mse_loss + lambda_reg * l2_reg


# 训练过程
def train_behavior_cloning(model, dataset, optimizer, batch_size=64, save_interval=10, save_dir='./models'):
    # 创建保存模型的目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    for epoch in range(100):  # 假设最多训练 100 个 epoch
        for batch_start in range(0, len(dataset), batch_size):
            # 获取一个批次
            batch = dataset[batch_start: batch_start + batch_size]
            obs_batch = torch.stack([item[0] for item in batch])  # 观测
            actions_batch = torch.stack([item[1] for item in batch])  # 专家动作

            # 模型预测
            pred_actions = model(obs_batch)

            # 计算损失
            loss = behavior_cloning_loss(pred_actions, actions_batch)

            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Batch Loss: {loss.item()}")

        # 每隔一定步数保存一次模型
        if epoch % save_interval == 0:
            model_filename = os.path.join(save_dir, f'behavior_cloning_epoch_{epoch}.pt')
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved at {model_filename}")


def evaluate_behavior_cloning(model, val_dataset):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataset:
            obs_batch = torch.stack([item[0] for item in batch])
            actions_batch = torch.stack([item[1] for item in batch])

            pred_actions = model(obs_batch)
            loss = behavior_cloning_loss(pred_actions, actions_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_dataset)
    print(f"Validation Loss: {avg_loss}")
    return avg_loss


if __name__ == "__main__":
    args_dict = {
        'hidden_size': "128 128",
        'activation_id': 1,
        'use_feature_normalization': False,
        'use_recurrent_policy': True,
        'recurrent_hidden_size': 128,
        'recurrent_hidden_layers': 1,
        'act_hidden_size': "128 128",
        'gain': 0.01
    }

    args = SimpleNamespace(**args_dict)
    obs_space = gym.spaces.Box(low=-10, high=10., shape=(12,))
    act_space = gym.spaces.MultiDiscrete([41, 41, 41, 30])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BehaviorCloningActor(args, obs_space, act_space, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    save_interval = 10
    batch_size = 64
    dataset = [(torch.randn(12), torch.randn(4)) for _ in range(1000)]

    model.train()

    rnn_states = np.zeros((1, 1, 128))
    for epoch in range(100):
        for batch_start in range(0, len(dataset), batch_size):
            batch = dataset[batch_start: batch_start + batch_size]
            obs_batch = torch.stack([item[0] for item in batch])
            actions_batch = torch.stack([item[1] for item in batch])

            pred_actions, rnn_states = model(obs_batch, rnn_states, deterministic=True)

            loss = behavior_cloning_loss(pred_actions, actions_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Batch Loss: {loss.item()}")

        # 调整学习率
        scheduler.step()

        if epoch % save_interval == 0:
            model_filename = os.path.join(save_dir, f'behavior_cloning_epoch_{epoch}.pt')
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved at {model_filename}")
