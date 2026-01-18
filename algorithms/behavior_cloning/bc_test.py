from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from torch.optim import Adam

from algorithms.behavior_cloning.ppo_actor_copy import PPOActor


def behavior_cloning_loss(action_log_probs, true_actions):
    loss = 0.0
    start_index = 0
    action_dims = [41, 41, 41, 30]  # 每个动作维度的大小
    for action_dim in action_dims:
        end_index = start_index + action_dim
        predicted_prob = action_log_probs[:, start_index:end_index]
        target_action = true_actions[:, start_index]

        print("predicted_prob size:", predicted_prob.size())  # 检查 predicted_prob 的尺寸
        print("target_action size:", target_action.size())  # 检查 target_action 的尺寸

        print("predicted_prob dtype:", predicted_prob.dtype)  # 检查 predicted_prob 的数据类型
        print("target_action dtype:", target_action.dtype)  # 检查 target_action 的数据类型

        print("predicted_prob max value:", predicted_prob.max().item())  # 检查 predicted_prob 的最大值
        print("predicted_prob min value:", predicted_prob.min().item())  # 检查 predicted_prob 的最小值

        print("target_action max value:", target_action.max().item())  # 检查 target_action 的最大值
        print("target_action min value:", target_action.min().item())  # 检查 target_action 的最小值

        loss += F.nll_loss(predicted_prob, target_action)
        start_index = end_index
    return loss


# 初始化网络
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
network = PPOActor(args, obs_space, act_space, device).to(device)

# 使用Adam优化器
optimizer = Adam(network.parameters(), lr=1e-4)


def train_behavior_cloning(expert_data, network, epochs=100, batch_size=64):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(expert_data), batch_size):
            batch = expert_data[i:i + batch_size]

            # 提取状态和动作
            states = torch.tensor([entry['state'] for entry in batch], dtype=torch.float32).to(device)
            actions = torch.tensor([entry['action'] for entry in batch], dtype=torch.int64).to(device)

            actions = actions.squeeze(1)  # 确保 actions 的形状是 [batch_size]

            # 初始化RNN状态和掩码
            rnn_states = torch.zeros(
                (states.size(0), network.recurrent_hidden_layers, network.recurrent_hidden_size)).to(device)
            masks = torch.ones(states.size(0), 1).to(device)

            # 前向传播得到预测的动作
            predicted_actions, action_log_probs, _ = network(states, rnn_states, masks, deterministic=True)

            # 计算损失
            loss = behavior_cloning_loss(action_log_probs, actions)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 打印当前epoch的平均损失
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(expert_data)}")


if __name__ == "__main__":
    # 定义状态空间和动作空间的大小
    state_dim = 12  # 状态空间的维度
    action_dims = [41, 41, 41, 30]  # 每个动作维度的大小

    # 生成一个专家数据集
    expert_data = []

    # 生成32个示例，每个示例包含状态和动作序列（每个状态是一个单独的时间步）
    for _ in range(32):
        state = np.random.uniform(-10, 10, (state_dim,))
        action = np.concatenate([np.random.randint(0, action_dims[i], 1) for i in range(4)], axis=0)

        # 确保动作在正确的范围内
        assert all(action[i] < action_dims[i] for i in range(len(action)))

        expert_data.append({'state': state, 'action': action})

    # 转换为numpy数组方便处理
    states = np.array([entry['state'] for entry in expert_data])  # shape: (32, 12)
    actions = np.array([entry['action'] for entry in expert_data])  # shape: (32, 4)

    # 将生成的数据转为PyTorch张量
    states_tensor = torch.tensor(states, dtype=torch.float32)  # shape: (32, 12)
    actions_tensor = torch.tensor(actions, dtype=torch.int64)  # shape: (32, 4)

    print("States Tensor Shape:", states_tensor.shape)
    print("Actions Tensor Shape:", actions_tensor.shape)

    # 训练网络
    train_behavior_cloning(expert_data, network)
