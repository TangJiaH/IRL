from typing import Optional, Sequence

import numpy as np

from .path_integral_irl import _normalize_returns, _path_integral_probabilities


class PathIntegralIRLWithDeepLearning:
    """Path Integral IRL with a reward network trained via backpropagation."""

    def __init__(self, reward_network, learning_rate: float = 0.1, epochs: int = 50, temperature: float = 1.0,
                 l2_reg: float = 0.0, normalize_returns: bool = True, uniform_mix: float = 0.05,
                 resample_count: Optional[int] = None, seed: Optional[int] = None,
                 replay_buffer_size: Optional[int] = None, replay_batch_size: Optional[int] = None,
                 temperature_decay: float = 0.98, min_temperature: float = 0.1,
                 grad_clip: Optional[float] = None):
        self.reward_network = reward_network
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.temperature = temperature
        self.l2_reg = l2_reg
        self.normalize_returns = normalize_returns
        self.uniform_mix = uniform_mix
        self.resample_count = resample_count
        self.replay_buffer_size = replay_buffer_size
        self.replay_batch_size = replay_batch_size
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.grad_clip = grad_clip
        self._rng = np.random.default_rng(seed)

    def fit(self, expert_trajectories: Sequence[np.ndarray], sampled_trajectories: Sequence[np.ndarray]) -> dict:
        import torch

        if not expert_trajectories:
            raise ValueError("expert_trajectories must not be empty.")
        if not sampled_trajectories:
            raise ValueError("sampled_trajectories must not be empty.")

        expert_sums = np.array([traj.sum(axis=0) for traj in expert_trajectories], dtype=np.float32)
        sampled_sums = np.array([traj.sum(axis=0) for traj in sampled_trajectories], dtype=np.float32)
        expert_expectation = expert_sums.mean(axis=0)

        optimizer = torch.optim.Adam(self.reward_network.parameters(), lr=self.learning_rate)
        history = []

        buffer_sums = sampled_sums
        if self.replay_buffer_size:
            buffer_size = max(int(self.replay_buffer_size), 1)
            if sampled_sums.shape[0] > buffer_size:
                indices = self._rng.choice(sampled_sums.shape[0], size=buffer_size, replace=False)
                buffer_sums = sampled_sums[indices]

        batch_size = self.replay_batch_size
        if batch_size is not None:
            batch_size = max(int(batch_size), 1)

        for epoch in range(self.epochs):
            batch_sums = buffer_sums
            if batch_size and buffer_sums.shape[0] > batch_size:
                indices = self._rng.choice(buffer_sums.shape[0], size=batch_size, replace=False)
                batch_sums = buffer_sums[indices]

            batch_tensor = torch.tensor(batch_sums, dtype=torch.float32)
            returns = self.reward_network(batch_tensor).squeeze(-1)
            returns_np = returns.detach().cpu().numpy()
            if self.normalize_returns:
                returns_np = _normalize_returns(returns_np)

            temperature = max(self.min_temperature, self.temperature * (self.temperature_decay ** epoch))
            probs = _path_integral_probabilities(returns_np, temperature, uniform_mix=self.uniform_mix)
            probs_tensor = torch.tensor(probs, dtype=torch.float32)

            if self.resample_count:
                count = max(int(self.resample_count), 1)
                indices = self._rng.choice(batch_sums.shape[0], size=count, replace=True, p=probs)
                expected = batch_sums[indices].mean(axis=0)
            else:
                expected = probs @ batch_sums

            expected_tensor = torch.tensor(expected, dtype=torch.float32)
            expert_tensor = torch.tensor(expert_expectation, dtype=torch.float32)
            mse = torch.mean((expected_tensor - expert_tensor) ** 2)
            l2_penalty = torch.tensor(0.0)
            if self.l2_reg > 0:
                l2_penalty = sum(torch.sum(param ** 2) for param in self.reward_network.parameters()) * self.l2_reg
            loss = mse + l2_penalty

            optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.reward_network.parameters(), max_norm=self.grad_clip)
            optimizer.step()

            history.append({
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "mse": float(mse.item()),
                "temperature": float(temperature),
            })

        return {"history": history}
