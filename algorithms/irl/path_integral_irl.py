from typing import Optional, Sequence

import numpy as np

from .maxent_irl import _normalize_weights, _trajectory_feature_sums


def _path_integral_probabilities(returns: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    scaled = returns / temperature
    scaled = scaled - np.max(scaled)
    weights = np.exp(scaled)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return np.full_like(weights, 1.0 / weights.size)
    return weights / weight_sum


class PathIntegralIRL:
    """Path Integral IRL (PI-IRL) for heading reward weights."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 50, temperature: float = 1.0,
                 l2_reg: float = 0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.temperature = temperature
        self.l2_reg = l2_reg

    def fit(self, expert_trajectories: Sequence[np.ndarray], sampled_trajectories: Sequence[np.ndarray],
            init_weights: Optional[Sequence[float]] = None) -> dict:
        if not expert_trajectories:
            raise ValueError("expert_trajectories must not be empty.")
        if not sampled_trajectories:
            raise ValueError("sampled_trajectories must not be empty.")

        expert_sums = _trajectory_feature_sums(expert_trajectories)
        sampled_sums = _trajectory_feature_sums(sampled_trajectories)

        expert_expectation = expert_sums.mean(axis=0)
        weights = _normalize_weights(init_weights or [0.25, 0.25, 0.25, 0.25])
        history = []

        for epoch in range(self.epochs):
            returns = sampled_sums @ weights
            probs = _path_integral_probabilities(returns, self.temperature)
            expected = probs @ sampled_sums
            grad = expert_expectation - expected - self.l2_reg * weights
            weights = _normalize_weights(weights + self.learning_rate * grad)
            history.append({
                "epoch": epoch + 1,
                "weights": weights.copy(),
                "grad": grad.copy(),
                "expected": expected.copy(),
                "entropy": -float(np.sum(np.clip(probs, 1e-8, 1.0) * np.log(np.clip(probs, 1e-8, 1.0)))),
                "mean_return": float(np.mean(returns)),
            })
        return {"weights": weights, "history": history}
