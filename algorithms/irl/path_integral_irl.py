from typing import Optional, Sequence

import numpy as np

from .maxent_irl import _normalize_weights, _trajectory_feature_sums


def _normalize_returns(returns: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    returns = returns.astype(np.float64, copy=False)
    mean = float(np.mean(returns))
    std = float(np.std(returns))
    if std <= eps:
        return returns - mean
    return (returns - mean) / (std + eps)


def _path_integral_probabilities(returns: np.ndarray, temperature: float, uniform_mix: float = 0.0) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    scaled = returns / temperature
    scaled = scaled - np.max(scaled)
    weights = np.exp(scaled, dtype=np.float64)
    weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        probs = np.full_like(weights, 1.0 / weights.size)
    else:
        probs = weights / weight_sum
    uniform_mix = float(np.clip(uniform_mix, 0.0, 1.0))
    if uniform_mix > 0:
        probs = (1.0 - uniform_mix) * probs + uniform_mix / probs.size
    return probs


class PathIntegralIRL:
    """Path Integral IRL (PI-IRL) for heading reward weights."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 50, temperature: float = 1.0,
                 l2_reg: float = 0.0, normalize_returns: bool = True, uniform_mix: float = 0.05,
                 resample_count: Optional[int] = None, seed: Optional[int] = None,
                 replay_buffer_size: Optional[int] = None, replay_batch_size: Optional[int] = None,
                 temperature_decay: float = 0.98, min_temperature: float = 0.1,
                 optimizer: str = "adam", lr_decay: float = 0.99, adam_beta1: float = 0.9,
                 adam_beta2: float = 0.999, adam_eps: float = 1e-8,
                 ensemble_runs: int = 1, ensemble_seed_offset: int = 1000):
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
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.ensemble_runs = max(int(ensemble_runs), 1)
        self.ensemble_seed_offset = ensemble_seed_offset
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)

    def fit(self, expert_trajectories: Sequence[np.ndarray], sampled_trajectories: Sequence[np.ndarray],
            init_weights: Optional[Sequence[float]] = None) -> dict:
        if not expert_trajectories:
            raise ValueError("expert_trajectories must not be empty.")
        if not sampled_trajectories:
            raise ValueError("sampled_trajectories must not be empty.")

        expert_sums = _trajectory_feature_sums(expert_trajectories)
        sampled_sums = _trajectory_feature_sums(sampled_trajectories)

        expert_expectation = expert_sums.mean(axis=0)
        history = []
        ensemble_weights = []

        for run_idx in range(self.ensemble_runs):
            run_seed = self._base_seed
            if run_seed is not None:
                run_seed = run_seed + run_idx * self.ensemble_seed_offset
            rng = np.random.default_rng(run_seed)

            weights = _normalize_weights(init_weights or [0.25, 0.25, 0.25, 0.25])
            m = np.zeros_like(weights)
            v = np.zeros_like(weights)

            buffer_sums = sampled_sums
            if self.replay_buffer_size:
                buffer_size = max(int(self.replay_buffer_size), 1)
                if sampled_sums.shape[0] > buffer_size:
                    indices = rng.choice(sampled_sums.shape[0], size=buffer_size, replace=False)
                    buffer_sums = sampled_sums[indices]

            batch_size = self.replay_batch_size
            if batch_size is not None:
                batch_size = max(int(batch_size), 1)

            for epoch in range(self.epochs):
                batch_sums = buffer_sums
                if batch_size and buffer_sums.shape[0] > batch_size:
                    indices = rng.choice(buffer_sums.shape[0], size=batch_size, replace=False)
                    batch_sums = buffer_sums[indices]

                returns = batch_sums @ weights
                if self.normalize_returns:
                    returns = _normalize_returns(returns)

                temperature = max(self.min_temperature, self.temperature * (self.temperature_decay ** epoch))
                probs = _path_integral_probabilities(returns, temperature, uniform_mix=self.uniform_mix)
                if self.resample_count:
                    count = max(int(self.resample_count), 1)
                    indices = rng.choice(batch_sums.shape[0], size=count, replace=True, p=probs)
                    expected = np.mean(batch_sums[indices], axis=0)
                else:
                    expected = probs @ batch_sums
                grad = expert_expectation - expected - self.l2_reg * weights

                lr = self.learning_rate * (self.lr_decay ** epoch)
                if self.optimizer.lower() == "adam":
                    m = self.adam_beta1 * m + (1.0 - self.adam_beta1) * grad
                    v = self.adam_beta2 * v + (1.0 - self.adam_beta2) * (grad ** 2)
                    m_hat = m / (1.0 - self.adam_beta1 ** (epoch + 1))
                    v_hat = v / (1.0 - self.adam_beta2 ** (epoch + 1))
                    step = lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
                else:
                    step = lr * grad
                weights = _normalize_weights(weights + step)

                history.append({
                    "run": run_idx + 1,
                    "epoch": epoch + 1,
                    "weights": weights.copy(),
                    "grad": grad.copy(),
                    "expected": expected.copy(),
                    "entropy": -float(np.sum(np.clip(probs, 1e-8, 1.0) * np.log(np.clip(probs, 1e-8, 1.0)))),
                    "mean_return": float(np.mean(returns)),
                    "temperature": float(temperature),
                    "learning_rate": float(lr),
                })

            ensemble_weights.append(weights)

        final_weights = _normalize_weights(np.mean(np.stack(ensemble_weights, axis=0), axis=0))
        return {"weights": final_weights, "history": history, "ensemble_weights": ensemble_weights}
