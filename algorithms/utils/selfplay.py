import numpy as np
from typing import Dict, List
from abc import ABC, abstractstaticmethod


def get_algorithm(algo_name):
    if algo_name == 'sp':
        return SP
    elif algo_name == 'fsp':
        return FSP
    elif algo_name == 'pfsp':
        return PFSP
    else:
        raise NotImplementedError("Unknown algorithm {}".format(algo_name))


class SelfplayAlgorithm(ABC):

    @abstractstaticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        pass

    @abstractstaticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass


class SP(SelfplayAlgorithm):

    @staticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        return list(agents_elo.keys())[-1]

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass


class FSP(SelfplayAlgorithm):

    @staticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        return np.random.choice(list(agents_elo.keys()))

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass


# class PFSP(SelfplayAlgorithm):
#
#     @staticmethod
#     def choose(agents_elo: Dict[str, float], lam=1, s=100, **kwargs) -> str:
#         history_elo = np.array(list(agents_elo.values()))
#         sample_probs = 1. / (1. + 10. ** (-(history_elo - np.median(history_elo)) / 400.)) * s
#         """ meta-solver """
#         k = float(len(sample_probs) + 1)
#         meta_solver_probs = np.exp(lam / k * sample_probs) / np.sum(np.exp(lam / k * sample_probs))
#         opponent_idx = np.random.choice(a=list(agents_elo.keys()), size=1, p=meta_solver_probs).item()
#         return opponent_idx
#
#     @staticmethod
#     def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]]) -> None:
#         pass

class PFSP:

    @staticmethod
    def choose(agents_elo: Dict[str, float], lam=1, s=100, **kwargs) -> str:
        """
        原始方法：选择一个对手。
        """
        agent_keys = list(agents_elo.keys())
        history_elo = np.array(list(agents_elo.values()))

        if not agent_keys:
            raise ValueError("策略池不能为空。")

        sample_probs = 1. / (1. + 10. ** (-(history_elo - np.median(history_elo)) / 400.)) * s
        k = float(len(sample_probs) + 1)

        # 计算概率并进行归一化
        exp_probs = np.exp(lam / k * sample_probs)
        meta_solver_probs = exp_probs / np.sum(exp_probs)

        # 确保概率和为 1 (处理浮点数精度问题)
        meta_solver_probs = meta_solver_probs / np.sum(meta_solver_probs)

        opponent_key = np.random.choice(a=agent_keys, size=1, p=meta_solver_probs).item()
        return opponent_key

    @staticmethod
    def choose_multiple(agents_elo: Dict[str, float], n: int, lam=1, s=100, **kwargs) -> List[str]:
        """
        新方法：选择 n 个不重复的对手。

        参数:
            agents_elo: 包含策略 ID 和其 Elo 分数的字典。
            n: 需要选择的不重复对手的数量。
            lam: PFSP 参数。
            s: PFSP 参数。

        返回:
            包含 n 个不重复对手策略 ID 的列表。
        """
        agent_keys = list(agents_elo.keys())
        history_elo = np.array(list(agents_elo.values()))
        num_available = len(agent_keys)

        # 如果可用策略为空，返回空列表
        if num_available == 0:
            return []

        # 如果需要的数量大于或等于可用数量，则返回所有策略
        if n >= num_available:
            print(f"警告：需要的数量 ({n}) >= 可用数量 ({num_available})。返回所有可用策略。")
            return agent_keys

        # 计算概率 (与原始方法相同)
        sample_probs = 1. / (1. + 10. ** (-(history_elo - np.median(history_elo)) / 400.)) * s
        k = float(num_available + 1)

        # 计算概率并进行归一化
        exp_probs = np.exp(lam / k * sample_probs)
        meta_solver_probs = exp_probs / np.sum(exp_probs)

        # 确保概率和为 1 (处理浮点数精度问题)
        meta_solver_probs = meta_solver_probs / np.sum(meta_solver_probs)

        # 使用 np.random.choice 选择 n 个不重复的对手
        # a: 从中选择的元素列表 (策略 ID)
        # size: 要选择的数量 n
        # replace=False: 确保选择是不重复的
        # p: 每个元素被选中的概率
        chosen_opponents = np.random.choice(
            a=agent_keys,
            size=n,
            replace=False,  # <<< 关键：设置为 False 实现不重复选择
            p=meta_solver_probs
        )

        return chosen_opponents.tolist()

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]]) -> None:
        pass

