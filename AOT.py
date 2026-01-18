import numpy as np

from envs.JSBSim.utils.utils import get_AOT, get_AO_TA_R, get_attack_angle


def print_test(ego_feature, enm_feature):
    # 调用 get_AO_TA_R
    ego_AO, ego_TA, R = get_AO_TA_R(ego_feature, enm_feature)
    attack_angle = get_attack_angle(ego_feature, enm_feature)
    print("get_AO_TA_R Output:")
    print(f"ego_AO: {np.rad2deg(ego_AO):.2f}")
    print(f"ego_TA: {np.rad2deg(ego_TA):.2f}")
    print(f"attack_angle: {attack_angle:.2f}")
    print(f"Relative Distance (R): {R:.2f}")
    print(f"reward:{1 / (50 * ego_AO / np.pi + 2) + 1 / 2 + min((np.arctanh(1. - max(2 * ego_TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5}")
    print(1-(ego_AO+ego_TA)/np.pi)


if __name__ == "__main__":
    # 测试数据
    ego_feature = (2, 1, 5, 10, 14, 11)  # (north, east, down, vn, ve, vd)
    enm_feature = (20, 112, 32, 11, 5, 3)  # (north, east, down, vn, ve, vd)
    # 绘制曲线
    print_test(ego_feature, enm_feature)
    print_test(enm_feature, ego_feature)

    obs = [1,2,3,4,5]
    a_obs = obs
    obs[0] = 2
    print(obs)
