import math
import os
import yaml
import pymap3d
import numpy as np


def parse_config(filename):
    """Parse JSBSim config file.

    Args:
        config (str): config file name

    Returns:
        (EnvConfig): a custom class which parsing dict into object.
    """
    filepath = os.path.join(get_root_dir(), 'configs', f'{filename}.yaml')
    assert os.path.exists(filepath), \
        f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
    with open(filepath, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return type('EnvConfig', (object,), config_data)


def get_root_dir():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')


def LLA2NEU(lon, lat, alt, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from Geodetic Coordinate System to NEU Coordinate System.

    Args:
        lon, lat, alt (float): target geodetic lontitude(°), latitude(°), altitude(m)
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (North, East, Up), unit: m
    """
    n, e, d = pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
    return np.array([n, e, -d])


def NEU2LLA(n, e, u, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from NEU Coordinate System to Geodetic Coordinate System.

    Args:
        n, e, u (float): target relative position w.r.t. North, East, Down
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (lon, lat, alt), unit: °, °, m
    """
    lat, lon, h = pymap3d.ned2geodetic(n, e, -u, lat0, lon0, alt0)
    return np.array([lon, lat, h])


def get_AO_TA_R(ego_feature, enm_feature, return_side=False):
    """Get AO & TA angles and relative distance between two agent.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd)

    Returns:
        (tuple): ego_AO, ego_TA, R
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y, delta_z])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy + delta_z * ego_vz
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy + delta_z * enm_vz
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        return ego_AO, ego_TA, R, side_flag


def get_attack_angle(ego_feature, enm_feature):
    """Get attack angle between two agent.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd)

    Returns:
        (tuple): attack_angle
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z

    target = np.array([delta_x, delta_y, delta_z])
    distance = np.linalg.norm(target)
    heading = np.array([ego_vx, ego_vy, ego_vz])

    attack_angle = np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1))

    return attack_angle


def get_HR(ego_feature, enm_feature):
    """Get HR between two agent.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd)

    Returns:
        (tuple): HO, HT
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature

    # 20000ft暂时取固定值6096m，6.096km
    ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])
    enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])

    # 计算能量高度(重力加速度取9.81)
    HO = ego_z + 0.5 * ego_v * ego_v / 9.81
    HT = enm_z + 0.5 * enm_v * enm_v / 9.81
    H = math.fabs(ego_z - enm_z)

    return HO, HT, H


def cup_reward(x: float, a: float, b: float, k: float) -> float:
    """
    杯型奖励函数
    :param x: 输入状态值（如速度、高度）
    :param a: 下边界阈值（如最小安全值）
    :param b: 上边界阈值（如最大安全值）
    :param k: 陡峭系数（控制过渡区平滑度）
    :return: 奖励值（负值表示惩罚）
    """
    term1 = 1 / (1 + np.exp((x - b) * k))
    term2 = 1 / (1 + np.exp((x - a) * k))
    return term1 - term2 - 1


def cup_reward_2(x: float, a: float, b: float, k: float) -> float:
    """
    杯型奖励函数
    :param x: 输入状态值（如速度、高度）
    :param a: 下边界阈值（如最小安全值）
    :param b: 上边界阈值（如最大安全值）
    :param k: 陡峭系数（控制过渡区平滑度）
    :return: 奖励值（负值表示惩罚）
    """
    term1 = 1 / (1 + np.exp((x - a) / k))
    term2 = 1 / (1 + np.exp(-(x - b) / k))
    return term1 - term2 + 1


def get_AOT(ego_feature, enm_feature):
    """
    Calculate AOT (Angle Off Tail) between two agents.

    Args:
        ego_feature (tuple): (north, east, down, vn, ve, vd)
        enm_feature (tuple): (north, east, down, vn, ve, vd)

    Returns:
        AOT (float): Angle off tail in radians
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature

    # Velocity magnitudes
    ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])
    enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])

    # Relative position
    _x, _y, _z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([_x, _y, _z])

    # Ego to enemy vector projection
    proj_dist = _x * ego_vx + _y * ego_vy + _z * ego_vz
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))

    # Calculate AOT
    aot_proj = _x * enm_vx + _y * enm_vy + _z * enm_vz
    AOT = np.arccos(np.clip(aot_proj / (R * enm_v + 1e-8), -1, 1))

    return AOT


def get2d_AO_TA_R(ego_feature, enm_feature, return_side=False):
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        return ego_AO, ego_TA, R, side_flag


def in_range_deg(angle):
    """ Given an angle in degrees, normalises in (-180, 180] """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def in_range_rad(angle):
    """ Given an angle in rads, normalises in (-pi, pi] """
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


def radians_to_degrees(radians):
    return radians * (180 / math.pi)
