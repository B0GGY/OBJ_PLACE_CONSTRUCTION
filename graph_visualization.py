import pickle
import math

from arguments import get_args
import habitat_sim
import habitat
import cv2
import os
from matplotlib import pyplot as plt
import random
import numpy as np
from arguments import get_args
import colorsys

args = get_args()

def load_graph(filename):
    with open(filename, 'rb') as f:
        g = pickle.load(f)
    return g

def get_node_size(node_list, points_dict, max_size = 200):
    total_num = 0
    for k in points_dict.keys():
        total_num += len(points_dict[k])

    return math.ceil((len(node_list)/total_num)*max_size)

def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for place in key_points.keys():
            # node_size = get_node_size(key_points[place], key_points)
            # print(node_size)
            for p in key_points[place]:
                plt.plot(p[0], p[1], marker="o", markersize=10, alpha=0.8, color=np.array(generate_distinct_color(place)) / 255)
            # avg_place = np.mean(key_points[place], axis=0)
            # plt.plot(avg_place[0], avg_place[1], marker="o", markersize=node_size, alpha=0.8,
            #          color=np.array(generate_distinct_color(place)) / 255)
            # print(avg_place)
    plt.show()


def graph_visualization(graph, topdown_map):
    vis_dict = {}
    total_data_points = 0
    for p in graph.vs['agent_state']:
        total_data_points += len(p)
    # print(total_data_points)
    max_size = 200
    for i in range(graph.vcount()):
        vis_dict[i]={'size':math.ceil((len(graph.vs[i]['agent_state'])/total_data_points)*max_size),
                     'avg_position':np.mean(graph.vs[i]['agent_state'], axis=0)/args.map_resolution}
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map

    for k in vis_dict.keys():

        plt.plot(vis_dict[k]['avg_position'][0], vis_dict[k]['avg_position'][1], marker="o", markersize=vis_dict[k]['size'], alpha=0.8,
                 color=np.array(generate_distinct_color(k)) / 255)
        neighbors = graph.neighbors(k, mode='all')
        for n in neighbors:
            plt.plot((vis_dict[k]['avg_position'][0], vis_dict[n]['avg_position'][0]), (vis_dict[k]['avg_position'][1], vis_dict[n]['avg_position'][1]), label="Line segment", color="blue")
    plt.show()

def generate_distinct_color(index, total_colors=30):
    """
    根据输入索引生成不同的颜色，并确保相同索引返回相同颜色。

    参数:
    - index: int，输入的数字索引。
    - total_colors: int，可选，预计总共有多少种不同颜色（用于区分度计算）。

    返回:
    - RGB 格式的颜色 (r, g, b)，范围 [0, 255]
    """
    # 使用哈希函数保持一致性，确保相同数字生成相同颜色
    unique_index = hash(index) % total_colors

    # 将索引映射到 HSV 色彩空间，确保均匀分布
    hue = (unique_index / total_colors)  # 色调分布在 [0, 1]
    saturation = 0.7  # 固定饱和度，值在 [0, 1]
    value = 0.9  # 固定亮度，值在 [0, 1]

    # 将 HSV 转换为 RGB 格式
    rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
    rgb = tuple(int(c * 255) for c in rgb_float)

    return rgb
#
# args = get_args()
# rgb_sensor = True
# depth_sensor = False
# semantic_sensor = False
# test_scene = '1LXtFkjw3qL'
#
# test_scene_path = os.path.join(args.data_root, test_scene, '{}.glb'.format(test_scene))
#
#
# def make_cfg(settings):
#     sim_cfg = habitat_sim.SimulatorConfiguration()
#     sim_cfg.gpu_device_id = 0
#     sim_cfg.scene_id = settings["scene"]
#     sim_cfg.enable_physics = settings["enable_physics"]
#
#     # Note: all sensors must have the same resolution
#     sensors = {
#         "color_sensor": {
#             "sensor_type": habitat_sim.SensorType.COLOR,
#             "resolution": [settings["height"], settings["width"]],
#             "position": [0.0, settings["sensor_height"], 0.0],
#         },
#         "depth_sensor": {
#             "sensor_type": habitat_sim.SensorType.DEPTH,
#             "resolution": [settings["height"], settings["width"]],
#             "position": [0.0, settings["sensor_height"], 0.0],
#         },
#         "semantic_sensor": {
#             "sensor_type": habitat_sim.SensorType.SEMANTIC,
#             "resolution": [settings["height"], settings["width"]],
#             "position": [0.0, settings["sensor_height"], 0.0],
#         },
#     }
#
#     sensor_specs = []
#     for sensor_uuid, sensor_params in sensors.items():
#         if settings[sensor_uuid]:
#             sensor_spec = habitat_sim.CameraSensorSpec()
#             sensor_spec.uuid = sensor_uuid
#             sensor_spec.sensor_type = sensor_params["sensor_type"]
#             sensor_spec.resolution = sensor_params["resolution"]
#             sensor_spec.position = sensor_params["position"]
#
#             sensor_specs.append(sensor_spec)
#
#     # Here you can specify the amount of displacement in a forward action and the turn angle
#     agent_cfg = habitat_sim.agent.AgentConfiguration()
#     agent_cfg.sensor_specifications = sensor_specs
#     agent_cfg.action_space = {
#         "move_forward": habitat_sim.agent.ActionSpec(
#             "move_forward", habitat_sim.agent.ActuationSpec(amount=args.march_amount)
#         ),
#         "turn_left": habitat_sim.agent.ActionSpec(
#             "turn_left", habitat_sim.agent.ActuationSpec(amount=args.turn_amount)
#         ),
#         "turn_right": habitat_sim.agent.ActionSpec(
#             "turn_right", habitat_sim.agent.ActuationSpec(amount=args.turn_amount)
#         ),
#     }
#
#     return habitat_sim.Configuration(sim_cfg, [agent_cfg])
#
#
# sim_settings = {
#     "width": 256,  # Spatial resolution of the observations
#     "height": 256,
#     "scene": test_scene_path,  # Scene path
#     "default_agent": 0,
#     "sensor_height": 1.5,  # Height of sensors in meters
#     "color_sensor": rgb_sensor,  # RGB sensor
#     "depth_sensor": depth_sensor,  # Depth sensor
#     "semantic_sensor": semantic_sensor,  # Semantic sensor
#     "seed": 1,  # used in the random navigation
#     "enable_physics": False,  # kinematics only
# }
#
# cfg = make_cfg(sim_settings)
#
# sim = habitat_sim.Simulator(cfg)
#
# # sim.seed(np.random.randint(0, 100000)) #解除注释可以随机初始化机器人
#
# # initialize an agent
# agent = sim.initialize_agent(sim_settings["default_agent"])
#
# # Set agent state
# agent_state = habitat_sim.AgentState()
# # agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
# # 下面的代码，允许agent随机初始化位置
#
# random_navigable_position = sim.pathfinder.get_random_navigable_point()
# # agent_state = agent.get_state()
# # print(random_navigable_position)
# # print(sim.pathfinder.is_navigable(random_navigable_position))
# bounds = sim.pathfinder.get_bounds()
# x_min, y_min, z_min = bounds[0]  # 世界坐标系的最小值
# x_max, y_max, z_max = bounds[1]  # 世界坐标系的最大值
#
# agent_state.position = random_navigable_position
# agent.set_state(agent_state)
#
# total_frames = 0
# max_frames = 1000
# action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
#
# observations = sim.reset()
# agent_state = habitat_sim.AgentState()
#
# # agent_position = agent.get_state().position
# # agent_rotation = agent.get_state().rotation
# # agent_states = (agent_position, agent_rotation)
# meters_per_pixel=args.map_resolution/10
# height=args.camera_height

obs_map = np.load('topo_vis/obs_map.npy')
exp_map = np.load('topo_vis/exp_map.npy')

g = load_graph('topo_vis/test_graph.pkl')
# sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
agent_positions = {}
num_point = 0
for agent_states in g.vs['agent_state']:
    agent_positions[num_point]=[]
    for agent_state in agent_states:
        agent_positions[num_point].append([int(agent_state[0]/args.map_resolution), int(agent_state[1]/args.map_resolution)])
    num_point += 1

display_map(obs_map+exp_map, agent_positions)
graph_visualization(g, obs_map+exp_map)