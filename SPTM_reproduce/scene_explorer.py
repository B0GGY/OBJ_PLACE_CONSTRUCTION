import torch
import numpy as np
import igraph as ig
from model import PlaceNet
from SPTM_argument import get_args
from PIL import Image
import pickle
from habitat.utils.visualizations import maps
from utils.map_builder import MapBuilder
from habitat_sim.utils.common import quat_to_angle_axis
import utils.pose as pu
import quaternion
import habitat_sim
import os
import cv2
import random

args = get_args()

class Explorer:
    def __init__(self, test_scene):
        self.test_scene = test_scene
        self.rgb_sensor = True
        self.depth_sensor = True
        self.semantic_sensor = False
        self.sim_settings = None
        self.cfg = None
        self.sim = None
        self.random_init = True

        # data sample
        self.rgbd_list = []
        self.position_list = []

    def preprocess_depth(self, depth):
        depth /= 10.
        # depth = depth[:, :, 0] * 1
        mask2 = depth > 0.99  # 0.99
        depth[mask2] = 0.
        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()  # replace the outlier with the max depth

        mask1 = depth == 0
        depth[mask1] = np.NaN
        depth = depth * 1000.

        return depth

    def make_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]
        sim_cfg.enable_physics = settings["enable_physics"]

        # Note: all sensors must have the same resolution
        sensors = {
            "color_sensor": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
                "hfov":settings['hfov']
            },
            "depth_sensor": {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "semantic_sensor": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
        }

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            if settings[sensor_uuid]:
                sensor_spec = habitat_sim.CameraSensorSpec()
                sensor_spec.uuid = sensor_uuid
                sensor_spec.sensor_type = sensor_params["sensor_type"]
                sensor_spec.resolution = sensor_params["resolution"]
                sensor_spec.position = sensor_params["position"]

                sensor_specs.append(sensor_spec)

        # Here you can specify the amount of displacement in a forward action and the turn angle
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=args.march_amount)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=args.turn_amount)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=args.turn_amount)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def get_settings(self):
        test_scene_path = os.path.join(args.data_root, self.test_scene, '{}.glb'.format(self.test_scene))
        sim_settings = {
            "width": args.env_frame_width,  # Spatial resolution of the observations
            "height": args.env_frame_height,
            "scene": test_scene_path,  # Scene path
            "hfov": args.hfov,
            "default_agent": 0,
            "sensor_height": args.camera_height,  # Height of sensors in meters
            "color_sensor": self.rgb_sensor,  # RGB sensor
            "depth_sensor": self.depth_sensor,  # Depth sensor
            "semantic_sensor": self.semantic_sensor,  # Semantic sensor
            "seed": 1,  # used in the random navigation
            "enable_physics": False,  # kinematics only
        }

        return sim_settings

    def get_sim_location(self, agent):
        agent_state = agent.get_state()
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def init_scene(self):
        self.sim_settings = self.get_settings()
        self.cfg = self.make_cfg(self.sim_settings)
        self.sim = habitat_sim.Simulator(self.cfg)


    def explore(self):
        self.init_scene()
        if self.random_init:
            self.sim.seed(np.random.randint(0, 100000)) #解除注释可以随机初始化机器人
        agent = self.sim.initialize_agent(self.sim_settings["default_agent"])

        # Set agent state
        agent_state = habitat_sim.AgentState()
        # agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
        # 下面的代码，允许agent随机初始化位置

        random_navigable_position = self.sim.pathfinder.get_random_navigable_point()
        # agent_state = agent.get_state()
        # print(random_navigable_position)
        # print(sim.pathfinder.is_navigable(random_navigable_position))

        agent_state.position = random_navigable_position
        agent.set_state(agent_state)

        total_frames = 0
        max_frames = args.max_steps
        action_names = list(self.cfg.agents[self.sim_settings["default_agent"]].action_space.keys())

        observations = self.sim.reset()
        agent_state = habitat_sim.AgentState()

        # agent_position = agent.get_state().position
        # agent_rotation = agent.get_state().rotation
        # agent_states = (agent_position, agent_rotation)
        meters_per_pixel = args.map_resolution / 10
        height = args.camera_height

        # sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
        # topology_mapper.graph_vis(sim_topdown_map)
        agent_states = self.get_sim_location(agent)  # agent.get_state()  # 机器人初始位姿
        rgb = observations["color_sensor"]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB).transpose(2,0,1)

        depth = self.preprocess_depth(observations['depth_sensor'])
        depth = np.expand_dims(depth, axis=0)
        rgbd = np.concatenate((rgb, depth), axis=0) # 前三个维度是rgb，最后一个是depth
        self.rgbd_list.append(rgbd)
        self.position_list.append(agent_states)

        while total_frames < max_frames-1:
            action = random.choice(action_names)
            # action = 'turn_left'
            for _ in range(args.action_repeat):
                observations = self.sim.step(action)
                agent_states = self.get_sim_location(agent)  # agent.get_state()
            rgb = observations["color_sensor"]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB).transpose(2,0,1)

            depth = self.preprocess_depth(observations['depth_sensor'])
            depth = np.expand_dims(depth, axis=0)

            rgbd = np.concatenate((rgb, depth), axis=0)

            self.rgbd_list.append(rgbd)
            self.position_list.append(agent_states)

            total_frames += 1
        # print(len(self.rgbd_list), len(self.position_list))
        return self.rgbd_list, self.position_list

    def reset_explorer(self, new_scene_name = None):
        if self.sim:
            self.sim.close()
        self.rgbd_list = []
        self.position_list = []

        if new_scene_name: # 说明需要重新加载场景
            self.test_scene = new_scene_name


if __name__ == '__main__':
    test_class = Explorer('1LXtFkjw3qL')
    test_class.explore()
    test_class.reset_explorer()
    test_class.explore()
    test_class.reset_explorer('1pXnuDYAj8r')
    test_class.explore()