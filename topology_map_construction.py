import torch
import numpy as np
import igraph as ig
from model import PlaceNet
from arguments import get_args
from PIL import Image
import pickle
from habitat.utils.visualizations import maps
from utils.map_builder import MapBuilder
from habitat_sim.utils.common import quat_to_angle_axis
import utils.pose as pu
import quaternion

def get_sim_location(agent):
    agent_state = agent.get_state()
    x = -agent_state.position[2]
    y = -agent_state.position[0]
    axis = quaternion.as_euler_angles(agent_state.rotation)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = quaternion.as_euler_angles(agent_state.rotation)[1]
    else:
        o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o

class TopologyMap:
    def __init__(self):
        self.args = get_args()
        # 初始化拓扑地图
        self.g = ig.Graph()
        self.g.vs['img_feature'] = []
        self.g.vs['agent_state'] = [] # 用来可视化最后的graph的
        print(self.g)
        # 初始化地点识别网络
        self.placenet = PlaceNet().to(self.args.device)
        self.placenet.load_state_dict(torch.load('model_params/model_params_52_0.9307131280388979_test.pth'))
        self.placenet.eval()
        # 地图构建参数
        self.score_t = 0.8  # 原文用的是shortcut的数量结合总节点数量来确定这个阈值，这边简化成为固定阈值
        self.last_loc = 0 #用于记录机器人的上一时刻所在的节点
        self.time_t = 5
        # 格点地图构建
        self.mapper = self.build_mapper()
        self.curr_loc_gt = [self.args.map_size_cm / 100.0 / 2.0,
                         self.args.map_size_cm / 100.0 / 2.0, 0.] # 后面graph中保存的agent state以这个为准，记得修改！！！
        # 这个curr_loc_gt和原版的不同是角度是弧度制
        self.last_pose = None  # 这个变量是用来记录机器人坐标系下的上一个位置姿态的，用来计算地图坐标系的坐标更新量用的
        self.map = None
        self.explored_map = None

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

    def map_update(self, agent_state, depth):
        # position = [agent_state.position[2], agent_state.position[0], agent_state.position[1]]
        # rotation = quat_to_angle_axis(agent_state.rotation)[0]  # 这个的计算有问题
        # position = agent_state
        print('agent_state {}'.format(agent_state))
        if self.last_pose is None:
            # 说明是第一帧输入
            mapper_gt_pose = (self.curr_loc_gt[0] * 100.0,
                              self.curr_loc_gt[1] * 100.0,
                              self.curr_loc_gt[2])
            self.last_pose = agent_state#(position[0], position[1], rotation)
        else:
            # 首先计算位置姿态更新量
            # diff_pose = (position[0]-self.last_pose[0], position[1]-self.last_pose[1], rotation-self.last_pose[2])
            diff_pose = pu.get_rel_pose_change(agent_state, self.last_pose) # 这个rotation还是有点问题，在穿过pi时计算会出现问题
            print(diff_pose)
            # 更新地图坐标系下坐标
            # self.curr_loc_gt[0] += diff_pose[0]
            # self.curr_loc_gt[1] += diff_pose[1]
            # self.curr_loc_gt[2] += diff_pose[2]
            self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt,
                               (diff_pose[0], diff_pose[1], diff_pose[2]))
            print(self.curr_loc_gt)
            mapper_gt_pose = (self.curr_loc_gt[0] * 100.0,
                              self.curr_loc_gt[1] * 100.0,
                              np.deg2rad(self.curr_loc_gt[2]))

            # 更新上一时刻位置
            self.last_pose = agent_state#(position[0], position[1], rotation)

        fp_proj, gt_map, fp_explored, explored_map, pano_map, pano_exp = self.mapper.update_map(depth, mapper_gt_pose)
        self.map = gt_map
        self.explored_map = explored_map
        return mapper_gt_pose

    def input_img(self, img_feature, agent_state, depth):  # 输入的是图片与深度图（暂时）
        # 为可视化创建格点地图
        agent_state = self.map_update(agent_state, depth)
        img_feature = img_feature.transpose(2,0,1)

        # plt.imshow(self.map+self.explored_map)
        # plt.show()
        #在图片输入之后需要立刻进行维度变换，以符合网络要求
        img_feature = torch.from_numpy(img_feature).unsqueeze(dim=0)
        img_feature = img_feature.to(self.args.device)
        if self.g.vcount() == 0: # 创建第一个地点节点
            self.g.add_vertex()
            self.g.vs[0]['img_feature']=[img_feature]
            self.g.vs[0]['agent_state']=[agent_state]
        else:  # 说明不是第一个地点节点，需要和现有的所有节点计算相似度
            revisit_idx = self.revisit_check(img_feature)

            if revisit_idx is None:
                # 说明是新的地点
                # 新建节点并保存图片
                self.g.add_vertex()
                self.g.vs[self.g.vcount()-1]['img_feature'] = [img_feature]
                self.g.vs[self.g.vcount() - 1]['agent_state'] = [agent_state]
                # 连接上一个时刻位置
                self.g.add_edge(self.last_loc, self.g.vcount()-1)
                # 更新机器人位置
                self.last_loc = self.g.vcount()-1

            else: # 说明是曾经访问过的地点
                # 更新原始节点特征
                self.update_node_feature(img_feature, agent_state, revisit_idx)
                # 将上一时刻节点和重新访问节点连接（除了两者相等或未超过时间阈值）
                if abs(self.last_loc - revisit_idx) > self.time_t and not self.g.are_adjacent(self.last_loc, revisit_idx):
                    self.g.add_edge(self.last_loc, revisit_idx)
                    self.last_loc = revisit_idx # 更新机器人位置

        # print(self.g)
        # print(self.g.vs['img_feature'])

    def revisit_check(self, img_feature):
        # 这里的归类机制是，计算所有的节点的概率，然后取最高的那个，每个节点的概率是所有图片的得分的平均
        scores = []
        for exist_imgs in self.g.vs['img_feature']:
            local_scores = []
            for img in exist_imgs:
                input_data = torch.concatenate((img,img_feature),dim=1).float().to(self.args.device)
                output = self.placenet(input_data)
                local_scores.append(output.cpu().detach())
                # label = torch.argmax(output)
                # print(label)

            mean_scores = torch.mean(torch.stack(local_scores, dim=0), dim=0)
            scores.append(mean_scores)
            # mean_scores = torch.median(torch.stack(local_scores, dim=0), dim=0)#
            # scores.append(mean_scores.values)
        scores = torch.concatenate(scores)
        max_score_idx = torch.argmax(scores[:,1])
        max_score = scores[max_score_idx, 1]
        if max_score > self.score_t:
            #说明存在相同节点
            return max_score_idx.cpu().item()
        else:
            #说明不存在相同节点
            return None

    def update_node_feature(self, img_feature, agent_state, revisit_idx):
        self.g.vs[revisit_idx]['img_feature'].append(img_feature)
        self.g.vs[revisit_idx]['agent_state'].append(agent_state)

    # def graph_vis(self, topdown_map, key_points=None):
    #     plt.figure(figsize=(12, 8))
    #     ax = plt.subplot(1, 1, 1)
    #     ax.axis("off")
    #     plt.imshow(topdown_map)
    #     # plot points on map
    #     if key_points is not None:
    #         for point in key_points:
    #             plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    #     plt.show(block=False)
    #     plt.pause(3)

    def save_graph(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.g, f)
        np.save('topo_vis/obs_map.npy', self.map)
        np.save('topo_vis/exp_map.npy', self.explored_map)

    def build_mapper(self): # 构建gt map用来可视化
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['fov'] = self.args.hfov
        params['resolution'] = self.args.map_resolution
        params['map_size_cm'] = self.args.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.args.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['visualize'] = self.args.visualize
        params['obs_threshold'] = self.args.obs_threshold
        # self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
        #                                      self.args.map_resolution)
        mapper = MapBuilder(params)
        return mapper

if __name__ == '__main__':
    import habitat_sim
    import habitat
    import cv2
    import os
    from matplotlib import pyplot as plt
    import random

    args = get_args()
    rgb_sensor = True
    depth_sensor = True
    semantic_sensor = False
    test_scene = '7y3sRwLe3Va'

    test_scene_path = os.path.join(args.data_root, test_scene, '{}.glb'.format(test_scene))
    # test_img1 = np.array(Image.open('/home/boggy/PycharmProjects/Habitat/data_sample/1LXtFkjw3qL/{}.png'.format(0)))
    # test_img2 = np.array(Image.open('/home/boggy/PycharmProjects/Habitat/data_sample/1LXtFkjw3qL/{}.png'.format(30)))
    # test_img3 = np.array(Image.open('/home/boggy/PycharmProjects/Habitat/data_sample/1LXtFkjw3qL/{}.png'.format(10)))
    # # print(type(test_img1))
    topology_mapper = TopologyMap()
    # test_class.input_img(test_img1, None)
    # test_class.input_img(test_img2, None)
    # test_class.input_img(test_img3, None)

    def make_cfg(settings):
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


    sim_settings = {
        "width": args.env_frame_width,  # Spatial resolution of the observations
        "height": args.env_frame_height,
        "scene": test_scene_path,  # Scene path
        "hfov":args.hfov,
        "default_agent": 0,
        "sensor_height": args.camera_height,  # Height of sensors in meters
        "color_sensor": rgb_sensor,  # RGB sensor
        "depth_sensor": depth_sensor,  # Depth sensor
        "semantic_sensor": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
    }

    cfg = make_cfg(sim_settings)

    sim = habitat_sim.Simulator(cfg)

    # sim.seed(np.random.randint(0, 100000)) #解除注释可以随机初始化机器人

    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])

    # Set agent state
    agent_state = habitat_sim.AgentState()
    # agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
    # 下面的代码，允许agent随机初始化位置

    random_navigable_position = sim.pathfinder.get_random_navigable_point()
    # agent_state = agent.get_state()
    # print(random_navigable_position)
    # print(sim.pathfinder.is_navigable(random_navigable_position))

    agent_state.position = random_navigable_position
    agent.set_state(agent_state)

    total_frames = 0
    max_frames = 1000
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

    observations = sim.reset()
    agent_state = habitat_sim.AgentState()

    # agent_position = agent.get_state().position
    # agent_rotation = agent.get_state().rotation
    # agent_states = (agent_position, agent_rotation)
    meters_per_pixel=args.map_resolution/10
    height=args.camera_height

    # sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
    # topology_mapper.graph_vis(sim_topdown_map)
    agent_states = get_sim_location(agent)#agent.get_state()  # 机器人初始位姿
    rgb = observations["color_sensor"]
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
    depth = topology_mapper.preprocess_depth(observations['depth_sensor'])

    # plt.imshow(depth)
    # plt.show()
    topology_mapper.input_img(rgb, agent_states, depth)

    while total_frames < max_frames:
        action = random.choice(action_names)
        # action = 'turn_left'
        for _ in range(args.action_repeat):
            observations = sim.step(action)
            agent_states = get_sim_location(agent)#agent.get_state()
        rgb = observations["color_sensor"]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        cv2.imwrite(os.path.join('topo_vis', 'trajectory_rgb','{}.png'.format(total_frames)), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        depth = topology_mapper.preprocess_depth(observations['depth_sensor'])
        # print(agent_states)
        topology_mapper.input_img(rgb, agent_states, depth)
        total_frames += 1
    topology_mapper.save_graph('topo_vis/test_graph.pkl')
