import habitat_sim
import habitat
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
from habitat.utils.visualizations import maps
import random
import cv2
from arguments import get_args
import torch
from torch.utils.data import DataLoader,IterableDataset

# test_scene = "data/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"

class DataSampler(IterableDataset):
    def __init__(self, split='train'):
        self.args = get_args()
        self.max_frames = self.args.max_steps
        self.rgb_sensor = True  # @param {type:"boolean"}
        self.depth_sensor = False  # @param {type:"boolean"}
        self.semantic_sensor = False  # @param {type:"boolean"}
        self.data_save = False
        self.time_stamp = 0
        self.max_frames = self.args.max_steps
        self.resample_t = 7
        self.MAX_ACTION_DISTANCE=self.args.MAX_ACTION_DISTANCE
        self.MAX_CONTINUOUS_PLAY = self.args.MAX_CONTINUOUS_PLAY
        self.NEGATIVE_SAMPLE_MULTIPLIER = self.args.NEGATIVE_SAMPLE_MULTIPLIER
        if split == 'train':
            self.BATCH_SIZE = self.args.batch_size
        else:
            self.BATCH_SIZE = 1
        self.data_root = self.args.data_root
        self.train_set = ['1LXtFkjw3qL','1pXnuDYAj8r','2azQ1b91cZZ','2n8kARJN3HM','2t7WUuJeko7','5LpN3gDmAk7','5q7pvUzZiYa','5ZKStnWn8Zo']#['1LXtFkjw3qL']#
        self.val_set = ['7y3sRwLe3Va','8WUmhLawc2A']
        self.EDGE_EPISODES = self.args.EDGE_EPISODES # 每个场景采样的次数
        self.split = split # train or val

    # # Change to do something like this maybe: https://stackoverflow.com/a/41432704
    # def display_save(self, rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    #
    #     scene_name = test_scene.split('/')[-1]
    #     scene_name = scene_name.split('.')[0]
    #
    #     try:
    #         os.makedirs(os.path.join('data_sample', scene_name))
    #     except:
    #         pass
    #
    #     cv2.imwrite(os.path.join('data_sample', scene_name,'{}.png'.format(self.time_stamp)), cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR))
    #     self.time_stamp += 1

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
                "move_forward", habitat_sim.agent.ActuationSpec(amount=self.args.march_amount)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=self.args.turn_amount)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=self.args.turn_amount)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def trajectory_generator(self, test_scene_path):
        sim_settings = {
            "width": 256,  # Spatial resolution of the observations
            "height": 256,
            "scene": test_scene_path,  # Scene path
            "default_agent": 0,
            "sensor_height": 1.5,  # Height of sensors in meters
            "color_sensor": self.rgb_sensor,  # RGB sensor
            "depth_sensor": self.depth_sensor,  # Depth sensor
            "semantic_sensor": self.semantic_sensor,  # Semantic sensor
            "seed": 1,  # used in the random navigation
            "enable_physics": False,  # kinematics only
        }

        x = []

        cfg = self.make_cfg(sim_settings)
        # cfg = make_simple_cfg(sim_settings)
        #
        # try:  # Needed to handle out of order cell run in Colab
        #     sim.close()
        # except NameError:
        #     pass
        sim = habitat_sim.Simulator(cfg)

        sim.seed(np.random.randint(0,100000))
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
        action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

        observations = sim.reset()
        random_navigable_position = agent.get_state().position

        max_dist = 0

        while total_frames < self.max_frames:

            rgb = np.array([])
            semantic = np.array([])
            depth = np.array([])

            # print('----')
            # print(random_navigable_position)
            # print(agent.get_state().position)
            if np.sqrt(np.sum((random_navigable_position - agent.get_state().position) ** 2)) > max_dist:
                max_dist = np.sqrt(np.sum((random_navigable_position - agent.get_state().position) ** 2))
            # print('*****')
            action = random.choice(action_names)
            # print("action", action)
            for _ in range(self.args.action_repeat):
                observations = sim.step(action)

            if self.rgb_sensor:
                rgb = observations["color_sensor"]
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
                # rgb = rgb.transpose(2,0,1)
            if self.semantic_sensor:
                semantic = observations["semantic_sensor"]
            if self.depth_sensor:
                depth = observations["depth_sensor"]

            # if self.data_save:
            #     self.display_save(rgb, semantic, depth)
            x.append(rgb)
            total_frames += 1
        if max_dist<=self.resample_t:
            print('resampling...')
            x = self.trajectory_generator(test_scene_path)
        sim.close()
        return x

    def data_generator(self):
        # while True:
        x_result = []
        y_result = []
        test_scenes = []
        if self.split == 'train':
            test_scenes = self.train_set
        else:
            test_scenes = self.val_set
        for test_scene in test_scenes:
            test_scene_path = os.path.join(self.data_root, test_scene, '{}.glb'.format(test_scene))
            for _ in range(self.EDGE_EPISODES):
                x = self.trajectory_generator(test_scene_path)  # 生成某个场景的一个轨迹
                first_second_label = []
                current_first = 0
                while True:
                    y = None
                    current_second = None
                    if random.random() < 0.5:  # 50%的机率产生正负样本
                        y = 1  # 产生同一地点样本
                        second = current_first + random.randint(1, self.MAX_ACTION_DISTANCE)
                        # 随机在20step中选择一个帧
                        if second >= self.MAX_CONTINUOUS_PLAY:  # 检查是否超过数据总长度
                            break  # 如果超过总长度，说明当前的轨迹已经采集数据完毕
                        current_second = second
                    else:
                        y = 0  # 产生不同地点样本
                        second = current_first + random.randint(1, self.MAX_ACTION_DISTANCE)
                        if second >= self.MAX_CONTINUOUS_PLAY:
                            break
                        current_second_before = None
                        current_second_after = None
                        # 在当前帧的前后25的范围（如果repeat是4,对应的就是100），算得就是边界距离
                        index_before_max = current_first - self.NEGATIVE_SAMPLE_MULTIPLIER * self.MAX_ACTION_DISTANCE
                        index_after_min = current_first + self.NEGATIVE_SAMPLE_MULTIPLIER * self.MAX_ACTION_DISTANCE
                        if index_before_max >= 0:
                            # 如果前向边界大于0，就选择0到前向边界的一帧作为结果
                            current_second_before = random.randint(0, index_before_max)
                        if index_after_min < self.MAX_CONTINUOUS_PLAY:
                            # 如果后向边界大于0，就选择后向边界到最后一帧之间的一帧作为结果
                            current_second_after = random.randint(index_after_min, self.MAX_CONTINUOUS_PLAY - 1)
                        if current_second_before is None:
                            # 前一帧到头了，说明刚开始，往后采样
                            current_second = current_second_after
                        elif current_second_after is None:
                            # 后一帧到头了，说明快结束了，往前采样
                            current_second = current_second_before
                        else:
                            # 在前一帧或者后一帧中间随机选取
                            if random.random() < 0.5:
                                current_second = current_second_before
                            else:
                                current_second = current_second_after
                    first_second_label.append((current_first, current_second, y))
                    current_first = second + 1  # 每次完成采样后向后移动一帧
                random.shuffle(first_second_label)
                # print('len of sampled data pairs {}'.format(len(first_second_label)))
                for first, second, y in first_second_label:
                    # 为后续数据进行处理
                    future_x = x[second]
                    current_x = x[first]
                    current_y = y
                    x_result.append(np.concatenate((current_x, future_x), axis=2))
                    y_result.append(current_y)
                number_of_batches = len(x_result) / self.BATCH_SIZE
                for batch_index in range(int(number_of_batches)):
                    # xrange在python3中直接替换为range，这部分相当于dataloader的作用
                    from_index = batch_index * self.BATCH_SIZE
                    to_index = (batch_index + 1) * self.BATCH_SIZE
                #     # 这个yield的功能类似于dataloader，就是可以在下一次调用时从中断的地方继续执行
                #     # 在我们的方法中需要融合到torch的dataloader中
                    # numpy输出
                    # yield (np.array(x_result[from_index:to_index]),
                    #        np.array(y_result[from_index:to_index])
                    #                                   )
                    # torch输出
                    # print(np.array(x_result[from_index:to_index]).shape)
                    yield (torch.from_numpy(np.array(x_result[from_index:to_index]).transpose(0,3,1,2)),
                           torch.from_numpy(np.array(y_result[from_index:to_index]))
                           )
                # return x_result, y_result
            # print('x result len {}'.format(len(x_result)))

    def __iter__(self):
        return iter(self.data_generator())

# class GeneratorDataset(IterableDataset):
#     def __init__(self, generator_func, test_scene):
#         self.generator_func = generator_func
#         self.test_scene = test_scene
#
#     def __iter__(self):
#         return iter(self.generator_func(self.test_scene))  # 生成器返回迭代器

if __name__ == '__main__':
    data_sampler = DataSampler(split='train')
    # data_sampler.trajectory_generator(test_scene)
    # gen = data_sampler.data_generator(test_scene)

    # data_sampler.get_data()

    # gen = data_sampler.get_data()#GeneratorDataset(data_sampler.data_generator)
    dataloader = DataLoader(data_sampler, batch_size=None)

    idx = 0
    for i in dataloader:
        # print(i[0].shape)
        # print(i[1].shape)
        # print('---')
        # np.save('tmp/{}_data.npy'.format(idx), i[0])
        # np.save('tmp/{}_label.npy'.format(idx), i[1])
        idx += 1

