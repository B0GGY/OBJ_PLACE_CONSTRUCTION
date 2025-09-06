import numpy as np
from model import PlaceNet
from SPTM_argument import get_args
import torch

args = get_args()

def top_number_to_threshold(n, top_number, values):
  top_number = min([top_number, n])
  threshold = np.percentile(values, (n - top_number) * 100 / float(n))
  return threshold

def sieve(shortcuts, top_number):
  if top_number == 0:
    return []
  probabilities = shortcuts[:, 0]
  n = shortcuts.shape[0]
  threshold = top_number_to_threshold(n, top_number, probabilities)
  print('Confidence threshold for top', top_number, 'out of', n, ':', threshold)
  sieved_shortcut_indexes = []
  for index in range(n):
    if probabilities[index] >= threshold:
      sieved_shortcut_indexes.append(index)
  return shortcuts[sieved_shortcut_indexes]

class InputProcessor:
    def __init__(self):
        self.placenet = PlaceNet().to(args.device)
        self.placenet.load_state_dict(torch.load('../model_params/model_params_52_0.9307131280388979_test.pth'))
        self.placenet.eval()
        self.encoder = self.placenet.resnet18
        self.edge_model = self.placenet.new_layers
        self.tensor_to_predict = None

    def preprocess_input(self, input):
        # if HIGH_RESOLUTION_VIDEO:
        #     return double_downsampling(input)
        # else:
        return input

    def set_memory_buffer(self, keyframes):
        # 这个函数需要给所有的输入帧编码并保存到一个list中
        # keyframes = [self.preprocess_input(keyframe) for keyframe in keyframes]
        # keyframes = torch.from_numpy(keyframes).float().to(args.device)
        memory_codes = self.encoder(torch.from_numpy(keyframes).float().to(args.device))
        list_to_predict = []
        for index in range(keyframes.shape[0]):
            x = torch.concat((memory_codes[0:1,:], memory_codes[index:index+1,:]), dim=0)
            list_to_predict.append(x)
        self.tensor_to_predict = torch.stack(list_to_predict, dim=0)


    def predict_single_input(self, input):
        # keyframes = [self.preprocess_input(keyframe) for keyframe in keyframes]
        memory_codes = self.encoder(torch.from_numpy(input).float().to(args.device))
        # [100, 2, 512]
        for index in range(self.tensor_to_predict.shape[0]):
            self.tensor_to_predict[index,0:1,:] = memory_codes
        probabilities = self.edge_model(self.tensor_to_predict.view(self.tensor_to_predict.shape[0],-1))

        return probabilities[:, 1]

class SPTM:
    def __init__(self):
        self.input_processor = InputProcessor()
        self.shortcuts = None

    def set_memory_buffer(self, keyframes):
        self.input_processor.set_memory_buffer(keyframes)

    def compute_shortcuts(self, rgbd_list, position_list):
        rgbd_list = np.array(rgbd_list)
        rgb_seq = rgbd_list[:,:3,:,:]
        depth_seq = rgbd_list[:,3:,:,:]

        self.set_memory_buffer(rgb_seq)

        shortcuts_matrix = []
        for k in range(len(rgbd_list)):
            probabilities = self.input_processor.predict_single_input(rgb_seq[k:k+1,:,:,:]).cpu().detach().numpy()
            shortcuts_matrix.append(probabilities)
        shortcuts = self.smooth_shortcuts_matrix(shortcuts_matrix, position_list)
        self.shortcuts = sieve(shortcuts, args.SMALL_SHORTCUTS_NUMBER)
        print(self.shortcuts.shape)

    def get_distance(self, point1, point2):
        """
        计算欧几里得距离
        :param point1: 点1的坐标 (list, tuple, or np.array)
        :param point2: 点2的坐标 (list, tuple, or np.array)
        :return: 两点之间的欧几里得距离
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def smooth_shortcuts_matrix(self, shortcuts_matrix, keyframe_coordinates):
        for first in range(len(shortcuts_matrix)):
            for second in range(first+1, len(shortcuts_matrix)):
                shortcuts_matrix[first][second] = (shortcuts_matrix[first][second] +
                                                      shortcuts_matrix[second][first]) / 2.0
        shortcuts = []
        for first in range(len(shortcuts_matrix)):
            for second in range(first + 1 + args.MIN_SHORTCUT_DISTANCE, len(shortcuts_matrix)):
                values = []
                for shift in range(-args.SHORTCUT_WINDOW, args.SHORTCUT_WINDOW + 1):
                    first_shifted = first + shift
                    second_shifted = second + shift
                    if first_shifted < len(shortcuts_matrix) and second_shifted < len(shortcuts_matrix) and first_shifted >= 0 and second_shifted >= 0:
                                values.append(shortcuts_matrix[first_shifted][second_shifted])
                quality = np.median(values)
                print(values)
                print(quality)
                distance = self.get_distance(keyframe_coordinates[first],
                                        keyframe_coordinates[second])
                shortcuts.append((quality, first, second, distance))
        return np.array(shortcuts)


if __name__ == '__main__':
    test_class = SPTM()