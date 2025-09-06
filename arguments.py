import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Active-Neural-SLAM')

    ## Data generation
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='max steps for sampling data (default: 10000)')
    parser.add_argument('--action_repeat', type = int, default=4, help='number of repeat one action')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--turn_amount', type=int, default=15)
    parser.add_argument('--march_amount', type=int, default=0.25)
    parser.add_argument('--data_root', type=str, default='data/scene_datasets/mp3d')
    parser.add_argument('--EDGE_EPISODES', type=int, default=2) # 这个根据电脑状况调整，先设置2,原文是10
    parser.add_argument('--MAX_ACTION_DISTANCE', type=int, default=5) # 原文设置为5
    parser.add_argument('--MAX_CONTINUOUS_PLAY', type=int, default=10000)
    parser.add_argument('--NEGATIVE_SAMPLE_MULTIPLIER', type=int, default=5)
    parser.add_argument('--LR', type=float, default=0.0001)
    parser.add_argument('--Beta1', type=float, default=0.9)
    parser.add_argument('--Beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--epoches', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')

    # env_setting
    parser.add_argument('--env_frame_width', type=int, default=256)
    parser.add_argument('--env_frame_height', type=int, default=256)
    parser.add_argument('--camera_height', type=float, default=1.5)
    parser.add_argument('--hfov', type=int, default=90)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--map_size_cm', type=int, default=4800)
    parser.add_argument('--du_scale', type=int, default=2)
    parser.add_argument('--vision_range', type=int, default=60)
    parser.add_argument('-v', '--visualize', type=int, default=1,
                        help='1:Render the frame (default: 0)')
    parser.add_argument('-ot', '--obs_threshold', type=float, default=1)

    # parse arguments
    args = parser.parse_args()

    return args