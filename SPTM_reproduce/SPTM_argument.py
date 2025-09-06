import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Active-Neural-SLAM')

    # network settings
    parser.add_argument('--device', type=str, default='cuda:0')

    ## Data generation
    parser.add_argument('--max_steps', type=int, default=100,
                        help='max steps for sampling data (default: 10000)')

    parser.add_argument('--action_repeat', type=int, default=4, help='number of repeat one action')

    # env&agent settings
    parser.add_argument('--turn_amount', type=int, default=15)
    parser.add_argument('--march_amount', type=int, default=0.25)

    parser.add_argument('--data_root', type=str, default='../data/scene_datasets/mp3d')
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
    parser.add_argument('--MIN_SHORTCUT_DISTANCE', type=int, default=5)
    parser.add_argument('--SHORTCUT_WINDOW', type=int, default=10)
    parser.add_argument('--SMALL_SHORTCUTS_NUMBER', type=int, default=2000) #这个参数可能需要调整，因为和scene有关


    args = parser.parse_args()

    return args