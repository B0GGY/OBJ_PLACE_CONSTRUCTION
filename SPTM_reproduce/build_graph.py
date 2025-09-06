from scene_explorer import Explorer
from sptm import SPTM

def build_graph():
    explorer = Explorer('1LXtFkjw3qL')
    memory = SPTM()
    rgbd_list, position_list = explorer.explore()
    memory.compute_shortcuts(rgbd_list, position_list)


if __name__ == '__main__':
    build_graph()