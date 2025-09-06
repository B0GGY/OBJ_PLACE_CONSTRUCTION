import habitat_sim
import cv2
import habitat
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
from habitat.utils.visualizations import maps
import random
from habitat_sim.utils.common import quat_from_magnum, quat_to_magnum

display=True


# Change to do something like this maybe: https://stackoverflow.com/a/41432704
def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")
    plt.clf()
    plt.ion()
    # plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    # plt.show(block=False)
    plt.pause(1)
    plt.ioff()


# test_scene = "data/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
#
# sim_settings = {
#     "scene": test_scene,  # Scene path
#     "default_agent": 0,  # Index of the default agent
#     "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
#     "width": 256,  # Spatial resolution of the observations
#     "height": 256,
# }


test_scene = "data/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"

rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}
semantic_sensor = False  # @param {type:"boolean"}

sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

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
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

cfg = make_cfg(sim_settings)
# cfg = make_simple_cfg(sim_settings)
#
# try:  # Needed to handle out of order cell run in Colab
#     sim.close()
# except NameError:
#     pass
sim = habitat_sim.Simulator(cfg)

# print out scene info
def print_scene_recur(scene, limit_output=10):
    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return None


# Print semantic annotation information (id, category, bounding box details)
# about levels, regions and objects in a hierarchical fashion

# scene = sim.semantic_scene
# print_scene_recur(scene)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        print("action: ", action)
        if display:
            display_sample(observations["color_sensor"],observations["semantic_sensor"],observations["depth_sensor"])


# action = "turn_right"
# navigateAndSee(action)
#
# action = "turn_right"
# navigateAndSee(action)
#
# action = "move_forward"
# navigateAndSee(action)
#
# action = "turn_left"
# navigateAndSee(action)

total_frames = 0
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

max_frames = 5

# while total_frames < max_frames:
#     action = random.choice(action_names)
#     print("action", action)
#     observations = sim.step(action)
#     rgb = observations["color_sensor"]
#     # semantic = observations["semantic_sensor"]
#     depth = observations["depth_sensor"]
#
#     if display:
#         display_sample(rgb, depth_obs=depth)
#
#     total_frames += 1



# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)
    plt.pause(3)

# @markdown ###Configure Example Parameters:
# @markdown Configure the map resolution:
meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
# @markdown ---
# @markdown Customize the map slice height (global y coordinate):
custom_height = False  # @param {type:"boolean"}
height = 1  # @param {type:"slider", min:-10, max:10, step:0.1}
# @markdown If not using custom height, default to scene lower limit.
# @markdown (Cell output provides scene height range from bounding box for reference.)

print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
if not custom_height:
    # get bounding box minumum elevation for automatic height
    height = sim.pathfinder.get_bounds()[0][1]

if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:
    # @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
    # This map is a 2D boolean array
    sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)

    if display:
        # @markdown Alternatively, you can process the map using the Habitat-Lab [maps module](https://github.com/facebookresearch/habitat-api/blob/master/habitat/utils/visualizations/maps.py)
        hablab_topdown_map = maps.get_topdown_map(
            sim.pathfinder, height, meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        hablab_topdown_map = recolor_map[hablab_topdown_map]
        print("Displaying the raw map from get_topdown_view:")
        display_map(sim_topdown_map)
        print("Displaying the map from the Habitat-Lab maps module:")
        display_map(hablab_topdown_map)

        # easily save a map to file:
        # map_filename = os.path.join(output_path, "top_down_map.png")
        # imageio.imsave(map_filename, hablab_topdown_map)


# @title Discrete and Continuous Navigation:

# @markdown Take moment to run this cell a couple times and note the differences between discrete and continuous navigation with and without sliding.

# @markdown ---
# @markdown ### Set example parameters:
seed = 7  # @param {type:"integer"}
# @markdown Optionally navigate on the currently configured scene and NavMesh instead of re-loading with defaults:
use_current_scene = False  # @param {type:"boolean"}


sim_settings["seed"] = seed
if not use_current_scene:
    # reload a default nav scene
    sim_settings["scene"] = "./data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
    cfg = make_cfg(sim_settings)
    try:  # make initialization Colab cell order proof
        sim.close()
    except NameError:
        pass
    sim = habitat_sim.Simulator(cfg)
random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])
# set new initial state
sim.initialize_agent(agent_id=0)
agent = sim.agents[0]

# @markdown Seconds to simulate:
sim_time = 10  # @param {type:"integer"}

# @markdown Optional continuous action space parameters:
continuous_nav = True  # @param {type:"boolean"}

# defaults for discrete control
# control frequency (actions/sec):
control_frequency = 3
# observation/integration frames per action
frame_skip = 1
if continuous_nav:
    control_frequency = 5  # @param {type:"slider", min:1, max:30, step:1}
    frame_skip = 12  # @param {type:"slider", min:1, max:30, step:1}


fps = control_frequency * frame_skip
print("fps = " + str(fps))
control_sequence = []
for _action in range(int(sim_time * control_frequency)):
    if continuous_nav:
        # allow forward velocity and y rotation to vary
        control_sequence.append(
            {
                "forward_velocity": random.random() * 2.0,  # [0,2)
                "rotation_velocity": (random.random() - 0.5) * 2.0,  # [-1,1)
            }
        )
    else:
        control_sequence.append(random.choice(action_names))

# create and configure a new VelocityControl structure
vel_control = habitat_sim.physics.VelocityControl()
vel_control.controlling_lin_vel = True
vel_control.lin_vel_is_local = True
vel_control.controlling_ang_vel = True
vel_control.ang_vel_is_local = True

# try 2 variations of the control experiment
for iteration in range(1):
    # reset observations and robot state
    observations = []

    video_prefix = "nav_sliding"
    sim.config.sim_cfg.allow_sliding = True
    # turn sliding off for the 2nd pass
    if iteration == 1:
        sim.config.sim_cfg.allow_sliding = False
        video_prefix = "nav_no_sliding"

    print(video_prefix)

    # manually control the object's kinematic state via velocity integration
    time_step = 1.0 / (frame_skip * control_frequency)
    print("time_step = " + str(time_step))
    for action in control_sequence:

        # apply actions
        if continuous_nav:
            # update the velocity control
            # local forward is -z
            vel_control.linear_velocity = np.array([0, 0, -action["forward_velocity"]])
            # local up is y
            vel_control.angular_velocity = np.array([0, action["rotation_velocity"], 0])

        else:  # discrete action navigation
            discrete_action = agent.agent_config.action_space[action]

            did_collide = False
            if agent.controls.is_body_action(discrete_action.name):
                did_collide = agent.controls.action(
                    agent.scene_node,
                    discrete_action.name,
                    discrete_action.actuation,
                    apply_filter=True,
                )
            else:
                for _, v in agent._sensors.items():
                    habitat_sim.errors.assert_obj_valid(v)
                    agent.controls.action(
                        v.object,
                        discrete_action.name,
                        discrete_action.actuation,
                        apply_filter=False,
                    )

        # simulate and collect frames
        for _frame in range(frame_skip):
            if continuous_nav:
                # Integrate the velocity and apply the transform.
                # Note: this can be done at a higher frequency for more accuracy
                agent_state = agent.state
                previous_rigid_state = habitat_sim.RigidState(
                    quat_to_magnum(agent_state.rotation), agent_state.position
                )

                # manually integrate the rigid state
                target_rigid_state = vel_control.integrate_transform(
                    time_step, previous_rigid_state
                )

                # snap rigid state to navmesh and set state to object/agent
                # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
                end_pos = sim.step_filter(
                    previous_rigid_state.translation, target_rigid_state.translation
                )

                # set the computed state
                agent_state.position = end_pos
                agent_state.rotation = quat_from_magnum(
                    target_rigid_state.rotation
                )
                agent.set_state(agent_state)

                # Check if a collision occured
                dist_moved_before_filter = (
                    target_rigid_state.translation - previous_rigid_state.translation
                ).dot()
                dist_moved_after_filter = (
                    end_pos - previous_rigid_state.translation
                ).dot()

                # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
                # collision _didn't_ happen. One such case is going up stairs.  Instead,
                # we check to see if the the amount moved after the application of the filter
                # is _less_ than the amount moved before the application of the filter
                EPS = 1e-5
                collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter

            # run any dynamics simulation
            sim.step_physics(time_step)

            # render observation
            observations.append(sim.get_sensor_observations())

    print("frames = " + str(len(observations)))
    # video rendering with embedded 1st person view
    # if do_make_video:
    #     # use the vieo utility to render the observations
    #     vut.make_video(
    #         observations=observations,
    #         primary_obs="color_sensor",
    #         primary_obs_type="color",
    #         video_file=output_directory + "continuous_nav",
    #         fps=fps,
    #         open_vid=show_video,
    #     )
    # plt.imshow(observations[0]['color_sensor'])
    # plt.show()
    # for i in range(len(observations)):
    #     cv2.imwrite('tmp/{}.png'.format(i),observations[i]['color_sensor'])
    sim.reset()

# [/embodied_agent_navmesh]