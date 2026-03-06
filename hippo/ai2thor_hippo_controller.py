import dataclasses
import json
import os
import re
import sys
from typing import Union

import cv2
import dataclasses
import json
import os
import re
import sys
from typing import Union

import cv2

import numpy as np
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from langchain.llms import Anyscale
from tqdm import tqdm

from ai2holodeck.constants import THOR_COMMIT_ID, OBJATHOR_ASSETS_DIR
from hippo.simulation.runtimeobjects import RuntimeObjectContainer
from hippo.reconstruction.scenedata import HippoObject, dict2xyztuple
from hippo.utils.file_utils import get_tmp_folder

import numpy as np
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from langchain.llms import Anyscale
from tqdm import tqdm

from ai2holodeck.constants import THOR_COMMIT_ID, OBJATHOR_ASSETS_DIR
from hippo.simulation.runtimeobjects import RuntimeObjectContainer
from hippo.reconstruction.scenedata import HippoObject, dict2xyztuple
from hippo.utils.file_utils import get_tmp_folder

def _get_self_install_dir():
    filepath = __file__
    return "/".join(filepath.split("/")[:-2])

def _get_ai2thor_install_dir():
    return _get_self_install_dir() + "/ai2thor"

def _get_ai2thor_install_build_dir():
    return _get_ai2thor_install_dir() + "/unity/builds"

def _get_ai2thorbuilds_dir(which="fixed"):
    return _get_self_install_dir() + f"/ai2thorbuilds/{which}"


def get_hippo_controller(scene, target_dir=None, objathor_asset_dir=OBJATHOR_ASSETS_DIR, get_runtime_container=False, **kwargs):
    scenepath = scene
    if isinstance(scene, str):
        if scene.endswith(".json"):
            with open(scene, "r") as f:
                scene = json.load(f)
        elif "procthor" in scene.lower():
            num = int(scene.replace("procthor", "")) #
            from procthorprocessing.procthor_utils import get_procthor10k
            scene = get_procthor10k()[num]

    if isinstance(scene, dict):
        if scene["objects"][0].get("IS_HIPPO", False) is True:
            hippo_objects = []
            for object in tqdm(scene["objects"], desc="Loading hippo objects..."):
                ho = HippoObject.from_holodeckdict(object)
                hippo_objects.append(ho)
            if get_runtime_container:
                runtime_container = RuntimeObjectContainer.create(hippo_objects, is_ai2thor_metadata=False)

            concrete_asset_dir = "/".join(scenepath.split("/")[:-1]) + "/concrete_assets/"
            if os.path.exists(concrete_asset_dir):
                target_dir = concrete_asset_dir
            else:
                if target_dir is None:
                    target_dir = get_tmp_folder()
                for ho in tqdm(hippo_objects, desc="Concretizing objects..."):
                    ho.concretize(target_dir, objathor_asset_dir)
        else:
            target_dir = objathor_asset_dir
            runtime_container = None

    try:
        os.unlink( _get_ai2thor_install_build_dir())
    except Exception as e:
        pass
    assert not os.path.exists(_get_ai2thor_install_build_dir())
    os.symlink(_get_ai2thorbuilds_dir(), _get_ai2thor_install_build_dir(), target_is_directory=True)


    cuda_visible_devices = list(
            map(
                int,
                filter(
                    lambda y: y.isdigit(),
                    map(
                        lambda x: x.strip(),
                        os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","),
                    ),
                ),
            )
        )
    print("\n\n~~~AI2THOR GPU NOTICE~~~\n\n")
    if len(cuda_visible_devices) > 0:
        print("AI2Thor controller will use GPU(s):", cuda_visible_devices)
        print("If this occurs on a cluster without a monitor, you need to set up a virtual display.")
        print("You can also diagnose this when facing a `vulkaninfo` error. Remove access to the GPU!")
    else:
        print("AI2Thor controller will use CPU only.")
    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~\n\n")


    try:
        def find_free_port_in_range(start=1024, end=65535):
            import socket
            sock = socket.socket()
            sock.bind(('', 0))
            return sock.getsockname()[-1]
            for port in range(start, end):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('', port))
                        return port
                    except OSError:
                        continue
            raise RuntimeError("No free ports found in range")
        
        ai2thor_port = find_free_port_in_range()
        print("Free port found for AI2Thor controller:", ai2thor_port)

        #os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/mila/c/charlie.gauthier/Holodeck/venv/lib/python3.10/site-packages/cv2/qt/plugins"
        #os.environ["QT_QPA_PLATFORM"] = "/home/mila/c/charlie.gauthier/Holodeck/venv/lib/python3.10/site-packages/cv2/qt/fonts"  # use offscreen rendering to avoid X11 issues
        #os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        #os.environ["TCM_ENABLE"] = "1"
        #os.environ["KMP_INIT_AT_FORK"] = "FALSE"
        #os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress TensorFlow warnings
        #os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["XDF_RUNTIME_DIR"] = "/tmp"  # fix for X11 issues in some environments
        
        #if "ENVIRONMENT" in os.environ:
        #    os.environ.pop("ENVIRONMENT")
        #print("Unsetting slurm variables...")
        # yay jank :)
        #UNSET_SLURM = []
        #for k, v in os.environ.items():
        #    if k.startswith("SLURM"):
        #        os.environ.pop(k)
        #        #UNSET_SLURM.append(f"{k}")
        #if len(UNSET_SLURM) > 0:
        #    UNSET_SLURM = "env -u " + " -u ".join(UNSET_SLURM)
        #else:
        #    UNSET_SLURM = ""
        #print("WILL UNSET SLURM ENV VARS:", UNSET_SLURM)


        print("Local executable path for AI2Thor controller:", f"{_get_ai2thor_install_build_dir()}/thor-Linux64-local/thor-Linux64-local")

        print("Attempting to create AI2Thor controller... this might time out. If so, well, I can't figure out how to fix it.")
        

        #import os
       # import json

        #with open(f"/home/mila/c/charlie.gauthier/Holodeck/hippo/env{'_NO' if 'YERP' in os.environ else ''}_works.json", "w") as f:
        #    json.dump(dict(os.environ), f, indent=2)



        controller = Controller(
            #commid_id=THOR_COMMIT_ID, #'1dfe13e4926bb2e0be475e28405e98514c4035dc', #commit_id=THOR_COMMIT_ID, #'1dfe13e4926bb2e0be475e28405e98514c4035dc', # THOR_COMMIT_ID,
            local_executable_path=f"{_get_ai2thor_install_build_dir()}/thor-Linux64-local/thor-Linux64-local" if sys.platform != "darwin" else None,
            port=ai2thor_port,
            local_build=True  if sys.platform != "darwin" else False,
            agentMode="default",
            makeAgentsVisible=False,
            scene=scene,
            #server_timeout=200, # double the default timeout
            #server_start_timeout=600, # double the default server start timeout
            #gridSize=0.01,
            action_hook_runner=ProceduralAssetHookRunner(
                asset_directory=target_dir,
                asset_symlink=True,
                verbose=True,
                load_file_in_unity=True
            ),
            **kwargs
        )
    except Exception as e:
        from hippo.utils.subproc import run_subproc
        print("Error while creating AI2Thor controller:", e)
        run_subproc(f'pkill python', shell=True, immediately_return=True)
        print("Exiting...")
        os._exit(1)
    
    print("AI2Thor controller created successfully.")

    if get_runtime_container:
        ai2thor_objects = controller.last_event.metadata["objects"]
        if runtime_container is None:
            runtime_container = RuntimeObjectContainer.create(ai2thor_objects, is_ai2thor_metadata=True)
        else:
            runtime_container = runtime_container.update_from_ai2thor(ai2thor_objects)  # fixme maybe always do this insted of just when None
        return controller, runtime_container
    else:
        return controller


def get_hippo_controller_OLDNOW(scene, target_dir=None, objathor_asset_dir=OBJATHOR_ASSETS_DIR, get_runtime_container=False, **kwargs):
    if isinstance(scene, str) and scene.endswith(".json"):
        with open(scene, "r") as f:
            scene = json.load(f)

    if isinstance(scene, dict):
        if scene["objects"][0]["IS_HIPPO"] is True:
            if target_dir is None:
                target_dir = get_tmp_folder()

            hippo_objects = []
            for object in tqdm(scene["objects"], desc="Concretizing objects..."):
                ho = HippoObject.from_holodeckdict(object)
                ho.concretize(target_dir, objathor_asset_dir)
                hippo_objects.append(ho)
            runtime_container = RuntimeObjectContainer.create(hippo_objects, is_ai2thor_metadata=False)
        else:
            target_dir = objathor_asset_dir
            runtime_container = None

    try:
        os.unlink(_get_ai2thor_install_build_dir())
    except:
        pass
    assert not os.path.exists(_get_ai2thor_install_build_dir())
    os.symlink(_get_ai2thorbuilds_dir(), _get_ai2thor_install_build_dir(), target_is_directory=True)

    controller = Controller(
        #commit_id=THOR_COMMIT_ID,
        #local_executable_path="../ai2thorbuilds/original/thor-Linux64-local/thor-Linux64-local",
        local_build=True,
        agentMode="default",
        makeAgentsVisible=False,
        scene=scene,
        #gridSize=0.01,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=target_dir,
            asset_symlink=True,
            verbose=True,
            load_file_in_unity=True
        ),
        **kwargs
    )

    if get_runtime_container:
        ai2thor_objects = controller.last_event.metadata["objects"]
        if runtime_container is None:
            runtime_container = RuntimeObjectContainer.create(ai2thor_objects, is_ai2thor_metadata=True)
        else:
            runtime_container = runtime_container.update_from_ai2thor(ai2thor_objects)
        return controller, runtime_container
    else:
        return controller

import os

import ai2thor.controller
from hippo.simulation.ai2thor_metadata_reader import get_object_list_from_controller, get_robot_inventory
from hippo.simulation.skill_simulator import Simulator

from hippo.simulation.spatialutils.filter_positions import build_grid_graph, filter_reachable_positions

from ai2holodeck.constants import THOR_COMMIT_ID
from hippo.ai2thor_hippo_controller import get_hippo_controller_OLDNOW, get_hippo_controller
from hippo.utils.file_utils import get_tmp_folder

TARGET_TMP_DIR = get_tmp_folder()

def get_controller(scene, get_runtime_container=False, **kwargs):
    if "target_dir" in kwargs:
        target_dir = kwargs.pop("target_dir")
    else:
        target_dir = TARGET_TMP_DIR
    return get_hippo_controller_OLDNOW(scene, target_dir=target_dir, get_runtime_container=get_runtime_container, **kwargs)

    if isinstance(scene, str):
        controller = ai2thor.controller.Controller(commit_id=THOR_COMMIT_ID, scene=scene, **kwargs)
    else:
        assert isinstance(scene, dict)
        if "target_dir" in kwargs:
            target_dir = kwargs.pop("target_dir")
        else:
            target_dir = TARGET_TMP_DIR
        controller = get_hippo_controller_OLDNOW(scene, target_dir=target_dir, **kwargs)
    return controller


from diskcache import FanoutCache
cache = FanoutCache('./diskcache8', size_limit=int(1e9), shards=8)

def get_list_of_objects(scene):
    with open(scene, "r") as f:
        scene_txt = f.read()
    runtime_container = get_runtime_container(scene, scene_txt)
    #return runtime_container.get_object_list_with_children_as_string()
    ret = runtime_container.as_llmjson()
    del ret["robot0"]
    return ret

@cache.memoize(typed=True)
def get_runtime_container(scene, scene_txt_for_memoization):
    print("CACHE MISS: getting runtime container...")

    runtime_container = get_sim(scene, just_runtime_container=True)

    return runtime_container

def resolve_scene_id(floor_name):
    if isinstance(floor_name, int) or floor_name.startswith("FloorPlan"):
        floor_name = str(floor_name)
        return f"FloorPlan{floor_name.replace('FloorPlan', '')}"

    if isinstance(floor_name, str) and "procthor" in floor_name:
        return floor_name

    if not floor_name.endswith(".json"):
        if os.path.exists(floor_name + "/scene.json"):
            floor_name += "/scene.json"

    assert floor_name.endswith(".json")

    return floor_name

    with open(floor_name, "r") as f:
        scene = json.load(f)
    return scene

def get_sim(floor_no, just_controller=False, just_runtime_container=False, just_controller_no_setup=False, humanviewing_params={}, renderInstanceSegmentation=False, width=1250, height=1250, GRID_SIZE=0.25):
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    import jax
    jax.config.update('jax_platform_name', "cpu")


    scene = resolve_scene_id(floor_no)

    assert sum(list(map(int, (just_controller, just_runtime_container, just_controller_no_setup)))) <= 1

    GRID_SIZE = GRID_SIZE
    c, runtime_container = get_hippo_controller(scene, get_runtime_container=True, width=width, height=height,
                                                snapGrid=False, snapToGrid=False, visibilityDistance=1, fieldOfView=90, gridSize=GRID_SIZE,
                                                rotateStepDegrees=20)

    if just_controller_no_setup:
        return c


    all_initial_objects = c.last_event.metadata['objects']

    #c, runtime_container = get_controller(scene, get_runtime_container=True, width=1000, height=1000,
    #                                      snapToGrid=False, visibilityDistance=100, fieldOfView=90, gridSize=0.25,
    #                                      rotateStepDegrees=20)
    no_robot = 1  # len(robots)

    convert = lambda x: np.array([np.array(dict2xyztuple(x))[0], np.array(dict2xyztuple(x))[2]])

    scene_bound_min = convert(c.last_event.metadata['sceneBounds']["center"]) - convert(c.last_event.metadata['sceneBounds']["center"])
    scene_bound_max = convert(c.last_event.metadata['sceneBounds']["center"]) + convert(c.last_event.metadata['sceneBounds']["center"])

    DEFAULT_ROBOT_HEIGHT = 0.95
    DEFAULT_ROBOT_ROT = 90
    CEILING_HEIGHT = c.last_event.metadata["sceneBounds"]['size']["y"]

    random_pose = np.random.uniform(low=scene_bound_min, high=scene_bound_max, size=(1, 2))
    c.step(
        action="TeleportFull",
        position={
            "x": random_pose[0][0],
            "y": DEFAULT_ROBOT_HEIGHT, #scene["metadata"]["agent"]["position"]["y"],
            "z": random_pose[0][1],
        },
        rotation=DEFAULT_ROBOT_ROT, #scene["metadata"]["agent"]["rotation"],
        standing=True,
        horizon=30,
        forceAction=True,
    )

    #gridsize = 1  # step size between points

    # Generate 1D arrays for x and y
    x = np.arange(scene_bound_min[0]+ GRID_SIZE*2, scene_bound_max[0] - GRID_SIZE, GRID_SIZE)
    y = np.arange(scene_bound_min[1]+ GRID_SIZE*2, scene_bound_max[1] - GRID_SIZE , GRID_SIZE)

    # Create 2D grid
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    grid_points = [(x, DEFAULT_ROBOT_HEIGHT, z) for x, z in grid_points]

    minx = min(np.array(grid_points)[:, 0])
    maxx = max(np.array(grid_points)[:, 0])
    minz = min(np.array(grid_points)[:, 2])
    maxz = max(np.array(grid_points)[:, 2])

    grid_points = list(filter(lambda point: minx < point[0] < maxx and minz < point[2] < maxz, grid_points))

    reachable_positions = grid_points

    #c.step(action="GetReachablePositions")
    #print(c.last_event.metadata['errorMessage'])
    #reachable_positions = c.last_event.metadata["actionReturn"]
    #reachable_positions = [(p["x"], p["y"], p["z"]) for p in reachable_positions]

    def clean_convert(point):
        # fixes conversion errors between np, jnp, lists
        import jax.numpy as jnp
        if isinstance(point, jnp.ndarray):
            point = np.array(point)
        if isinstance(point, np.ndarray):
            point = point.tolist()
        rounded = [round(x / GRID_SIZE) * GRID_SIZE for x in point]
        return tuple(rounded)
    full_reachability_graph = build_grid_graph(reachable_positions, GRID_SIZE, clean_convert=clean_convert)
    full_reachability_graph.name = f"full_reachability_graph(GRID_SIZE={GRID_SIZE})"
    full_reachability_graph.clean_convert = clean_convert

    # initialize n agents into the scene
    c.step(
        dict(action='Initialize', agentMode="default", snapGrid=False, snapToGrid=False, gridSize=GRID_SIZE,
             rotateStepDegrees=90, visibilityDistance=1, fieldOfView=90, agentCount=no_robot, renderInstanceSegmentation=renderInstanceSegmentation),
        raise_for_failure=True
    )



    def random_teleport():
        filtered_reachable_positions = filter_reachable_positions(full_reachability_graph, runtime_container)
        import numpy as np
        possible_start_nodes = (list(filtered_reachable_positions.nodes))
        import random
        seed_for_floor_no = hash(floor_no)
        random.seed(seed_for_floor_no)
        possible_start_nodes = list(sorted(possible_start_nodes))
        random_pos = random.choice(possible_start_nodes)# np.random.choice(possible_start_nodes, 1)
        c.step(
            action="TeleportFull",
            position={
                "x": random_pos[0],
                "y": DEFAULT_ROBOT_HEIGHT, # scene["metadata"]["agent"]["position"]["y"],
                "z": random_pos[2],
            },
            rotation=DEFAULT_ROBOT_ROT, #scene["metadata"]["agent"]["rotation"],
            standing=True,
            horizon=30,
            forceAction=True,
        )
    random_teleport()

    THIRD_PARTY_CAMERAS = {
        "robot": None, # ai2thor primitive
        "top_down": 0,
        "altered_first": 1
    }
    # add a top view camera
    event = c.step(action="GetMapViewCameraProperties")
    event = c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

    def get_altered_cam_position():
        cam_position = c.last_event.metadata["agent"]["position"]
        cam_rotation = c.last_event.metadata["agent"]["rotation"]
        print("cam_rotation", cam_rotation)
        camera_horizon = c.last_event.metadata["agent"]["cameraHorizon"]
        cam_rotation['x'] = camera_horizon
        first_person_camera = {
            'position': cam_position,
            'rotation': cam_rotation,
            'orthographic': False,
            'fieldOfView': 90,
        }
        return first_person_camera

    def update_altered_cam_position():
        cam_params = get_altered_cam_position()
        c.step(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=THIRD_PARTY_CAMERAS["altered_first"],
            position=cam_params["position"],
            rotation=cam_params["rotation"],
            fieldOfView=90
        )
    c.step(action="AddThirdPartyCamera", **get_altered_cam_position())
    old_step = c.step
    def new_step(*args,**kwargs):
        if isinstance(args, tuple) and len(args) == 1 and isinstance(args[0], dict):
            kwargs = args[0]
            args = tuple([])
        if isinstance(args, dict):
            kwargs = args
            args = tuple([])
        if kwargs["action"] == "UpdateThirdPartyCamera":
            print("Third Party Camera Updated")
            ret = old_step(*args,**kwargs)
            return ret
        before = c.last_event.metadata # noqa
        ret = old_step(*args,**kwargs)
        middle = c.last_event.metadata  # noqa
        if kwargs["action"].startswith("Move") or kwargs["action"].startswith("Rotate") or kwargs["action"].startswith("Look"):
            update_altered_cam_position()
            after = c.last_event.metadata # noqa
            #print("Middle", middle["agent"]["position"])
            #print("After", after["agent"]["position"])
            i=0
        return ret
    c.step = new_step

    def get_segmented_held_object():
        obj = None
        for obj in c.last_event.metadata['objects']:
            if obj['isPickedUp']:
                break
            obj = None
        if obj is None:
            return None

        if c.last_event.instance_segmentation_frame is None:
            return None

        # Get instance segmentation and frame
        segmented_first_person_image = c.last_event.instance_segmentation_frame
        frame = c.last_event.cv2img

        # Get the object's unique color
        color = tuple(c.last_event.object_id_to_color[obj['name']])  # (r,g,b)

        # 1. Create a mask for the object
        mask = np.all(segmented_first_person_image == color, axis=-1).astype(np.uint8)

        # 2. Add alpha channel (RGBA)
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        # 3. Apply transparency: alpha=0 where mask==0
        rgba_frame[:, :, 3] = mask * 255  # scale 0/1 -> 0/255

        # 4. Find bounding box around the mask to crop
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            cropped_object = rgba_frame[y_min:y_max + 1, x_min:x_max + 1]
        else:
            cropped_object = None  # object not found

        return cropped_object

    c.get_segmented_held_object = get_segmented_held_object





    # maybe need to do this https://github.com/allenai/Holodeck/issues/18#issuecomment-1919531859

    # get reachabel positions

    # randomize postions of the agents  now done above
    # for i in range (no_robot):
    #    init_pos = random.choice(reachable_positions_)
    #    c.step(dict(action="Teleport", position=init_pos, agentId=i))

    # run a few physics steps to make bad objects fall through floor...
    from tqdm import trange
    for _ in trange(10):
        c.step(
            action="MoveAhead",
            moveMagnitude=0.01,
            #rotation=DEFAULT_ROBOT_ROT,  # scene["metadata"]["agent"]["rotation"],
            #standing=True,
            #horizon=30,
            #forceAction=True,
        )
    # fix bad objects...
    for obj in c.last_event.metadata['objects']:
        if obj["position"]['y'] < -1:
            c.step(
                action="SetMassProperties",
                objectId=obj['objectId'],
                mass=0.0,
                drag=1500,
                angularDrag=1500
            )
            y = None
            for otherobj in all_initial_objects:
                if otherobj['objectId'] == obj['objectId']:
                    y = otherobj['position']['y']
                    break
            assert y is not None
            c.step(
                action="PlaceObjectAtPoint",
                objectId=obj['objectId'],
                position={
                    "x": obj['position']['x'],
                    "y": y,#+otherobj["axisAlignedBoundingBox"]["size"]['y'],
                    "z": obj['position']['z'],
                }
            )
    random_teleport()

    runtime_container = runtime_container.set_robots(get_object_list_from_controller(c))
    runtime_container = runtime_container.update_from_ai2thor(get_object_list_from_controller(c))

    simulator = None


    # setting up tools for human viewing
    from hippo.simulation.humanviewing import HumanViewing
    c.humanviewing = HumanViewing(c,runtime_container,None, **humanviewing_params)

    if just_controller:
        return c
    if just_runtime_container:
        c.stop()
        return runtime_container

    simulator = Simulator(controller=c, no_robots=no_robot, objects=runtime_container,
                          full_reachability_graph=full_reachability_graph)
    simulator.start_action_listener()

    hu = HumanViewing(c, runtime_container, simulator, **humanviewing_params)
    c.humanviewing = hu
    simulator.humanviewing = hu

    return simulator
    #return c, runtime_container, no_robot, reachable_positions


