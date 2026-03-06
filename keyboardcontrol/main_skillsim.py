import time

import cv2
import gzip
import json
import math
import os
from typing import Sequence
from tqdm import tqdm
import numpy as np


actionList = {
    "MoveAhead": "w",
    "MoveBack": "s",
    "MoveLeft": "a",
    "MoveRight": "d",
    "RotateLeft": "q",
    "RotateRight": "e",
    "LookUp": "r",
    "LookDown": "f",
    "Record": "t",
    "FINISH": "p",
    # Interact action
    "PickupObject": "z",
    "PutObject": "x",
    "OpenObject": "c",
    "CloseObject": "v",
    "ToggleObjectOn": "b",
    "ToggleObjectOff": "n",
    "SliceObject": "m",
}


def get_interact_object(env):
    candidates = []
    objectId = ''

    for obj in env.last_event.metadata["objects"]:
        if obj["visible"]:
            candidates.append(obj["objectId"])

    if len(candidates) == 0:
        print('no valid interact object candidates')
        return None
    else:
        print('===========choose index from the candidates==========')

        # Create mapping from characters to indices
        char_to_index = {}
        index = 0

        # Add digits 0-9
        for i in range(10):
            if index < len(candidates):
                char_to_index[str(i)] = index
                print(f'{i} : {candidates[index]}')
                index += 1

        # Add lowercase letters a-z (will also handle uppercase via normalization)
        for c in 'abcdefghijklmnopqrstuvwxyz':
            if index < len(candidates):
                char_to_index[c] = index
                char_to_index[c.upper()] = index  # Map both cases to same index
                print(f'{c}/{c.upper()} : {candidates[index]}')
                index += 1

        # Add remaining uppercase letters if we have more than 36 objects
        for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if index < len(candidates) and c not in char_to_index:
                char_to_index[c] = index
                print(f'{c} : {candidates[index]}')
                index += 1

        # Define shift key codes (common shift key codes)
        SHIFT_CODES = {225, 65505, 65506}  # Left Shift, Right Shift, etc.

        while True:
            keystroke = cv2.waitKey()
            print(f"Raw keystroke: {keystroke}")

            if keystroke == actionList["FINISH"]:
                print("stop interact")
                break

            # Ignore shift keys
            if keystroke in SHIFT_CODES:
                print("Ignoring Shift key...")
                continue

            # Convert keystroke to character
            try:
                # Convert ASCII code to character
                if 0 <= keystroke <= 255:
                    char = chr(keystroke)
                    print(f"Converted to char: '{char}'")

                    # Normalize to lowercase for letters (since we map both cases)
                    if char.isalpha():
                        normalized_char = char.lower()
                    else:
                        normalized_char = char

                    # Check if character is in our mapping
                    if normalized_char in char_to_index:
                        object_index = char_to_index[normalized_char]
                        objectId = candidates[object_index]
                        print(f"Selected object: {objectId}")
                        break
                    else:
                        print(f"INVALID KEY '{char}' - not in available options")
                else:
                    print(f"INVALID KEYCODE {keystroke}")
            except Exception as e:
                print(f"ERROR processing keystroke: {e}")
                continue

        return objectId

def keyboard_play(env, top_down_frames, first_view_frames, is_rotate, rotate_per_frame, HU_PLAN=[], sim=None, USE_ALTERED_FIRST_PERSON=True,
         AUGMENT_FIRST_PERSON=True):
    sim.kill_sim_on_condition_failure = False
    sim.raise_exception_on_condition_failure = False
    env.humanviewing.set_plan(HU_PLAN)
    env.humanviewing.incr_action_idx()
    def render_first_person():
        if USE_ALTERED_FIRST_PERSON:
            frame = env.humanviewing.get_latest_altered_robot_frame()
        else:
            frame = env.humanviewing.get_latest_robot_frame()
        if AUGMENT_FIRST_PERSON:
            frame = env.humanviewing.get_augmented_robot_frame(frame)
        env.humanviewing.display_frame(frame)
        return frame

    #first_view_frame = env.humanviewing.get_augmented_robot_frame(env.humanviewing.get_latest_robot_frame())
    #env.humanviewing.display_frame(first_view_frame)
    first_view_frame = render_first_person()

    if sim is None:
        raise NotImplementedError()

    cv2.imshow("top_view", cv2.cvtColor(env.humanviewing.get_latest_topdown_frame(), cv2.COLOR_RGB2BGR))

    step = 0
    while True:
        keystroke = cv2.waitKey(0)
        step += 1

        if keystroke == ord(actionList["FINISH"]):
            env.stop()
            cv2.destroyAllWindows()
            print("action: STOP")
            break

        robots = [{'name': 'robot1',
                   'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn',
                              'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject',
                              'PullObject']},
                  {'name': 'robot2',
                   'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn',
                              'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject',
                              'PullObject']}]

        if keystroke == ord(actionList["MoveAhead"]):
            action = "MoveAhead"
            print("action: MoveAhead")
        elif keystroke == ord(actionList["MoveBack"]):
            action = "MoveBack"
            print("action: MoveBack")
        elif keystroke == ord(actionList["MoveLeft"]):
            action = "MoveLeft"
            print("action: MoveLeft")
        elif keystroke == ord(actionList["MoveRight"]):
            action = "MoveRight"
            print("action: MoveRight")
        elif keystroke == ord(actionList["RotateLeft"]):
            action = "RotateLeft"
            print("action: RotateLeft")
        elif keystroke == ord(actionList["RotateRight"]):
            action = "RotateRight"
            print("action: RotateRight")
        elif keystroke == ord(actionList["LookUp"]):
            action = "LookUp"
            print("action: LookUp")
        elif keystroke == ord(actionList["LookDown"]):
            action = "LookDown"
            print("action: LookDown")

        elif keystroke == ord(actionList["PickupObject"]):
            action = "PickupObject"
            print("action: PickupObject")
            objectId = get_interact_object(env)
            sim.PickupObject(robots[0], objectId)
            if len(sim.exception_queue) <= 0:
                pickedupobj = objectId
        elif keystroke == ord(actionList["PutObject"]):
            action = "PutObject"
            print("action: PutObject")
            objectId = get_interact_object(env)
            if len(sim.exception_queue) <= 0:
                sim.PutObject(robots[0], pickedupobj, objectId)
                pickedupobj = None
        elif keystroke == ord(actionList["OpenObject"]):
            action = "OpenObject"
            print("action: OpenObject")
            objectId = get_interact_object(env)
            sim.OpenObject(robots[0], objectId)
        elif keystroke == ord(actionList["CloseObject"]):
            action = "CloseObject"
            print("action: CloseObject")
            objectId = get_interact_object(env)
            sim.CloseObject(robots[0], objectId)
        elif keystroke == ord(actionList["ToggleObjectOn"]):
            action = "ToggleObjectOn"
            print("action: ToggleObjectOn")
            objectId = get_interact_object(env)
            sim.ToggleObjectOn(robots[0], objectId)
        elif keystroke == ord(actionList["ToggleObjectOff"]):
            action = "ToggleObjectOff"
            print("action: ToggleObjectOff")
            objectId = get_interact_object(env)
            sim.ToggleObjectOff(robots[0], objectId)
        elif keystroke == ord(actionList["SliceObject"]):
            action = "SliceObject"
            print("action: SliceObject")
            objectId = get_interact_object(env)
            sim.SliceObject(robots[0], objectId)
        else:
            print("INVALID KEY", keystroke)
            continue

        print("Interact object:")
        try:
            print(objectId)
        except UnboundLocalError:
            pass

        if action.startswith("Move") or action.startswith("Rotate") or action.startswith("Look"):
            env.step(action=action)
            pass
        else:
            env.humanviewing.incr_action_idx()
        #time.sleep()

        # agent step
        #if "Object" in action:
        #    pass #env.step(action=action, objectId=objectId, forceAction=True)
        #else:


        #if action == "PutObject" and env.last_event.metadata["error"]:
        print("Waiting for sim to reply")
        print(env.last_action)

        #first_view_frame = env.humanviewing.get_augmented_robot_frame(env.humanviewing.get_latest_altered_robot_frame())
        #env.humanviewing.display_frame(first_view_frame)
        first_view_frame = render_first_person()

        # remove the ceiling
        cv2.imshow("top_view", cv2.cvtColor(env.humanviewing.get_latest_topdown_frame(), cv2.COLOR_RGB2BGR))

        top_down_frames.append(env.humanviewing.get_latest_topdown_frame())
        first_view_frames.append(first_view_frame)


def show_video(frames: Sequence[np.ndarray], fps: int = 10):
    """Show a video composed of a sequence of frames.

    Example:
    frames = [
        controller.step("RotateRight", degrees=5).frame
        for _ in range(72)
    ]
    show_video(frames, fps=5)
    """
    frames = ImageSequenceClip(frames, fps=fps)
    return frames


def export_video(path, frames):
    """Merges all the saved frames into a .mp4 video and saves it to `path`"""

    video = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        5,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        # assumes that the frames are RGB images. CV2 uses BGR.
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.destroyAllWindows()
    video.release()


def compute_rotate_camera_pose(center, pose, degree_per_frame=6):
    """degree_per_frame: set the degree of rotation for each frame"""

    def rotate_pos(x1, y1, x2, y2, degree):
        angle = math.radians(degree)
        n_x1 = (x1 - x2) * math.cos(angle) - (y1 - y2) * math.sin(angle) + x2
        n_y1 = (x1 - x2) * math.sin(angle) + (y1 - y2) * math.cos(angle) + y2

        return n_x1, n_y1

    # print(math.sqrt((pose["position"]["x"]-center["x"])**2 + (pose["position"]["z"]-center["z"])**2))
    x, z = rotate_pos(pose["position"]["x"], pose["position"]["z"], center["x"], center["z"], degree_per_frame)
    pose["position"]["x"], pose["position"]["z"] = x, z

    direction_x = center["x"] - x
    direction_z = center["z"] - z
    pose["rotation"]["y"] = math.degrees(math.atan2(direction_x, direction_z))

    # print(math.sqrt((pose["position"]["x"]-center["x"])**2 + (pose["position"]["z"]-center["z"])**2))

    return pose


def initialize_side_camera_pose(scene_bound, pose, third_fov=60, slope_degree=45, down_angle=70, scale_factor=8):
    """
    down_angle: the x-axis rotation angle of the camera, represents the top view of the front view from top to bottom, which needs to be less than 90 degrees
    ensure the line vector between scene's center & camera 's angel equal down_angle
    scale_factor scale the camera's view, make it larger ensure camera can see the whole scene
    """
    fov_rad = np.radians(third_fov)
    pitch_rad = np.radians(down_angle)
    distance = (scene_bound["center"]["y"] / 2) / np.tan(fov_rad / 2)
    pose["position"]["y"] = scene_bound["center"]["y"] + distance * scale_factor * np.sin(pitch_rad)
    pose["position"]["z"] = scene_bound["center"]["z"] - distance * scale_factor * np.cos(pitch_rad)

    pose["rotation"]["x"] = down_angle

    pose["orthographic"] = False
    del pose["orthographicSize"]

    pose = compute_rotate_camera_pose(scene_bound["center"], pose, slope_degree)

    return pose


def main(scene_name="FloorPlan205_physics", gridSize=0.25, rotateStepDegrees=15,
         BEV=False, slope_degree=45, down_angle=65, use_procthor=False, procthor_scene_file="", procthor_scene_num=100,
         is_rotate=True, rotate_per_frame=6, generate_video=False, generate_gif=False, HU_PLAN=[],USE_ALTERED_FIRST_PERSON=True,
         AUGMENT_FIRST_PERSON=True):
    ## procthor room
    if use_procthor:
        with gzip.open(procthor_scene_file, "r") as f:
            houses = [line for line in tqdm(f, total=10000, desc=f"Loading train")]
        ## procthor train set's room
        house = json.loads(houses[procthor_scene_num])
    else:
        ## select room, 1-30，201-230，301-330，401-430 are ithor's room
        house = scene_name

    #from SMARTLLM.smartllm.utils.resolve_scene import resolve_scene_id

    #floor_no = "/home/charlie/Desktop/Holodeck/SMARTLLM/hipposcenes/9/scene.json"
    #scene = resolve_scene_id(floor_no)
    #from SMARTLLM.smartllm.utils.get_controller import get_controller
    #controller = get_controller(scene, get_runtime_container=False, width=1000, height=1000, snapToGrid=False,
    #                                      visibilityDistance=100, fieldOfView=90, gridSize=0.25, rotateStepDegrees=20)

    from hippo.ai2thor_hippo_controller import get_sim
    sim = get_sim(scene_name, renderInstanceSegmentation=True)
    controller = sim.controller
    """
    _ = Controller(
        agentMode="default",
        visibilityDistance=5,
        renderInstanceSegmentation=True,
        scene=house,
        # step sizes
        gridSize=gridSize,
        snapToGrid=False,
        rotateStepDegrees=rotateStepDegrees,
        # camera properties
        width=1200,
        height=800,
        fieldOfView=90,
        platform=CloudRendering,
    )
    """

    """
    ## add third view camera
    event = controller.step(action="GetMapViewCameraProperties")
    ## third camera's fov
    third_fov = 60

    if not BEV:
        ## top_view(slope)
        pose = initialize_side_camera_pose(event.metadata["sceneBounds"], event.metadata["actionReturn"], third_fov,
                                           slope_degree, down_angle)
    else:
        ## BEV
        pose = event.metadata["actionReturn"]
        is_rotate = False  ## assume that BEV do not need rotation

    event = controller.step(
        action="AddThirdPartyCamera",
        skyboxColor="black",
        fieldOfView=third_fov,
        **pose
    )
    """

    ## collect frame
    first_view_frames = []
    third_view_frames = []

    ## use keyboard control agent
    keyboard_play(controller, third_view_frames, first_view_frames, is_rotate, rotate_per_frame, HU_PLAN, sim, USE_ALTERED_FIRST_PERSON=USE_ALTERED_FIRST_PERSON,
         AUGMENT_FIRST_PERSON=AUGMENT_FIRST_PERSON)

    ## use frames generate video
    if generate_video:

        if not os.path.exists("./video"):
            os.mkdir("./video")

        export_video("./video/first_view_{}.mp4".format(scene_name), first_view_frames)
        export_video("./video/third_view_{}.mp4".format(scene_name), third_view_frames)

    ## use frames generate gif
    if generate_gif:

        if not os.path.exists("./gif"):
            os.mkdir("./gif")

        clip = show_video(third_view_frames, fps=5)
        clip.write_gif("./gif/third_view_{}.gif".format(scene_name))
        clip2 = show_video(first_view_frames, fps=5)
        clip2.write_gif("./gif/first_view_{}.gif".format(scene_name))


if __name__ == "__main__":
    PLAN1 = """
TASK: make a "yellow on black on blue" tower
1. PickUpObj (green)
2. PutDownObj (green, table)
3. PickUpObj (black)
4. PutDown0bj (black, table)
5. PickUpObj (blue)
6. PutDownObj (blue, table)
7. PickUpObj (black)
8. PutDown0bj (black, yellow)
9. PickUpObj (blue)
10. PutDownObj (blue, black)
DONE. Should now have a "yellow on black on blue" tower.
    """

    PLAN2 = """
TASK: make a "yellow on black on blue" tower
1. PickUpObj (green)
2. PickUpObj (black)
3. PickUpObj (blue)
4. PickUpObj (yellow)
5. PutDownObj (blue, table)
6. PutDownObj (black, blue)
7. PutDownObj (yellow, black)
DONE. Should now have a "yellow on black on blue" tower.
        """

    PLAN3 = """
TASK: make a "yellow on black on blue" tower
1. PickUpObj (green)
2. PutDownObj (green, table)
3. PickUpObj (black)
4. PutDown0bj (black, table)
5. PickUpObj (blue)
6. PutDownObj (blue, table)
7. PickUpObj (black)
8. PutDown0bj (black, blue)
9. PickUpObj (yellow)
10. PutDownObj (yellow, black)
DONE. Should now have a "yellow on black on blue" tower.
"""

    PLAN4 = """
TASK: put one of the traffic cones on top of the dumpster
1. PickUpObj(traffic cone #2)
2. GoToObj(dumpster)
3. PutDown0bj(traffic cone #2, dumpster)
DONE. One of the traffic cones should be on top of the dumpster
"""

    PLAN5 = """
TASK: place the red mug in the fridge
1. PickUpObj(red mug)
2. OpenObj(fridge)
3. PutDown0bj (red mug, fridge)
DONE. red mug should be in the fridge
"""
    TEMPPLAN = """1. PickupObject("mug")
2. OpenObject("fridge")
3. PutObjectIn("mug", "fridge")"""

    HU_PLAN = TEMPPLAN
    # for survey:
    # /Users/charlie/Projects/Holodeck/hippo/sampled_scenes/realscenes/i_box2_stacked_wide/TRELLIS-yes-mask_False-aspect weighted-tight-new-centroid/in_order_0
    # /Users/charlie/Projects/Holodeck/hippo/sampled_scenes/realscenes/i_mug_kitchen_surely4/TRELLIS-yes-mask_True-aspect weighted/in_order_0
    # /Users/charlie/Projects/Holodeck/hippo/sampled_scenes/realscenes/i_mug_kitchen_surely4/TRELLIS-yes-mask_True-aspect weighted/in_order_0

    main(scene_name="/Users/charlie/Projects/Holodeck/hippo/sampled_scenes/realscenes/i_box2_stacked_wide/TRELLIS-yes-mask_False-aspect weighted-tight-new-centroid/in_order_0",  # FloorPlan19_physics ## room
         gridSize=0.25, rotateStepDegrees=15,  ## agent step len and rotate degree
         BEV=False,  ## Bird's-eye view or top view(slope)
         slope_degree=60,  ## top view(slope)'s initial rotate degree
         down_angle=65,  ## top view(slope)'s pitch angle, should be 0-90, 90 equal to Bird's-eye view
         use_procthor=False,  ## use procthor room, True: select room from procthor train set, need dataset dir
         procthor_scene_file="",  ## procthor train set dir
         procthor_scene_num=100,  ## select scene from procthor train set
         is_rotate=True,  ## top_view rotate?
         rotate_per_frame=6,  ## top_view rotate degree
         generate_video=True,  ## use frames generate video
         generate_gif=True,  ## use frames generate gif
         HU_PLAN=HU_PLAN,
         USE_ALTERED_FIRST_PERSON=True,
         AUGMENT_FIRST_PERSON=False
         )