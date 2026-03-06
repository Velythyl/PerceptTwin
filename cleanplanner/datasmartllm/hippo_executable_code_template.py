import os
import subprocess # noqa
import sys # noqa
import threading  # noqa
import time  # noqa
from glob import glob # noqa
import json # noqa

import cv2
import numpy as np

from hippo.ai2thor_hippo_controller import get_sim
from hippo.simulation.singlefilelog import set_filepath
from llmqueries.llm import set_api_key

import traceback
try:
    vid_frames = []
    FAILURE_STATE = False


    def thread_exception_handler(args):
        global FAILURE_STATE
        if str(args.exc_value) == "Done!":
            pass
        else:
            FAILURE_STATE = True
        print("Condition failure caught by thread handler")
        assert len(vid_frames) > 0
        save_video(vid_frames)
        os._exit(1)

    threading.excepthook = thread_exception_handler
    def save_video(frames, fps=40):
        global KILL_CAPTURE_THREAD
        global vid_frames
        KILL_CAPTURE_THREAD = True
        time.sleep(1)
        print("Saving video...")
        if not frames:
            raise ValueError("Frame list is empty!")
        
        #if sys.platform == "darwin":
        #    fps = 40
        #else:
        #    fps = 100 # really slow on cluster for some reason

        output_path = f"{executable_output_dir}/output.mp4"  # noqa

        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

        # Get frame size from the first frame
        height, width = frames[0].shape[:2]

        # Try a more browser-friendly codec (H.264 in mp4)
        # Fallback to mp4v if H.264 isn't available in your OpenCV build
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # or try "H264"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"H264")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print("⚠️ H.264 codec not available, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            out.write(frame)

        out.release()
        vid_frames = []
        print(f"Video saved at {output_path}")
        if FAILURE_STATE:
            return os._exit(1)
        else:
            return os._exit(0)

    >>> FILL IN SETUP CODE HERE <<< # noqa

    no_robot = len(robots)
    print(scene_name)

    set_api_key(api_key_path)
    set_filepath(tmp_hippo_log_dir)
    simulator = get_sim(scene_name, GRID_SIZE=0.5, width=1250, height=1250)
    simulator.set_task_description(plan_pretty_print)
    simulator.humanviewing.set_plan(plan_pretty_print)
    simulator.kill_sim_on_condition_failure = False # adds a 10 sec delay
    simulator.raise_exception_on_condition_failure = 10
    simulator.judge_llm = JUDGE_LLM_FROM_CFG
    print("Using judge LLM:", simulator.judge_llm)
    KILL_CAPTURE_THREAD = False
    OLD_NUM_ACTIONS = len(simulator.done_actions)
    simulator.humanviewing.incr_action_idx()
    def capture_frames():
        global OLD_NUM_ACTIONS
        while KILL_CAPTURE_THREAD is False:
            if len(simulator.done_actions) != OLD_NUM_ACTIONS:
                OLD_NUM_ACTIONS = len(simulator.done_actions)
                simulator.humanviewing.incr_action_idx()
            #print("Capturing frames...")
            if sys.platform == "darwin":
                time.sleep(0.1)
            else:
                time.sleep(0.5)
            vid_frames.append(simulator.humanviewing.get_augmented_robot_frame(simulator.humanviewing.get_latest_robot_frame()))
    frame_thread = threading.Thread(target=capture_frames)
    frame_thread.daemon = True
    frame_thread.start()

    >>> FILL IN PLAN CODE HERE <<<  # noqa

    simulator.Done()
    threading.Lock().acquire()  # lock self to handle exit in the thread exception handler

except Exception as e:
    print("Exception in main thread:", e)
    traceback.print_exc()
    try:
        simulator.controller.stop()
    except:
        pass
    #if len(vid_frames) > 0:
    #    save_video(vid_frames)
    os._exit(1)
