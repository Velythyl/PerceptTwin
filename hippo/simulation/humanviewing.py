import dataclasses
import json
import os
import re
import sys
from typing import Union, List

import cv2

import numpy as np
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from langchain.llms import Anyscale
from tqdm import tqdm

from ai2holodeck.constants import THOR_COMMIT_ID, OBJATHOR_ASSETS_DIR
from hippo.simulation.runtimeobjects import RuntimeObjectContainer
from hippo.simulation.semanticverifllm.llm_semantic_verification import CorrectFinalState
from hippo.simulation.skillsandconditions.conditions import Condlist
from hippo.reconstruction.scenedata import HippoObject, dict2xyztuple
from hippo.utils.file_utils import get_tmp_folder

import os

import ai2thor.controller
from hippo.simulation.ai2thor_metadata_reader import get_object_list_from_controller, get_robot_inventory
from hippo.simulation.skill_simulator import Simulator

from hippo.simulation.spatialutils.filter_positions import build_grid_graph, filter_reachable_positions

from ai2holodeck.constants import THOR_COMMIT_ID
from hippo.ai2thor_hippo_controller import get_hippo_controller_OLDNOW, get_hippo_controller
from hippo.utils.file_utils import get_tmp_folder
@dataclasses.dataclass
class HumanViewing:
    c: Controller
    r: RuntimeObjectContainer
    s: Union[Simulator, None]
    plan: list[str] = dataclasses.field(default_factory=lambda : [])
    current_action_idx: int = 0   # 0 = nothing yet, len(plan)+1 = all done
    use_altered_first_person_frame_instead: bool = False

    def set_plan(self, plan):
        if isinstance(plan, str):
            plan = list(filter(lambda x: len(x) > 0, map(str.strip, plan.split("\n"))))
        self.plan = plan
        return plan

    def incr_action_idx(self):
        self.current_action_idx = self.current_action_idx + 1

    def get_latest_robot_frame(self):
        first_view_frame = self.c.last_event.frame
        return first_view_frame

    def get_latest_topdown_frame(self):
        #self.c.step(action="ToggleMapView")
        frame = self.c.last_event.third_party_camera_frames[0]
        #self.c.step(action="ToggleMapView")
        return frame

    def get_latest_altered_robot_frame(self):
        return self.c.last_event.third_party_camera_frames[1]

    def get_augmented_robot_frame_OLD(self, frame, message=None, held_item=None, hud_scale=2):
        import cv2, re
        if held_item is None:
            inventory = get_robot_inventory(self.c, 0)
            assert len(inventory) <= 1
            if len(inventory) == 0:
                held_item = None
            else:
                raw_item = inventory[0]
                held_item = re.match(r"^([^-]+)", raw_item).group(1)

        if message is None:
            if self.s is not None:
                message_queue = self.s.exception_queue
                if len(message_queue) <= 0:
                    message = None
                    if self.s.currently_thinking:
                        message = "Judge currently thinking..."
                else:
                    exceptionqueue = self.s.exception_queue[-1].args[0]
                    if not isinstance(exceptionqueue, List):
                        exceptionqueue = [exceptionqueue]
                    message = Condlist(exceptionqueue).error_message()
                    message = "FAILURE: " + " AND ".join(message)
                    print(f"Got system message: {message}")
                    #self.s.exception_queue.pop()
        if message is None or message == "":
            message = "[None yet]"

        hud_frame = frame.copy()
        h, w = hud_frame.shape[:2]

        def draw_box(img, top_left, bottom_right, color=(0, 0, 0), alpha=0.5):
            overlay = img.copy()
            cv2.rectangle(overlay, top_left, bottom_right, color, -1)
            return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        def wrap_text(text, font, font_scale, thickness, max_width):
            """Split text into lines so each fits max_width, breaking long words if needed"""
            words = text.split(" ")
            lines, current = [], ""

            def measure_width(txt):
                (tw, _), _ = cv2.getTextSize(txt, font, font_scale, thickness)
                return tw

            for word in words:
                test_line = word if current == "" else current + " " + word
                if measure_width(test_line) <= max_width:
                    current = test_line
                else:
                    if current:  # push current line
                        lines.append(current)
                    # Now handle the long word (break into chunks if needed)
                    if measure_width(word) <= max_width:
                        current = word
                    else:
                        chunk = ""
                        for ch in word:
                            if measure_width(chunk + ch) <= max_width:
                                chunk += ch
                            else:
                                lines.append(chunk)
                                chunk = ch
                        current = chunk  # remainder stays in current
            if current:
                lines.append(current)
            return lines

        pad = int(20 * hud_scale)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 * hud_scale
        title_font_scale = 0.5 * hud_scale
        thickness = max(1, int(1 * hud_scale))

        # Calculate section widths (1/3 of the viewer each)
        section_width = w // 3
        inner_pad = pad // 2  # Padding inside each section

        # ---------------- Plan Display (Left section) ----------------
        plan_x = inner_pad
        plan_y = inner_pad
        title_plan = "Plan:"
        title_plan_size = cv2.getTextSize(title_plan, font, title_font_scale, thickness)[0]

        plan_box_width = section_width - 2 * inner_pad

        # Wrap each step individually
        wrapped_plan_steps = []
        for idx, step in enumerate(self.plan or ["[No plan yet]"], start=1):
            prefix = f""
            (prefix_width, _), _ = cv2.getTextSize(prefix, font, font_scale, thickness)

            # Wrap with prefix width reserved on the first line
            lines = wrap_text(step, font, font_scale, thickness,
                              plan_box_width - inner_pad * 2 - prefix_width)

            if lines:
                lines[0] = prefix + lines[0]
            else:
                lines = [prefix]

            wrapped_plan_steps.extend(lines)

        plan_box_height = title_plan_size[1] + (len(wrapped_plan_steps) * int(20 * hud_scale)) + 2 * inner_pad

        plan_box_tl = (plan_x, plan_y + title_plan_size[1] + inner_pad)
        plan_box_br = (plan_box_tl[0] + plan_box_width, plan_box_tl[1] + plan_box_height)
        hud_frame = draw_box(hud_frame, plan_box_tl, plan_box_br, (50, 50, 0), 0.7)

        cv2.putText(hud_frame, title_plan, (plan_x, plan_y + title_plan_size[1]),
                    font, title_font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        line_y = plan_box_tl[1] + inner_pad + title_plan_size[1]
        step_idx = None
        for line in wrapped_plan_steps:
            match = re.match(r"^(\d+)\.", line)
            if match:
                step_idx = int(match.group(1))

            color = (180, 180, 180)
            if step_idx is not None:
                if self.current_action_idx == step_idx:
                    color = (0, 255, 255)
                elif self.current_action_idx > step_idx:
                    color = (255, 140, 0)
            else:
                color = (255, 140, 0)

            cv2.putText(hud_frame, line,
                        (plan_x + inner_pad, line_y),
                        font, font_scale, color, thickness, cv2.LINE_AA)
            line_y += int(20 * hud_scale)

            # Advance step index only when this line is the first of a step
            #if line.startswith(f"{step_idx}. "):
            #    step_idx += 1

        # ---------------- System Message (Middle section) ----------------
        sys_x = section_width + inner_pad
        sys_y = inner_pad
        title_msg = "System message:"
        title_size = cv2.getTextSize(title_msg, font, title_font_scale, thickness)[0]

        msg_box_width = section_width - 2 * inner_pad
        message_lines = wrap_text(message, font, font_scale, thickness, msg_box_width - inner_pad * 2)
        msg_box_height = title_size[1] + (len(message_lines) * int(20 * hud_scale)) + 3 * inner_pad

        msg_box_tl = (sys_x, sys_y + title_size[1] + inner_pad)
        msg_box_br = (msg_box_tl[0] + msg_box_width, msg_box_tl[1] + msg_box_height)
        hud_frame = draw_box(hud_frame, msg_box_tl, msg_box_br, (0, 0, 50), 0.7)

        cv2.putText(hud_frame, title_msg, (sys_x, sys_y + title_size[1]),
                    font, title_font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        line_y = msg_box_tl[1] + inner_pad + title_size[1]
        for line in message_lines:
            cv2.putText(hud_frame, line,
                        (msg_box_tl[0] + inner_pad, line_y),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            line_y += int(20 * hud_scale)

        # ---------------- Held Item (Right section) ----------------
        item_x = 2 * section_width + inner_pad
        item_y = inner_pad
        title_item = "Held item:"
        item_text = f"[{held_item}]" if held_item else "[Empty Gripper]"
        title_item_size = cv2.getTextSize(title_item, font, title_font_scale, thickness)[0]

        held_object_image = self.c.get_segmented_held_object()
        if held_object_image is not None:
            held_object_image = cv2.cvtColor(held_object_image, cv2.COLOR_RGBA2BGRA)

        # Calculate the box dimensions
        item_box_width = section_width - 2 * inner_pad

        # If we have an image, adjust the box height to accommodate it
        if held_object_image is not None:
            # Calculate maximum dimensions for the image (80% of box width)
            max_img_width = int(item_box_width * 0.8)
            max_img_height = max_img_width  # Keep it square

            # Get original dimensions
            img_h, img_w = held_object_image.shape[:2]

            # Calculate scaling factor
            scale = min(max_img_width / img_w, max_img_height / img_h)

            # Resize the image
            new_width = int(img_w * scale)
            new_height = int(img_h * scale)
            resized_image = cv2.resize(held_object_image, (new_width, new_height))

            # Calculate text height needed
            text_height = title_item_size[1] + (1 * int(20 * hud_scale)) + 2 * inner_pad

            # Total box height = text height + image height + padding
            item_box_height = text_height + new_height + inner_pad

            # Create the box
            item_box_tl = (item_x, item_y + title_item_size[1] + inner_pad)
            item_box_br = (item_box_tl[0] + item_box_width, item_box_tl[1] + item_box_height)
            hud_frame = draw_box(hud_frame, item_box_tl, item_box_br, (50, 0, 0), 0.7)

            # Draw title and text
            cv2.putText(hud_frame, title_item, (item_x, item_y + title_item_size[1]),
                        font, title_font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            line_y = item_box_tl[1] + inner_pad + title_item_size[1]
            cv2.putText(hud_frame, item_text,
                        (item_box_tl[0] + inner_pad, line_y),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Calculate image position (centered horizontally)
            img_x = item_box_tl[0] + (item_box_width - new_width) // 2
            img_y = line_y + int(20 * hud_scale) + inner_pad

            # Overlay the image on the HUD frame
            tmp = np.ones((hud_frame.shape[0], hud_frame.shape[1], 4), dtype=hud_frame.dtype)
            tmp[:,:,:3] = hud_frame
            hud_frame = tmp
            hud_frame[img_y:img_y + new_height, img_x:img_x + new_width] = resized_image
            hud_frame = hud_frame[:,:,:3]

        else:
            # No image - just show text as before
            item_lines = wrap_text(item_text, font, font_scale, thickness, item_box_width - inner_pad * 2)
            item_box_height = title_item_size[1] + (len(item_lines) * int(20 * hud_scale)) + 3 * inner_pad

            item_box_tl = (item_x, item_y + title_item_size[1] + inner_pad)
            item_box_br = (item_box_tl[0] + item_box_width, item_box_tl[1] + item_box_height)
            hud_frame = draw_box(hud_frame, item_box_tl, item_box_br, (50, 0, 0), 0.7)

            cv2.putText(hud_frame, title_item, (item_x, item_y + title_item_size[1]),
                        font, title_font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            line_y = item_box_tl[1] + inner_pad + title_item_size[1]
            for line in item_lines:
                cv2.putText(hud_frame, line,
                            (item_box_tl[0] + inner_pad, line_y),
                            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                line_y += int(20 * hud_scale)

        return hud_frame

    def get_augmented_robot_frame(self, frame, message=None, held_item=None, hud_scale=2):
        import cv2, re
        if held_item is None:
            inventory = get_robot_inventory(self.c, 0)
            assert len(inventory) <= 1
            if len(inventory) == 0:
                held_item = None
            else:
                raw_item = inventory[0]
                held_item = re.match(r"^([^-]+)", raw_item).group(1)

        if message is None:
            if self.s is not None:
                message_queue = self.s.exception_queue
                if len(message_queue) <= 0:
                    message = None
                    if self.s.currently_thinking:
                        message = "Judge currently thinking..."
                else:
                    if isinstance(self.s.exception_queue[-1], CorrectFinalState):
                        message = f"Correct Plan: {self.s.exception_queue[-1].reason}"
                    else:
                        exceptionqueue = self.s.exception_queue[-1].args[0]
                        if not isinstance(exceptionqueue, List):
                            exceptionqueue = [exceptionqueue]
                        message = Condlist(exceptionqueue).error_message()
                        message = "FAILURE: " + " AND ".join(message)
                    print(f"Got system message: {message}")
                    # self.s.exception_queue.pop()
        if message is None or message == "":
            message = "[None yet]"

        hud_frame = frame.copy()
        h, w = hud_frame.shape[:2]

        def draw_box(img, top_left, bottom_right, color=(0, 0, 0), alpha=0.5):
            overlay = img.copy()
            cv2.rectangle(overlay, top_left, bottom_right, color, -1)
            return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        def wrap_text(text, font, font_scale, thickness, max_width):
            """Split text into lines so each fits max_width, breaking long words if needed"""
            words = text.split(" ")
            lines, current = [], ""

            def measure_width(txt):
                (tw, _), _ = cv2.getTextSize(txt, font, font_scale, thickness)
                return tw

            for word in words:
                test_line = word if current == "" else current + " " + word
                if measure_width(test_line) <= max_width:
                    current = test_line
                else:
                    if current:  # push current line
                        lines.append(current)
                    # Now handle the long word (break into chunks if needed)
                    if measure_width(word) <= max_width:
                        current = word
                    else:
                        chunk = ""
                        for ch in word:
                            if measure_width(chunk + ch) <= max_width:
                                chunk += ch
                            else:
                                lines.append(chunk)
                                chunk = ch
                        current = chunk  # remainder stays in current
            if current:
                lines.append(current)
            return lines

        pad = int(20 * hud_scale)
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_font_scale = 0.5 * hud_scale
        title_font_scale = 0.5 * hud_scale
        thickness = max(1, int(1 * hud_scale))

        # Calculate section widths (1/3 of the viewer each)
        section_width = w // 3
        inner_pad = pad // 2  # Padding inside each section

        # ---------------- Plan Display (Left section) ----------------
        plan_x = inner_pad
        plan_y = inner_pad
        title_plan = "Plan:"
        title_plan_size = cv2.getTextSize(title_plan, font, title_font_scale, thickness)[0]

        plan_box_width = section_width - 2 * inner_pad
        max_plan_height = h - 2.5 * inner_pad  # Maximum allowed height for the plan section

        # Function to calculate plan height with given font scale
        def calculate_plan_height(font_scale):
            wrapped_plan_steps = []
            for idx, step in enumerate(self.plan or ["[No plan yet]"], start=1):
                prefix = f""
                (prefix_width, _), _ = cv2.getTextSize(prefix, font, font_scale, thickness)

                # Wrap with prefix width reserved on the first line
                lines = wrap_text(step, font, font_scale, thickness,
                                  plan_box_width - inner_pad * 2 - prefix_width)

                if lines:
                    lines[0] = prefix + lines[0]
                else:
                    lines = [prefix]

                wrapped_plan_steps.extend(lines)

            line_height = int(20 * hud_scale * (font_scale / base_font_scale))
            return title_plan_size[1] + (len(wrapped_plan_steps) * line_height) + 2 * inner_pad

        # Adjust font scale if plan is too long
        font_scale = base_font_scale
        min_font_scale = 0.1 * hud_scale  # Minimum font scale to prevent text from becoming unreadable

        plan_height = calculate_plan_height(font_scale)
        while plan_height > max_plan_height and font_scale > min_font_scale:
            font_scale -= 0.025
            plan_height = calculate_plan_height(font_scale)

        # Now generate the wrapped plan with the adjusted font scale
        wrapped_plan_steps = []
        for idx, step in enumerate(self.plan or ["[No plan yet]"], start=1):
            prefix = f""
            (prefix_width, _), _ = cv2.getTextSize(prefix, font, font_scale, thickness)

            # Wrap with prefix width reserved on the first line
            lines = wrap_text(step, font, font_scale, thickness,
                              plan_box_width - inner_pad * 2 - prefix_width)

            if lines:
                lines[0] = prefix + lines[0]
            else:
                lines = [prefix]

            wrapped_plan_steps.extend(lines)

        line_height = int(20 * hud_scale * (font_scale / base_font_scale))
        plan_box_height = title_plan_size[1] + (len(wrapped_plan_steps) * line_height) + 2 * inner_pad

        plan_box_tl = (plan_x, plan_y + title_plan_size[1] + inner_pad)
        plan_box_br = (plan_box_tl[0] + plan_box_width, plan_box_tl[1] + plan_box_height)
        hud_frame = draw_box(hud_frame, plan_box_tl, plan_box_br, (50, 50, 0), 0.7)

        cv2.putText(hud_frame, title_plan, (plan_x, plan_y + title_plan_size[1]),
                    font, title_font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        line_y = plan_box_tl[1] + inner_pad + title_plan_size[1]
        step_idx = None
        for line in wrapped_plan_steps:
            match = re.match(r"^(\d+)\.", line)
            if match:
                step_idx = int(match.group(1))

            color = (180, 180, 180)
            if step_idx is not None:
                if self.current_action_idx == step_idx:
                    color = (0, 255, 255)
                elif self.current_action_idx > step_idx:
                    color = (255, 140, 0)
            else:
                color = (255, 140, 0)

            cv2.putText(hud_frame, line,
                        (plan_x + inner_pad, line_y),
                        font, font_scale, color, thickness, cv2.LINE_AA)
            line_y += line_height

        # ---------------- System Message (Middle section) ----------------
        sys_x = section_width + inner_pad
        sys_y = inner_pad
        title_msg = "System message:"
        title_size = cv2.getTextSize(title_msg, font, title_font_scale, thickness)[0]

        msg_box_width = section_width - 2 * inner_pad
        message_lines = wrap_text(message, font, base_font_scale, thickness, msg_box_width - inner_pad * 2)
        msg_box_height = title_size[1] + (len(message_lines) * int(20 * hud_scale)) + 3 * inner_pad

        msg_box_tl = (sys_x, sys_y + title_size[1] + inner_pad)
        msg_box_br = (msg_box_tl[0] + msg_box_width, msg_box_tl[1] + msg_box_height)
        hud_frame = draw_box(hud_frame, msg_box_tl, msg_box_br, (0, 0, 50), 0.7)

        cv2.putText(hud_frame, title_msg, (sys_x, sys_y + title_size[1]),
                    font, title_font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        line_y = msg_box_tl[1] + inner_pad + title_size[1]
        for line in message_lines:
            cv2.putText(hud_frame, line,
                        (msg_box_tl[0] + inner_pad, line_y),
                        font, base_font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            line_y += int(20 * hud_scale)

        # ---------------- Held Item (Right section) ----------------
        item_x = 2 * section_width + inner_pad
        item_y = inner_pad
        title_item = "Held item:"
        item_text = f"[{held_item}]" if held_item else "[Empty Gripper]"
        title_item_size = cv2.getTextSize(title_item, font, title_font_scale, thickness)[0]

        held_object_image = self.c.get_segmented_held_object()
        if held_object_image is not None:
            held_object_image = cv2.cvtColor(held_object_image, cv2.COLOR_RGBA2BGRA)

        # Calculate the box dimensions
        item_box_width = section_width - 2 * inner_pad

        # If we have an image, adjust the box height to accommodate it
        if held_object_image is not None:
            # Calculate maximum dimensions for the image (80% of box width)
            max_img_width = int(item_box_width * 0.8)
            max_img_height = max_img_width  # Keep it square

            # Get original dimensions
            img_h, img_w = held_object_image.shape[:2]

            # Calculate scaling factor
            scale = min(max_img_width / img_w, max_img_height / img_h)

            # Resize the image
            new_width = int(img_w * scale)
            new_height = int(img_h * scale)
            resized_image = cv2.resize(held_object_image, (new_width, new_height))

            # Calculate text height needed
            text_height = title_item_size[1] + (1 * int(20 * hud_scale)) + 2 * inner_pad

            # Total box height = text height + image height + padding
            item_box_height = text_height + new_height + inner_pad

            # Create the box
            item_box_tl = (item_x, item_y + title_item_size[1] + inner_pad)
            item_box_br = (item_box_tl[0] + item_box_width, item_box_tl[1] + item_box_height)
            hud_frame = draw_box(hud_frame, item_box_tl, item_box_br, (50, 0, 0), 0.7)

            # Draw title and text
            cv2.putText(hud_frame, title_item, (item_x, item_y + title_item_size[1]),
                        font, title_font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            line_y = item_box_tl[1] + inner_pad + title_item_size[1]
            cv2.putText(hud_frame, item_text,
                        (item_box_tl[0] + inner_pad, line_y),
                        font, base_font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Calculate image position (centered horizontally)
            img_x = item_box_tl[0] + (item_box_width - new_width) // 2
            img_y = line_y + int(20 * hud_scale) + inner_pad

            # Overlay the image on the HUD frame
            tmp = np.ones((hud_frame.shape[0], hud_frame.shape[1], 4), dtype=hud_frame.dtype)
            tmp[:, :, :3] = hud_frame
            hud_frame = tmp
            hud_frame[img_y:img_y + new_height, img_x:img_x + new_width] = resized_image
            hud_frame = hud_frame[:, :, :3]

        else:
            # No image - just show text as before
            item_lines = wrap_text(item_text, font, base_font_scale, thickness, item_box_width - inner_pad * 2)
            item_box_height = title_item_size[1] + (len(item_lines) * int(20 * hud_scale)) + 3 * inner_pad

            item_box_tl = (item_x, item_y + title_item_size[1] + inner_pad)
            item_box_br = (item_box_tl[0] + item_box_width, item_box_tl[1] + item_box_height)
            hud_frame = draw_box(hud_frame, item_box_tl, item_box_br, (50, 0, 0), 0.7)

            cv2.putText(hud_frame, title_item, (item_x, item_y + title_item_size[1]),
                        font, title_font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            line_y = item_box_tl[1] + inner_pad + title_item_size[1]
            for line in item_lines:
                cv2.putText(hud_frame, line,
                            (item_box_tl[0] + inner_pad, line_y),
                            font, base_font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                line_y += int(20 * hud_scale)

        return hud_frame

    def display_frame(self, frame):
        import cv2
        if frame is None:
            if self.use_altered_first_person_frame_instead:
                frame = self.get_latest_altered_robot_frame()
            else:
                frame = self.get_latest_robot_frame()
        cv2.imshow("first_view", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def display_augmented_frame(self):
        if self.use_altered_first_person_frame_instead:
            frame = self.get_latest_altered_robot_frame()
        else:
            frame = self.get_latest_robot_frame()

        return self.display_frame(self.get_augmented_robot_frame(frame))
