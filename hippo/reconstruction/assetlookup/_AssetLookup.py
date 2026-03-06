import copy

from hippo.reconstruction.assetlookup.assetalign import align
from hippo.reconstruction.scenedata import HippoObject
import itertools
from typing import Any, Callable

import numpy as np
import open_clip
from sentence_transformers import SentenceTransformer

from ai2holodeck.generation.holodeck import confirm_paths_exist
from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever
from ai2holodeck.generation.rooms import FloorPlanGenerator
from ai2holodeck.generation.utils import get_bbox_dims, get_annotations
from ai2holodeck.generation.walls import WallGenerator
from hippo.reconstruction.scenedata import HippoRoomPlan, HippoObject


class AssetLookup:
    def __init__(self):
        self.wall_generator = WallGenerator(None)

        self.floor_generator = FloorPlanGenerator(
            self.clip_model, self.clip_preprocess, self.clip_tokenizer, None
        )

    def lookup_assets(self, obj: HippoObject):
        raise NotImplementedError()

    def generate_rooms(self, cfg, scene, plan: str=None, hipporoom: HippoRoomPlan=None):
        if not plan:
            assert hipporoom is not None
            plan = hipporoom.asholodeckstr()
        if not hipporoom:
            assert plan is not None

        rooms = self.floor_generator.get_plan("programmatic floor query", plan)
        scene["rooms"] = rooms

        scene["wall_height"] = cfg.scene.wall_height
        wall_height, walls = self.wall_generator.generate_walls(scene)
        scene["wall_height"] = wall_height
        scene["walls"] = walls

        def apply_to_positions(data: Any, func: Callable[[float, float, float], dict]):
            """
            Recursively applies `func` to every {'position': {'x': float, 'y': float, 'z': float}} sub-dictionary.

            :param data: The input data (can be dict, list, or other types).
            :param func: A function that takes (x, y, z) and returns a modified dict {'x': ..., 'y': ..., 'z': ...}.
            :return: The modified data structure.
            """
            if isinstance(data, dict):
                if "position" in data and isinstance(data["position"], dict):
                    pos = data["position"]
                    if all(k in pos and isinstance(pos[k], (int, float)) for k in ["x", "y", "z"]):
                        data["position"] = func(pos["x"], pos["y"], pos["z"])

                for key in data:
                    data[key] = apply_to_positions(data[key], func)

            elif isinstance(data, list):
                for i in range(len(data)):
                    data[i] = apply_to_positions(data[i], func)

            return data

        metadata = scene["metadata"]
        maxxcoord = scene["rooms"][0]["vertices"][2][0]
        maxzcoord = scene["rooms"][0]["vertices"][2][1]
        apply_to_positions(metadata, lambda x, y, z: {"x": 0.5, "y": y, "z": 0.5})

        return scene