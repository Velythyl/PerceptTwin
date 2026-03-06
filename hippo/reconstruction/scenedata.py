import json
import math
import os
import shutil
import uuid
from dataclasses import field, dataclass
from typing import Tuple, List, Union, Dict, Any

import numpy as np
import open3d as o3d
from typing_extensions import Self

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR
from hippo.reconstruction.assetlookup.assetIsPlane import ask_llm_if_plane
from hippo.utils.dict_utils import recursive_map
from hippo.utils.o3d_np_v3v import v3v
from hippo.utils.selfdataclass import SelfDataclass
from hippo.utils.spatial_utils import transform_ai2thor_object


@dataclass
class _Hippo(SelfDataclass):
    def asdict(self):
        dico = super().asdict()
        dico = recursive_map(dico, lambda x: x if not isinstance(x, np.ndarray) else x.tolist())
        return dico

@dataclass
class HippoRoomPlan(_Hippo):
    id: str
    floor_type: str = "light grey drywall, smooth"
    wall_type: str = "light grey drywall, smooth"
    coords: List[Tuple[float,float]] =  ((0, 0), (0, 6), (7, 6), (7, 0))
    center: Tuple[float,float] = (3.5, 3)
    size: Tuple[float,float] = (7, 6)
    wall_height: float = 3

    def asprompt(self):
        # todo
        return "room name (must be unique) | floor type | wall type | coordinates of the foor corners of the room"

    def asholodeckstr(self):
        return f"{self.id} | {self.floor_type} | {self.wall_type} | {self.coords}"

def xyztuple2dict(tup):
    return {'x': tup[0], 'y': tup[1], 'z': tup[2]}

def dict2xyztuple(dic):
    return (dic['x'], dic['y'], dic['z'])

def xyztuple_precision(tup):
    return (round(tup[0], 3), round(tup[1],3), round(tup[2],3))

from diskcache import FanoutCache, Cache
CACHEPATH = "/".join(__file__.split("/")[:-1]) + "/diskcache"
cache = Cache(CACHEPATH)

@dataclass
class HippoObject(_Hippo):
    object_name: str
    object_description: str
    roomId: Union[HippoRoomPlan, str]

    _is_planar_according_to_llm: bool = False

    _position: Tuple[float, float, float] = (0, 0, 0)
    _rotation: Tuple[float, float, float] = (0, 0, 0)

    _is_objaverse_asset: Tuple[bool] = tuple()
    _found_assetIds: Tuple[str] = tuple()
    _found_sizes: Tuple[Tuple[float,float,float]] = tuple()
    _found_scores: Tuple[float] = tuple()

    _clip_features: np.ndarray = None
    _desired_size: Tuple[float, float, float] = None

    _id: str = field(default_factory=lambda: str(uuid.uuid4().hex)[:4])
    _assetMetadata: Dict[str,Any] = field(default_factory=dict)

    _skill_metadata: Tuple[str] = ("can be turned on/off", "can be picked up", "objects can be put down on this", "can be opened and closed", "can be sliced", "can be broken")

    _cg_paths: Dict[str, str] = field(default_factory=dict)
    _cg_pcd_points: Tuple[float] = tuple()
    _cg_pcd_colours: Tuple = tuple()

    _assets_dir: Tuple[str] = tuple()

    def set_pcd_(self, pcd: Union[o3d.geometry.PointCloud, np.array], pcd_colours: np.array=None):
        if isinstance(pcd, o3d.geometry.PointCloud):
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
        else:
            points = pcd
            colors = pcd_colours

        points = tuple(points.tolist())
        colours = tuple(colors.tolist())

        return self.replace(_cg_pcd_points=points, _cg_pcd_colours=colours)

    def get_pcd(self):
        assert len(self._cg_pcd_points) > 0
        pcd = o3d.geometry.PointCloud()
        pcd.points = v3v(np.array(self._cg_pcd_points))
        pcd.colors = v3v(np.array(self._cg_pcd_colours))
        return pcd

    def add_asset_info_(self, found_assets, found_sizes, found_scores, assets_dir):
        # fixme we used to compute this here, is it even relevant ?
        #_is_objaverse_asset = []
        #for fa in found_assets:
        #    try:
        #        from ai2thor.util.runtime_assets import load_existing_thor_asset_file
        #        obj = load_existing_thor_asset_file(OBJATHOR_ASSETS_DIR, f"{fa}/{fa}")
        #        IS_OBJAVERSE_OBJECT = True
        #    except:
        #        IS_OBJAVERSE_OBJECT = False
        #    _is_objaverse_asset.append(IS_OBJAVERSE_OBJECT)

        #
        # I think its fine because we did use_thor_objects=False for ObjectRetriever, so objs should always be Objaverse
        #

        import llmqueries


        _is_objaverse_asset = [True] * len(found_assets)
        return self.replace(
            _is_planar_according_to_llm=ask_llm_if_plane(self.object_name),
            _is_objaverse_asset=tuple(_is_objaverse_asset),
            _found_assetIds=found_assets,
            _found_sizes=found_sizes,
            _found_scores=found_scores,
            _assets_dir=assets_dir,
        )

    def set_skill_metadata(self, skill_metadata):
        skill_metadata2 = tuple()
        for s in skill_metadata:
            if "put down on this" in s:
                continue
            skill_metadata2 = (*skill_metadata2, s)
        skill_metadata = (*skill_metadata, "objects can be put down on this")
        print(f"Object {self.object_name} has the following skills: {skill_metadata}")

        return self.replace(_skill_metadata=skill_metadata)

    @property
    def _concrete_assetIds(self):
        ret = []
        for is_objaverse_asset, og_id, concrete_id in zip(self._is_objaverse_asset, self._found_assetIds, self._concrete_Ids):
            if is_objaverse_asset:
                ret.append(concrete_id)
            else:
                ret.append(og_id)
        return ret

        #return [f"{self.object_name_id}-{id[:4]}" for id in self._found_assetIds]

    @property
    def _concrete_Ids(self):
        return [f"{self.object_name_id}-{id[:4]}" for id in self._found_assetIds]

    @property
    def _found_size_scaling(self):
        return np.array(self._desired_size) / np.array(self._found_sizes)

    @property
    def object_name_id(self):
        return self.object_name + f"-{self._id}"

    def as_holodeckdict(self):
        assert len(self._concrete_assetIds) >= 1

        asdict = super().asdict()
        if isinstance(asdict["roomId"], HippoRoomPlan):
            asdict["roomId"] = asdict["roomId"].id

        asdict.update(
            {
                "material": None,
                "layer": "Procedural0",
                "kinematic": True,
            }
        )

        asdict["assetId"] = self._concrete_assetIds[0]
        asdict["id"] = self._concrete_Ids[0]
        size = (self._found_sizes[0][0], self._desired_size[1], self._found_sizes[0][2])
        asdict["size"] = size
        asdict["position"] = xyztuple2dict(asdict["_position"])
        asdict["rotation"] = xyztuple2dict(asdict["_rotation"])

        del asdict["_position"]
        del asdict["_rotation"]
        del asdict["_cg_pcd_points"]
        del asdict["_cg_pcd_colours"]
        del asdict["_clip_features"]

        asdict["IS_HIPPO"] = True

        return asdict

    def to_runtimeobject(self):
        from hippo.simulation.runtimeobjects import RuntimeObject
        return RuntimeObject.from_hippoobject(self)

    @classmethod
    def from_holodeckdict(cls, holodeck_dict):
        self = cls.fromdict(holodeck_dict)
        self = self.replace(
            _position=dict2xyztuple(holodeck_dict["position"]),
            _rotation=dict2xyztuple(holodeck_dict["rotation"]),
        )
        return self

    def __len__(self) -> int:
        return len(self._found_assetIds)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.replace(
                _found_assetIds=self._found_assetIds[index.start:index.stop:index.step],
                _found_sizes=self._found_sizes[index.start:index.stop:index.step],
                _found_scores=self._found_scores[index.start:index.stop:index.step],
            )

        return self.replace(
            _found_assetIds=(self._found_assetIds[index],),
            _found_sizes=(self._found_sizes[index],),
            _found_scores=(self._found_scores[index],),
        )

    def concretize(self, cfg, target_directory):
        print(f"Concretizing {self.object_name}")

        all_selves = self

        for self in all_selves:
            assert len(self) == 1

            assetId = self._found_assetIds[0]
            asset_dir = self._assets_dir[0]

            try:
                from ai2thor.util.runtime_assets import load_existing_thor_asset_file
                obj = load_existing_thor_asset_file(asset_dir, f"{assetId}/{assetId}") # load_existing_thor_asset_file(OBJATHOR_ASSETS_DIR, f"{assetId}/{assetId}")
                IS_OBJAVERSE_OBJECT = True
            except:
                IS_OBJAVERSE_OBJECT = False

            if IS_OBJAVERSE_OBJECT:
                scaling = self._found_size_scaling[0]
                concrete_assetId = self._concrete_assetIds[0]

                target_directory_for_this_self = os.path.join(target_directory, concrete_assetId)
                if os.path.exists(target_directory_for_this_self):
                    return
                os.makedirs(target_directory_for_this_self, exist_ok=False)

                # 1. rotate first
                # 2. calculate scaling from the rotated obj pcd
                # 3. rescale obj
                # 4. save obj without actually doing the rotation
                # 5. write rotation into the scene definition


                from hippo.reconstruction.assetlookup.assettranslate import translate
                #obj = translate(obj, self._cg_pcd_points)

                if cfg.assetfitting.rotate:
                    from hippo.reconstruction.assetlookup.assetalign import align, pcd_or_mesh_to_np, \
                        rotate_point_cloud_y_axis, add_scaling_to_transmat
                    euler_rots, transformation_mat_rots = align(pcd_to_align=obj, target_pcd=self._cg_pcd_points, round_rot=cfg.assetfitting.round_rot, is_planar_according_to_llm=self._is_planar_according_to_llm)
                    obj = transform_ai2thor_object(obj, np.array(transformation_mat_rots))

                from hippo.reconstruction.assetlookup.assetscale import scale_object
                obj = scale_object(cfg, obj, self._cg_pcd_points)
                #obj = axis_scale(obj, self._cg_pcd_points)

                new_obj = {}
                for toplevel_k, toplevel_v in obj.items():
                    if isinstance(toplevel_v, str) and assetId in toplevel_v:
                        toplevel_v = toplevel_v.replace(assetId, concrete_assetId)
                    new_obj[toplevel_k] = toplevel_v
                print("Obj name", self.object_name)
                #print(euler_rots[1])
                new_obj["yRotOffset"] = 0 #euler_rots[1]

                save_path = f"{target_directory_for_this_self}/{self._concrete_assetIds[0]}.pkl.gz"
                from ai2thor.util.runtime_assets import save_thor_asset_file
                save_thor_asset_file(new_obj, save_path)

                original_asset_dir = os.path.join(asset_dir, assetId)
                for other_file in os.listdir(original_asset_dir):
                    if not other_file.endswith(f"{self._concrete_assetIds[0]}.pkl.gz"):
                        shutil.copy(os.path.join(original_asset_dir, other_file), os.path.join(target_directory_for_this_self, other_file))


                with open(os.path.join(target_directory_for_this_self, "thor_metadata.json"), "w") as f:
                    json.dump({"assetMetadata": {
                    "primaryProperty": "CanPickup" if "can be picked up" in self._skill_metadata else "Static",
                    "secondaryProperties": [
                        "Receptacle"
                    ] if "objects can be put down on this" in self._skill_metadata else []
                }}, f, indent=4)

            return

def figure_out_scaling(self_pcd, asset, euler_rot_to_align_asset_to_self):
    from hippo.reconstruction.assetlookup.assetalign import pcd_or_mesh_to_np, rotate_point_cloud_y_axis

    unrotated_obj_pcd = pcd_or_mesh_to_np(asset)
    # should this be 2*np.pi - ?
    # or 1- ?
    # or nothing?
    # right now, 2*np.p1 -  with 90 deg round works. Nothing else works though.. Some problem with how scaling interacts with rotations
    rotated_self_pcd = rotate_point_cloud_y_axis(pcd_or_mesh_to_np(self_pcd), - math.radians(euler_rot_to_align_asset_to_self[1]))
    # rotated_obj_pcd = transform_point_cloud(pcd_or_mesh_to_np(obj), transformation_mat_rots)

    from hippo.utils.spatial_utils import pcd_bbox_size
    # obj_pcd_size = np.array(dict2xyztuple(pcd_bbox_size(rotated_obj_pcd)))
    self_pcd_size = np.array(dict2xyztuple(pcd_bbox_size(rotated_self_pcd)))
    obj_pcd_size = np.array(dict2xyztuple(pcd_bbox_size(unrotated_obj_pcd)))

    # if "sideboard" in self.object_name:
    #    scaling = np.array([5, 1, 1])
    # else:
    scaling = np.array(self_pcd_size) / obj_pcd_size
    return scaling


def axis_scale(ai2thor_obj, target_pcd):
    from hippo.utils.spatial_utils import pcd_bbox_size
    from hippo.reconstruction.assetlookup.assetalign import pcd_or_mesh_to_np, add_scaling_to_transmat

    FIXED_TARGET_SIZE = np.array(dict2xyztuple(pcd_bbox_size(target_pcd)))

    obj_pcd_size = np.array(dict2xyztuple(pcd_bbox_size(pcd_or_mesh_to_np(ai2thor_obj))))

    scaling = FIXED_TARGET_SIZE / obj_pcd_size
    print(np.linalg.norm(scaling))
    ai2thor_obj = transform_ai2thor_object(ai2thor_obj, add_scaling_to_transmat(np.eye(4), scaling))
    return ai2thor_obj


def aspect_scale(ai2thor_obj, target_pcd, mode='fit'):
    """
    Scale object while maintaining aspect ratio.

    Args:
        ai2thor_obj: The object to scale
        target_pcd: The target point cloud to match
        mode: 'fit' (scale to fit inside target, default) or
              'fill' (scale to fill target, may exceed in some dimensions)
    """
    from hippo.utils.spatial_utils import pcd_bbox_size
    from hippo.reconstruction.assetlookup.assetalign import pcd_or_mesh_to_np, add_scaling_to_transmat

    target_size = np.array(dict2xyztuple(pcd_bbox_size(target_pcd)))
    obj_size = np.array(dict2xyztuple(pcd_bbox_size(pcd_or_mesh_to_np(ai2thor_obj))))

    scale_factors = target_size / obj_size

    if mode == 'fit':
        uniform_scale = np.min(scale_factors)  # Ensures object fits within target
    elif mode == 'fill':
        uniform_scale = np.max(scale_factors)  # Ensures object fills target
    else:
        raise ValueError("mode must be either 'fit' or 'fill'")

    scaling = np.array([uniform_scale, uniform_scale, uniform_scale])
    ai2thor_obj = transform_ai2thor_object(ai2thor_obj, add_scaling_to_transmat(np.eye(4), scaling))

    return ai2thor_obj