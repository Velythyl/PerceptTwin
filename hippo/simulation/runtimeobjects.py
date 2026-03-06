import copy
import functools
from dataclasses import field, dataclass
from typing import List, Dict, Any

import jax
import numpy as np
from typing_extensions import Self

import jax.numpy as jnp

from hippo.simulation.spatialutils.proximity_spatial_funcs import isOnTop, isInside, isBeside, distance
from hippo.reconstruction.scenedata import dict2xyztuple, HippoObject, _Hippo, xyztuple_precision
from hippo.utils.git_diff import git_diff


def clip_number_string(x):
    return str(round(x, 2))

#from hippo.hippocontainers.skills import *

@dataclass
class RuntimeObject(_Hippo):
    #object_name: str
    #object_description: str
    id: str

    position: jnp.ndarray
    rotation: jnp.ndarray
    size: jnp.ndarray

    skill_portfolio: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    heldBy: str = None

    _force_inside_target: str = None
    _force_contain_target: str = None


    def change_posrotsize(self, position, rotation, size) -> Self:
        if isinstance(position, dict):
            position = dict2xyztuple(position)

        if isinstance(rotation, dict):
            rotation = dict2xyztuple(rotation)

        if isinstance(size, dict):
            size = dict2xyztuple(size)

        position = xyztuple_precision(position)
        rotation = xyztuple_precision(rotation)
        size = xyztuple_precision(size)

        position = jnp.array(position)
        rotation = jnp.array(rotation)
        size = jnp.array(size)
        return self.replace(position=position, rotation=rotation, size=size).cast_precision()

    def cast_precision(self):
        return self.replace(
            position=jnp.round(self.position, 3),
            rotation=jnp.round(self.rotation, 3),
            size=jnp.round(self.size, 3)
        )
    
    def as_llmjson(self):
        dico = self.asdict()

        import numpy as np
        def arr2tup(key):
            arr = dico[key]
            arr = np.array(arr)
            arr = arr.tolist()
            dico[key] = (clip_number_string(float(arr[0])), clip_number_string(float(arr[1])), clip_number_string(float(arr[2])))
        arr2tup("position"); arr2tup("rotation"); arr2tup("size")

        del dico["skill_portfolio"]
        dico.update(self.skill_portfolio.output_dict())

        dico.update(dico["metadata"])
        del dico["metadata"]
        del dico["size"] # todo do we need to keep this ?
        del dico["rotation"] # todo do we need to keep this ?
        ret = {}
        for k, v in dico.items():
            if k.startswith("_"):
                continue
            ret[k] = v
        return ret

    @classmethod
    def from_hippoobject(cls, hippoobject):
        dico = hippoobject.as_holodeckdict()

        #skill_metadata = dico["_skill_metadata"]
        #for k in ["breakable", "toggleable", "sliceable"]:
        #    assert k in skill_metadata

        #for k in ["isBroken", "isToggled", "isSliced"]:
        #    if k not in skill_metadata:
        #        skill_metadata[k] = False

        #dico.update(skill_metadata)
        return cls.fromdict(dico)

    @classmethod
    def fromdict(cls, dico):
        #object_name = dico["object_name"]
        #object_description = dico["object_description"]
        id = dico["id"]
        position = dico["position"]
        rotation = dico["rotation"]
        size = dico["size"]

        if isinstance(position, dict):
            position = dict2xyztuple(position)

        if isinstance(rotation, dict):
            rotation = dict2xyztuple(rotation)

        if isinstance(size, dict):
            size = dict2xyztuple(size)

        position = jnp.array(position)
        rotation = jnp.array(rotation)
        size = jnp.array(size)

        if "_skill_metadata" not in dico:
            raise NotImplementedError()

        from hippo.simulation.skillsandconditions.skills_abstract import SkillPortfolio
        skill_portfolio = SkillPortfolio.create(dico["_skill_metadata"])

        self = cls(
            id=id,
            position=position,
            rotation=rotation,
            size=size,
            heldBy=dico["heldBy"] if "heldBy" in dico else None,
            skill_portfolio=skill_portfolio
        )

        return self.cast_precision()

@dataclass
class RuntimeRobot(_Hippo):
    id: int

    @property
    def name(self):
        return f"robot{self.id+1}"

    position: jnp.ndarray
    rotation: jnp.ndarray
    size: jnp.ndarray
    holding: str

    @classmethod
    def fromdict(cls, dico):
        # object_name = dico["object_name"]
        # object_description = dico["object_description"]
        id = dico["id"]
        position = dico["position"]
        rotation = dico["rotation"]

        if isinstance(position, dict):
            position = dict2xyztuple(position)

        if isinstance(rotation, dict):
            rotation = dict2xyztuple(rotation)


        position = jnp.array(position)
        rotation = jnp.array(rotation)


        self = cls(
            id=id,
            position=position,
            rotation=rotation,
            holding=None
        )

        self = self.set_holding(dico["inventory"])

        return self.cast_precision()

    def cast_precision(self):
        return self.replace(
            position=jnp.round(self.position, 3),
            rotation=jnp.round(self.rotation, 3),
            size=jnp.round(self.size, 3)
        )

    def set_holding(self, inventory):
        assert len(inventory) <= 1
        if len(inventory) == 1:
            holding = inventory[0]["objectId"]
        else:
            holding = None

        return self.replace(holding=holding)

    def change_posrotsize(self, position, rotation, size) -> Self:
        if isinstance(position, dict):
            position = dict2xyztuple(position)

        if isinstance(rotation, dict):
            rotation = dict2xyztuple(rotation)

        if isinstance(size, dict):
            size = dict2xyztuple(size)

        position = xyztuple_precision(position)
        rotation = xyztuple_precision(rotation)
        size = xyztuple_precision(size)

        position = jnp.array(position)
        rotation = jnp.array(rotation)
        size = jnp.array(size)
        return self.replace(position=position, rotation=rotation, size=size).cast_precision()

    def as_llmjson(self):
        dico = self.asdict()

        import numpy as np
        def arr2tup(key):
            arr = dico[key]
            arr = np.array(arr)
            arr = arr.tolist()
            dico[key] = (clip_number_string(float(arr[0])), clip_number_string(float(arr[1])), clip_number_string(float(arr[2])))
        arr2tup("position"); arr2tup("rotation"); arr2tup("size")

        #del dico["skill_portfolio"]
        #dico.update(self.skill_portfolio.output_dict())

        #dico.update(dico["metadata"])
        #del dico["metadata"]
        dico["id"] = self.name
        del dico["size"] # todo do we need to keep this ?
        del dico["rotation"] # todo do we need to keep this ?
        return dico

@dataclass
class RuntimeObjectContainer(_Hippo):
    object_names: List[str]
    objects_map: Dict[str, RuntimeObject]

    robot_names: List[str]
    robots_map: Dict[str, RuntimeRobot]

    obj_isOnTopOf: jnp.ndarray

    def force_obj1_inside_obj2(self, obj1, obj2):
        obj1 = self.objects_map[obj1]
        obj2 = self.objects_map[obj2]

        obj1 = obj1.replace(_force_inside_target=obj2.id)
        obj2 = obj2.replace(_force_contain_target=obj1.id)

        ret = copy.deepcopy(self.objects_map)
        ret[obj1.id] = obj1
        ret[obj2.id] = obj2

        return self.replace(objects_map=ret).resolve_spatial_attributes()

    def undo_force_obj1_inside_obj2(self, obj1, obj2):
        obj1 = self.objects_map[obj1]
        obj2 = self.objects_map[obj2]

        obj1 = obj1.replace(_force_inside_target=None)
        obj2 = obj2.replace(_force_contain_target=None)

        ret = copy.deepcopy(self.objects_map)
        ret[obj1.id] = obj1
        ret[obj2.id] = obj2

        return self.replace(objects_map=ret).resolve_spatial_attributes()

    @property
    def obj_hasOnTopOf(self):
        return self.obj_isOnTopOf.T

    obj_isInsideOf: jnp.ndarray

    @property
    def obj_hasInsideOf(self):
        return self.obj_isInsideOf.T

    obj_isBesideOf: jnp.ndarray # symmetric

    @property
    def obj_hasBesideOf(self):
        return self.obj_isBesideOf  # symmetric

    obj_distances: jnp.ndarray

    def get_obj2id_that_obj1id_is_inside_of(self, obj1_id):
        obj1_index = self.object_names.index(obj1_id)
        listed = self.obj_isInsideOf[obj1_index]

        if listed.any():
            return self.object_names[int(np.array(listed.nonzero()[0][0]))]

    def get_obj2id_that_obj1id_is_ontop_of(self, obj1_id):
        obj1_index = self.object_names.index(obj1_id)
        listed = self.obj_isOnTopOf[obj1_index]

        if listed.any():
            return self.object_names[int(np.array(listed.nonzero()[0][0]))]

    def get_obj2id_that_obj1id_is_beside_of(self, obj1_id):
        obj1_index = self.object_names.index(obj1_id)
        listed = self.obj_isBesideOf[obj1_index]

        if listed.any():
            return self.object_names[int(np.array(listed.nonzero()[0][0]))]

    def is_obj_inside_anything(self, obj1_id):
        obj1_index = self.object_names.index(obj1_id)
        return self.obj_isInsideOf[obj1_index].any()

    def is_obj_surface_free(self, obj1_id):
        return not self.obj_hasOnTopOf[self.object_names.index(obj1_id)].any()

    def get_objects_that_are_on_obj(self, obj1_id):
        ret = []
        for n in self.object_names:
            if n == obj1_id:
                continue
            if self.obj_hasOnTopOf[self.object_names.index(obj1_id)][self.object_names.index(n)]:
                ret.append(n)
        return ret

    @classmethod
    def coerce_ai2thor_metadata_objects(cls, metadata_objects):
        ret = []
        for o in metadata_objects:
            if o.get("ISROBOT", False):
                ret.append(o)
                continue

            # removes
            if o["assetId"] == "":
                continue


            # cleans
            if "objectId" in o and o["objectId"]:
                o["id"] = o["objectId"]
            else:
                o["id"] = o["name"]

            o["size"] = o["axisAlignedBoundingBox"]["size"]
            o["position"] = o["axisAlignedBoundingBox"]["center"]

            skill_metadata = []
            from hippo.simulation.skillsandconditions.skill_names import canonical_enabledname_2_llmname
            for k, v in canonical_enabledname_2_llmname().items():
                if o[k]:
                    skill_metadata.append(v)
            o["_skill_metadata"] = skill_metadata



            ret.append(o)

        return ret


    @classmethod
    def create(cls, objects: List[RuntimeObject] | List[Dict] | List[HippoObject], is_ai2thor_metadata=False):
        if is_ai2thor_metadata:
            objects = cls.coerce_ai2thor_metadata_objects(objects)

        object_names = []
        objects_map = {}
        for o in objects:

             if isinstance(o, dict):
                 o = RuntimeObject.fromdict(o)
             if isinstance(o, HippoObject):
                 o = RuntimeObject.from_hippoobject(o)

             objects_map[o.id] = o
             object_names.append(o.id)


        proto = jnp.zeros((len(objects), len(objects)), dtype=bool)

        self = cls(
            object_names,
            objects_map,
            robots_map={},
            robot_names=[],
            obj_isOnTopOf=proto,
            obj_isInsideOf=proto,
            obj_isBesideOf=proto,
            obj_distances=proto.astype(jnp.float32),
        )

        return self.resolve_spatial_attributes()

    def resolve_spatial_attributes(self) -> Self:
        positions = [self.objects_map[o].position for o in self.object_names]
        positions = positions + [self.robots_map[o].position for o in self.robot_names]
        positions = jnp.stack(positions)

        sizes = [self.objects_map[o].size for o in self.object_names] + [self.robots_map[o].position for o in self.robot_names]
        sizes = jnp.stack(sizes)

        kwargs = resolve_spatial_attributes(positions, sizes)

        for i, k in enumerate(self.object_names):
            v = self.objects_map[k]
            try:
                if v._force_inside_target is not None:
                    j = self.object_names.index(v._force_inside_target)
                    kwargs["obj_isInsideOf"] = kwargs["obj_isInsideOf"].at[i,j].set(True)
                    kwargs["obj_isOnTopOf"] = kwargs["obj_isOnTopOf"].at[i,j].set(False)
                    kwargs["obj_isBesideOf"] = kwargs["obj_isBesideOf"].at[i,j].set(False)
                    kwargs["obj_distances"] = kwargs["obj_distances"].at[i,j].set(False)
            except:
                pass

        #p_patate = positions[22]
        #s_patate = sizes[22]

        #p_fridge = positions[6]
        #s_fridge = sizes[6]
        #p_patate = swap_yz(p_patate[None])[0]
        #p_fridge = swap_yz(p_fridge[None])[0]

        #print(isInside(p_patate, s_patate, p_fridge, s_fridge))

        #p_patate = p_patate.at[-1].set(p_patate[-1] + s_patate[-1]/2)
        #p_fridge = p_fridge.at[-1].set(p_fridge[-1] + s_fridge[-1]/2)
        #print(isInside(p_patate, s_patate, p_fridge, s_fridge))

        return self.replace(**kwargs)


    def bool_spatial_attribute_to_list(self, j, attrvec):
        attrvec = np.array(attrvec).squeeze()
        attrvec[j] = False
        valid_indices = np.arange(len(attrvec))[attrvec]

        robot_names = [f"robot{i+1}" for i in self.robot_names]

        ret = []
        for valid_index in valid_indices:
            ret.append((self.object_names+robot_names)[valid_index])
        return ret

    def float_spatial_attribute_to_dict(self, j, attrvec):
        attrvec = np.array(attrvec).squeeze()
        attrvec[j] = False
        valid_indices = np.arange(len(attrvec))[attrvec > 0]
        robot_names = [f"robot{i + 1}" for i in self.robot_names]
        ret = {}
        for valid_index in valid_indices:
            ret[(self.object_names+robot_names)[valid_index]] = clip_number_string(float(attrvec[valid_index]))
        return ret

    def set_robots(self, objects: List[Dict]):
        robot_names = []
        robotdico = {}
        for object in objects:
            if object.get("ISROBOT", False) is False:
                continue

            id = object["id"]
            existing_robot = RuntimeRobot(id, None, None, None, None)
            existing_robot = existing_robot.change_posrotsize(object["position"], object["rotation"], object["size"])
            existing_robot = existing_robot.set_holding(object["inventory"])
            robotdico[id] = existing_robot
            robot_names.append(id)
        return self.replace(robots_map=robotdico, robot_names=robot_names)

    def update_from_ai2thor(self, objects: List[Dict]):
        objects = RuntimeObjectContainer.coerce_ai2thor_metadata_objects(objects)

        NUM_UPDATED_OBJECTS = 0

        robotdico = {}
        dico = {}
        for object in objects:
            id = object["id"]

            if id not in self.objects_map:

                if object.get("ISROBOT", False):
                    assert id in self.robots_map
                    existing_robot = self.robots_map[id]
                    existing_robot = existing_robot.change_posrotsize(object["position"], object["rotation"], object["size"])
                    existing_robot = existing_robot.set_holding(object["inventory"])
                    robotdico[id] = existing_robot
                continue

            existing_object = self.objects_map[id]
            existing_object = existing_object.change_posrotsize(object["position"], object["rotation"], object["size"])

            heldBy = object.get("heldBy", None)
            if object["isPickedUp"]:
                assert heldBy is not None, f"object {id} is picked up but has no heldBy"

            existing_object = existing_object.replace(heldBy=heldBy)

            dico[id] = existing_object

            NUM_UPDATED_OBJECTS += 1
        assert NUM_UPDATED_OBJECTS == len(self.objects_map)
        return self.replace(objects_map=dico, robots_map=robotdico).resolve_spatial_attributes()

    #def diff(self, new_runtimecontainer: Self, last_action=None):
    #    selfdict = self.as_llmjson()
    #    newdict = new_runtimecontainer.as_llmjson()
    #
    #    return git_diff(selfdict, newdict)

    def get_object_by_id(self, object_id):
        return self.objects_map[object_id]

    def get_object_by_id_might_not_exist(self, object_id, sas=None):
        try:
            return self.get_object_by_id(object_id)
        except KeyError:
            if sas is None:
                from hippo.simulation.skillsandconditions.sas import SimulationActionState
                sas = SimulationActionState(self, None, None, object_id, None)
            from hippo.simulation.skillsandconditions.conditions import CONDITION_ObjectExists, maybe_raise_condition_exception
            return maybe_raise_condition_exception([CONDITION_ObjectExists().replace(sas=sas.replace(target_object_id=object_id), state=False, success=False)])

    def update_object(self, object) -> Self:
        dico = copy.deepcopy(self.objects_map)
        dico[object.id] = object
        return self.replace(objects_map=dico)
    
    def as_llmjson(self):

        obj_isOnTopOf = np.array(self.obj_isOnTopOf)
        obj_isInsideOf = np.array(self.obj_isInsideOf)
        obj_isBesideOf = np.array(self.obj_isBesideOf)
        obj_distances = np.array(self.obj_distances)


        final_dict = {}
        for i, name in enumerate((self.object_names+self.robot_names)):
            if isinstance(name, str):   # fixme this is gross
                objdict = self.objects_map[name].as_llmjson()
                objdict["isOnTopOf"] = self.bool_spatial_attribute_to_list(i, obj_isOnTopOf[i])
                objdict["isInsideOf"] = self.bool_spatial_attribute_to_list(i, obj_isInsideOf[i])
                # objdict["isBesideOf"] = self.bool_spatial_attribute_to_list(i, obj_isBesideOf[i])
            else:
                objdict = self.robots_map[name].as_llmjson()
                name = f"robot{name}"

            objdict["object_distances"] = self.float_spatial_attribute_to_dict(i, obj_distances[i])

            final_dict[name] = objdict

        return final_dict

    def get_object_list_with_children(self):
        def is_robot_key(k):
            if "robot" not in k:
                return False
            try:
                if str(int(k.replace("robot", ""))) == k.replace("robot", ""):
                    return True
            except:
                pass
            return False

        asllmjson = self.as_llmjson()
        ret = {k: {"hasInside": [], "hasOnTopOf": []} for k in asllmjson.keys() if not is_robot_key(k)}
        for k, v in asllmjson.items():
            if is_robot_key(k):
                continue

            if len(v["isInsideOf"]) > 0:
                for e in v["isInsideOf"]:
                    if is_robot_key(e):
                        continue
                    ret[e]["hasInside"].append(k)
            if len(v["isOnTopOf"]) > 0:
                for e in v["isOnTopOf"]:
                    if is_robot_key(e):
                        continue
                    ret[e]["hasOnTopOf"].append(k)

        for k, v in copy.deepcopy(ret).items():
            for e in (v["hasInside"] + v["hasOnTopOf"]):
                if e in ret:
                    del ret[e]

        for k, v in ret.items():
            if len(v["hasInside"]) == 0:
                del v["hasInside"]
            if len(v["hasOnTopOf"]) == 0:
                del v["hasOnTopOf"]

        return ret

    def get_held_object(self):
        for name, object in self.objects_map.items():
            if object.heldBy:
                return object
        return None

    def get_object_list_with_children_as_string(self):
        ret = self.get_object_list_with_children()
        ret = str(ret).replace(": {}", "")
        return ret

@jax.jit
def swap_yz(mat):
    Z = mat[:,-1]
    return mat.at[:,-1].set(mat[:,-2]).at[:,-2].set(Z)

@jax.jit
def center_position(pos, siz):
    return pos + siz/2 #pos.at[-1].set(pos[-1] + siz[-1] /2)

@jax.jit
def resolve_spatial_attributes(positions, sizes):

    def for_each_object(positions, sizes, p1,s1):
        def each_other_object(p1, s1, p2,s2):
            return isOnTop(p1,s1,p2,s2), isInside(p1,s1,p2,s2), isBeside(p1,s1,p2,s2), distance(p1,s1,p2,s2)
        return jax.vmap(functools.partial(each_other_object, p1, s1))(positions, sizes)

    positions = swap_yz(positions)
    sizes = swap_yz(sizes)

    #positions = positions.at[:,-1].set(positions[:,-1] + sizes[:,-1] / 2)
    #positions = jax.vmap(center_position)(positions, sizes)

    obj_isOnTopOf, obj_isInsideOf, obj_isBesideOf, obj_distances = jax.vmap(functools.partial(for_each_object, positions, sizes))(positions, sizes)

    def fix_for_each_object(arr, i):
        return arr.at[i].set(False)
    indices = jnp.arange(positions.shape[0])
    obj_isOnTopOf = jax.vmap(fix_for_each_object)(obj_isOnTopOf, indices)
    obj_isInsideOf = jax.vmap(fix_for_each_object)(obj_isInsideOf, indices)
    obj_isBesideOf = jax.vmap(fix_for_each_object)(obj_isBesideOf, indices)

    # last one is always robot
    obj_isOnTopOf = obj_isOnTopOf.at[:,-1].set(False).at[-1,:].set(False)
    obj_isInsideOf = obj_isInsideOf.at[:,-1].set(False).at[-1,:].set(False)

    def fix_conditionals(obj_isOnTopOf, obj_isInsideOf, obj_isBesideOf):
        obj_isInsideOf = jnp.zeros_like(obj_isInsideOf) * jnp.any(obj_isOnTopOf).astype(int) + obj_isInsideOf * (1-jnp.any(obj_isOnTopOf))
        obj_isBesideOf = jnp.zeros_like(obj_isBesideOf) * jnp.any(obj_isInsideOf).astype(int) + obj_isBesideOf * (1-jnp.any(obj_isInsideOf))
        return obj_isOnTopOf.astype(bool), obj_isInsideOf.astype(bool), obj_isBesideOf.astype(bool)
    obj_isOnTopOf, obj_isInsideOf, obj_isBesideOf = jax.vmap(fix_conditionals)(obj_isOnTopOf, obj_isInsideOf, obj_isBesideOf)

    obj_distances = jax.vmap(lambda x: jnp.round(x, decimals=2))(obj_distances.squeeze())


    return {
        "obj_isOnTopOf": obj_isOnTopOf.squeeze(),
        "obj_isInsideOf": obj_isInsideOf.squeeze(),
        "obj_isBesideOf": obj_isBesideOf.squeeze(),
        "obj_distances": obj_distances.squeeze(),
    }



