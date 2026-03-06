import copy
import itertools
from typing import Callable

from hippo.conceptgraph.conceptgraph_intake import load_conceptgraph, vis_cg
from hippo.reconstruction.scenedata import HippoObject, HippoRoomPlan
from hippo.utils.o3d_np_v3v import v3v
from hippo.utils.spatial_utils import get_size, disambiguate, disambiguate2, disambiguate3, get_bounding_box, filter_points_by_y_quartile, is_fully_contained
from hippo.utils.string_utils import get_uuid
from omegaconf import ListConfig
import numpy as np

def chunks(xs, n):
    n = max(1, n)
    return (xs[i:i+n] for i in range(0, len(xs), n))

def filter_segGroups(cg):
    cg = copy.deepcopy(cg)
    objects = cg["segGroups"]


    """
    for i, obj in enumerate(copy.deepcopy(cg["segGroups"])):
       for out in ["wall", "BACKGROUND", "door"]:
           if out in obj["label"] or out in obj["caption"]:
               cg["segGroups"][i] = None
    ret = list(filter(lambda seggroup: seggroup is not None, cg["segGroups"]))
    cg["segGroups"] = ret
    """

    BATCH_SIZE = 5
    chunked = chunks(cg["segGroups"], BATCH_SIZE)

    counter = 0
    for i, chunk in enumerate(chunked):
        string = []
        for j, obj in enumerate(chunk):
            string += [f"{j}: {obj['label']} - {obj['caption']}"]
        string = "\n".join(string)

        prompt = f"""
I have a list of objects. I want to remove "background" objects, such as walls, doors, windows, ceilings. 
Do not remove objects that can be interacted with by humans, even if they are affixed to background objects.
If there is an electronic component to the object, always keep it.
Only remove objects that are part of the structure of the building such as walls, doors, windows, ceilings, sideboards, wall dividers, etc.
Also, some objects have faulty labels, such as describing entire parts of the scene. We need to remove the objects with captions that describe more than one object. 
We only want to keep singular objects.
Make sure to remove all walls, doors, windows, and ceilings! Even if doors are technically interactable.

First, write your reasoning. Then, write a list of objects to remove by their ID. So, just return something like:

```
[0,4,9]
```
Where 0,4 and 9 are IDs of object to remove.
Make sure to include the ```.
It is OK to output an empty list if there are no objects to remove.

Here are my objects:
{string}
""".strip()

        from llmqueries.llm import LLM
        _, response = LLM(prompt, "gpt-4.1-mini-2025-04-14", max_tokens=1000, temperature=0, stop=None, logprobs=1,
                          frequency_penalty=0)

        partial_parse = response.split('```\n[')[-1].split("]\n```")[0]

        for resp_obj in partial_parse.split(","):
            try:
                objid = int(resp_obj.strip())
                objid = counter + objid
                cg["segGroups"][objid] = None
            except:
                pass

        counter += len(chunk)

    #out_filter = ["wall", "BACKGROUND", "door"]

    #for i, obj in enumerate(copy.deepcopy(cg["segGroups"])):
    #    for out in ["wall", "BACKGROUND", "door"]:
    #        if out in obj["label"] or out in obj["caption"]:
    #            cg["segGroups"][i] = None
    ret = list(filter(lambda seggroup: seggroup is not None, cg["segGroups"]))
    cg["segGroups"] = ret
    return cg

import open3d as o3d

def get_hippos(cfg, path, pad=lambda bounddists: bounddists * 0.25):
    cg = load_conceptgraph(cfg, path)

    if hasattr(cfg.scene, "intake_swap_yz") and cfg.scene.intake_swap_yz:
        def swap_yz_func(pcd):
            points = pcd.points
            points = np.array(points)
            points[:, [1, 2]] = points[:, [2, 1]]
            pcd.points = v3v(points)
            return pcd

        for v in cg["segGroups"]:
            v["pcd"] = swap_yz_func(v["pcd"])

    #vis_cg([o["pcd"] for o in cg["segGroups"]])

    #cg = filter_segGroups(cg)
    cg_objects = cg["segGroups"]

    #vis_cg([o["pcd"] for o in cg["segGroups"]])

    hippo_objects = []

    pcds = []
    pcd_colors = []
    for i, obj in enumerate(cg_objects):
        #if i in [3, 13, 14, 25, 26]:
        #    continue

        object_name = obj["label"].lower().replace(":","").replace(",", " ").strip()
        RM_OBJ = False
        for substr in cfg.scene.remove_obj_name_substr:
            if len(substr) > 0 and substr in object_name:
                RM_OBJ = True
                break

        object_description = obj["caption"]
        for substr in cfg.scene.remove_obj_desc_substr:
            if len(substr) > 0 and substr in object_description:
                RM_OBJ = True
                break
        if RM_OBJ:
            continue

        pcd = np.asarray(obj["pcd"].points)[:,[0,2,1]]
        pcd_color = np.asarray(obj["pcd"].colors)

        clip_features = obj["clip_features"]

        pcds.append(pcd)
        pcd_colors.append(pcd_color)

        hippo_objects.append(
            HippoObject(
                object_name=object_name,
                object_description=object_description,
                roomId=None, _position=None, _clip_features=clip_features,
                _desired_size=None, _cg_paths=obj["paths"])
        )

    allpoints = np.concatenate(pcds)
    _minbound, maxbound = allpoints.min(axis=0), allpoints.max(axis=0)

    bounddists = maxbound - _minbound

    ### Figure out wall padding
    if isinstance(pad, Callable):
        pad = pad(bounddists)
        pad = np.array(pad)

    if isinstance(pad, int) or isinstance(pad, float):
        pad = np.ones(3) * pad

    if (isinstance(pad, np.ndarray) or isinstance(pad, tuple) or isinstance(pad, list) or isinstance(pad, ListConfig)):
        if len(pad) == 3:
            pad = np.array(pad)
            pad[1] = 0
            lowpad = pad
            highpad = pad
        elif len(pad) == 4:
            lowpad = [pad[0],0,pad[1]]
            highpad = [pad[2],0,pad[3]]
        else:
            raise ValueError("invalid pad")
    else:
        raise ValueError("invalid pad")
    #pad = np.ones(3) * pad
    #pad[1] = 0

    def shift_above_0(item):
        return item - _minbound + lowpad

    pcds = [shift_above_0(pcd) for pcd in pcds]

    allpoints = np.concatenate(pcds)
    minbound, maxbound = allpoints.min(axis=0), allpoints.max(axis=0)
    minbound, maxbound = minbound-lowpad, maxbound+highpad
    assert (minbound == 0).all()

    id2objs = {}
    name2objs = {}
    id2bbox = {}
    for hippo_object, pcd, pcd_color in (zip(hippo_objects,pcds, pcd_colors)):
        original_pcd = pcd

        #pcd, pcd_color = filter_points_by_y_quartile(pcd, 1, 99, points_colors=pcd_color)

        def weighted_centroid(pcd: np.ndarray, scale: float = 1.0) -> np.ndarray:
            """
            Estimate centroid of noisy point cloud by weighting points based on distance to naive mean.

            Parameters:
                pcd (np.ndarray): point cloud of shape (N, 3)
                scale (float): scaling factor for weighting function. Larger means less aggressive weighting.

            Returns:
                position (np.ndarray): estimated position, shape (3,)
            """
            if pcd.ndim != 2 or pcd.shape[1] != 3:
                raise ValueError("pcd must be of shape (N, 3)")

            # Step 1: naive centroid
            bbox_min = np.min(pcd, axis=0)
            bbox_max = np.max(pcd, axis=0)
            bbox_center = 0.5 * (bbox_min + bbox_max)
            naive_centroid = bbox_center

            # Step 2: compute distances to naive centroid
            distances = np.linalg.norm(pcd - naive_centroid, axis=1)

            # Step 3: compute weights
            # Here we use Gaussian weights: w_i = exp( - (d_i / scale)^2 )
            #if scale <= 0:
            #    scale = np.std(distances)  # fallback: use spread of points
            #weights = np.exp(-(distances / scale) ** 2)
            weights = 1 / distances

            # Step 4: compute weighted centroid
            weighted_sum = np.sum(weights[:, None] * pcd, axis=0)
            sum_weights = np.sum(weights)
            refined_centroid = weighted_sum / sum_weights

            #refined_centroid[1] = np.min(pcd[:,1])
            return refined_centroid

        position = weighted_centroid(pcd) #np.mean(pcd, axis=0)
        #position[1] = np.min(pcd[:,1])
        #if position[1] < 0.1:
        #    position[1] = 0.0
        print(hippo_object.object_name)
        print(position)
        assert (position >= (minbound+lowpad)).all()
#        assert (position <= (maxbound-pad)).all()
        size = get_size(pcd, as_dict=False)
        hippo_object = hippo_object.replace(_position=position, _desired_size=size)
        hippo_object = hippo_object.set_pcd_(pcd, pcd_color)

        names_to_add = [hippo_object.object_name] #+ hippo_object.object_name.split(" ")
        for name_to_add in names_to_add:
            if name_to_add not in name2objs:
                name2objs[name_to_add] = []
            id2bbox[hippo_object.object_name_id] = (get_bounding_box(original_pcd), len(original_pcd), original_pcd)
            name2objs[name_to_add].append((hippo_object, get_bounding_box(original_pcd), len(original_pcd), original_pcd))
        id2objs[hippo_object.object_name_id] = hippo_object

    def keep_delete(keep, id, other_id):
        if not keep:
            if id in id2objs:

                if False:# other_id in id2objs:
                    # we are keeping other_id
                    # merging...

                    points = tuple(list(id2objs[other_id]._cg_pcd_points) + list(id2objs[id]._cg_pcd_points))
                    colors = tuple(list(id2objs[other_id]._cg_pcd_colours) + list(id2objs[id]._cg_pcd_colours))
                    position = weighted_centroid(np.array(points))
                    size = get_size(np.array(points), as_dict=False)

                    cg_paths = {'mask': tuple(list(id2objs[other_id]._cg_paths['mask']) + list(id2objs[id]._cg_paths['mask'])), "rgb": list(tuple(id2objs[other_id]._cg_paths['rgb']) + tuple(id2objs[id]._cg_paths['rgb']))}

                    id2objs[other_id] = id2objs[other_id].replace(_position=position, _desired_size=size, _cg_pcd_colours=colors, _cg_pcd_points=points, _cg_paths=cg_paths)

                del id2objs[id]

                #vis_id2obj()

    def vis_id2obj():
        pcds = []
        for k, v in id2objs.items():
            pcd = o3d.geometry.PointCloud()
            pcd.points = v3v(np.array(v._cg_pcd_points))
            pcd.colors = v3v(np.array(v._cg_pcd_colours))
            pcds.append(pcd)
        return vis_cg(pcds)

    #vis_id2obj()

    # disambiguation step
    for name, hippo_objects in name2objs.items():
        if len(hippo_objects) == 1:
            continue

        for (obj1, bbox1, num_pcd1, original_pcd_1), (obj2, bbox2, num_pcd2, original_pcd_2) in itertools.combinations(hippo_objects, 2):
            if obj1.object_name_id not in id2objs:
                continue
            if obj2.object_name_id not in id2objs:
                continue
            if obj1.object_name_id == obj2.object_name_id:
                print("How is this happening? FIXME in conceptgraph_to_hippo")
                continue

            keep1, keep2 = disambiguate(bbox1,original_pcd_1, bbox2,original_pcd_2)

            if False: #sum((keep1, keep2)) <= 1:
                print("Removing")
                print(obj2.object_name_id if keep1 else obj1.object_name_id)
                print("in favor of")
                print(obj1.object_name_id if keep1 else obj2.object_name_id)
                print("Vis prior")
                vis_id2obj()

            keep_delete(keep1, obj1.object_name_id, obj2.object_name_id)
            keep_delete(keep2, obj2.object_name_id, obj1.object_name_id)

            if False: #sum((keep1, keep2)) <= 1:
                vis_id2obj()
            continue
            if keep1 == keep2:  # either true, true or false, false
                keep_delete(keep1, obj1.object_name_id)
                keep_delete(keep2, obj2.object_name_id)
            elif keep1 != keep2:    # same thing as else lol
                if num_pcd2 > num_pcd1:
                    keep_delete(False, obj1.object_name_id)
                else:
                    keep_delete(False, obj2.object_name_id)

    MOVED_OBJS = []
    for k1, (bbox1,len1,original_pcd_1) in id2bbox.items():
        for k2, (bbox2,len2,original_pcd_2) in id2bbox.items():
            if k1 == k2 or k1 in MOVED_OBJS:
                continue
            def needs_to_move(bbox1,original_pcd_1, bbox2,original_pcd_2):
                if is_fully_contained(bbox1, bbox2):
                    return (True, False)  # obj2 is fully contained in obj1
                if is_fully_contained(bbox2, bbox1):
                    return (False, True)  # obj1 is fully contained in obj2
                return (False, False)
            obj1_full_contains_obj2, obj2_full_contains_obj1 = needs_to_move(bbox1,original_pcd_1, bbox2,original_pcd_2)

            if obj1_full_contains_obj2 != obj2_full_contains_obj1:
                def raise_objX_above_objY_OLD(objX, bboxX, objY, bboxY):
                    min_x = bboxX[1] # (x,y,z,x,y,z)
                    max_y = bboxY[4]
                    dist = 0.05 # abs(max_y - min_x)

                    x_pcd = np.array(id2objs[objX]._cg_pcd_points)
                    x_pcd = x_pcd + dist
                    x_pcd = tuple(x_pcd.tolist())

                    x_pos = id2objs[objX]._position
                    x_pos = (x_pos[0], x_pos[1]+dist, x_pos[2])

                    id2objs[objX] = id2objs[objX].replace(_position=x_pos, _cg_pcd_points=x_pcd)
                    MOVED_OBJS.append(objX)
                    #MOVED_OBJS.append(objY)

                def normalize_bbox(b):
                    xmin, ymin, zmin, xmax, ymax, zmax = b
                    if xmin > xmax:
                        xmin, xmax = xmax, xmin
                    if ymin > ymax:
                        ymin, ymax = ymax, ymin
                    if zmin > zmax:
                        zmin, zmax = zmax, zmin
                    return (xmin, ymin, zmin, xmax, ymax, zmax)

                def mtv_for_aabbs(bboxX, bboxY):
                    """
                    Returns the minimum translation vector (tx,ty,tz) to apply to bboxX
                    so that bboxX and bboxY are no longer intersecting.
                    If they are already disjoint, returns (0,0,0).
                    Assumes axis-aligned bounding boxes.
                    """
                    xminX, yminX, zminX, xmaxX, ymaxX, zmaxX = normalize_bbox(bboxX)
                    xminY, yminY, zminY, xmaxY, ymaxY, zmaxY = normalize_bbox(bboxY)

                    # overlaps (positive if intersecting along that axis)
                    ox = min(xmaxX, xmaxY) - max(xminX, xminY)
                    oy = min(ymaxX, ymaxY) - max(yminX, yminY)
                    oz = min(zmaxX, zmaxY) - max(zminX, zminY)

                    # if any overlap <= 0 then boxes are separated already
                    if ox <= 0 or oy <= 0 or oz <= 0:
                        return (0.0, 0.0, 0.0)

                    # choose smallest overlap axis
                    overlaps = {'x': ox, 'y': oy, 'z': oz}
                    axis = min(overlaps, key=overlaps.get)
                    depth = overlaps[axis]

                    # find direction: push X away from Y along chosen axis
                    centerX = ((xminX + xmaxX) / 2.0, (yminX + ymaxX) / 2.0, (zminX + zmaxX) / 2.0)
                    centerY = ((xminY + xmaxY) / 2.0, (yminY + ymaxY) / 2.0, (zminY + zmaxY) / 2.0)

                    tx = ty = tz = 0.0
                    if axis == 'x':
                        sign = -1.0 if centerX[0] < centerY[0] else 1.0
                        tx = sign * depth
                    elif axis == 'y':
                        sign = -1.0 if centerX[1] < centerY[1] else 1.0
                        ty = sign * depth
                    else:  # 'z'
                        sign = -1.0 if centerX[2] < centerY[2] else 1.0
                        tz = sign * depth

                    return (tx, ty, tz)
                def raise_objX_above_objY(objX, bboxX, objY, bboxY):
                    vec = np.array([0,0.05,0]) # mtv_for_aabbs(bboxY, bboxX) # np.array([0,0.05,0]) # empirically seems to be fine #mtv_for_aabbs(bboxY, bboxX)
                    vec = np.absolute(vec)
                    print(f"Bumping {objX} by {vec}")

                    x_pcd = np.array(id2objs[objX]._cg_pcd_points)
                    x_pcd = x_pcd + np.array(vec)
                    x_pcd = tuple(x_pcd.tolist())

                    x_pos = id2objs[objX]._position
                    x_pos = (x_pos[0]+vec[0], x_pos[1]+vec[1], x_pos[2]+vec[2])

                    id2objs[objX] = id2objs[objX].replace(_position=x_pos, _cg_pcd_points=x_pcd)
                    MOVED_OBJS.append(objX)
                    #MOVED_OBJS.append(objY)
                if obj1_full_contains_obj2:
                    raise_objX_above_objY(k2, bbox2, k1, bbox1)
                else:
                    raise_objX_above_objY(k1, bbox1, k2, bbox2)



    hippo_objects = list(id2objs.values())

    minbound = (minbound[0] ,minbound[2])
    maxbound = (maxbound[0],maxbound[2])
    roomId = cg["sceneId"] + f"-{get_uuid()}"
    coords = ((minbound[0], minbound[1]), (minbound[0], maxbound[1]), (maxbound[0], maxbound[1]), (maxbound[0], minbound[1]))
    roomplan = HippoRoomPlan(id=roomId, coords=coords, center=((maxbound[0]-minbound[0]) / 2, (maxbound[1]-minbound[1])/2), size=(((maxbound[0]-minbound[0]), (maxbound[1]-minbound[1]))), wall_height=cfg.scene.wall_height, floor_type=cfg.scene.floor_material, wall_type=cfg.scene.wall_material)

    hippo_objects = [h.replace(roomId=roomId) for h in hippo_objects]

    def vis_final_pcds():
        pcds = []
        for ho in hippo_objects:
            pcd = o3d.geometry.PointCloud()
            pcd.points = v3v(np.array(ho._cg_pcd_points))
            pcd.colors = v3v(np.array(ho._cg_pcd_colours))
            pcds.append(pcd)
        return vis_cg(pcds)
    if False:
        vis = vis_final_pcds()

    return roomplan, hippo_objects


if __name__ == "__main__":
    _, hippos = get_hippos("../sacha_kitchen", pad=2)
    hippos = [hippo.to_runtimeobject() for hippo in hippos]