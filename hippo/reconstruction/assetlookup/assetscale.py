import numpy as np
def dict2xyztuple(dic):
    return (dic['x'], dic['y'], dic['z'])
from hippo.utils.spatial_utils import pcd_bbox_size, transform_ai2thor_object
from hippo.reconstruction.assetlookup.assetalign import pcd_or_mesh_to_np, add_scaling_to_transmat

#from diskcache import FanoutCache, Cache
#CACHEPATH = "/".join(__file__.split("/")[:-1]) + "/diskcache"
#cache = Cache(CACHEPATH)
#@cache.memoize()
def scale_object(cfg, ai2thor_obj, target_pcd):
    if cfg.assetfitting.scaling == "axis":
        return axis_scale(ai2thor_obj, target_pcd)
    elif cfg.assetfitting.scaling.startswith("aspect"):
        type = cfg.assetfitting.scaling.replace("aspect", "").strip()

        assert type in ["fit", "fill", "avg", "weighted"]

        ret = aspect_scale(ai2thor_obj, target_pcd, type)
        return axis_scale(ret, target_pcd, vertical_only=True)  # forces asset to match exactly in vertical axis

    raise NotImplemented(f"Scaling not implemented for <{cfg.assetfitting.scaling}>")


def axis_scale(ai2thor_obj, target_pcd, vertical_only=False):


    FIXED_TARGET_SIZE = np.array(dict2xyztuple(pcd_bbox_size(target_pcd)))

    obj_pcd_size = np.array(dict2xyztuple(pcd_bbox_size(pcd_or_mesh_to_np(ai2thor_obj, mesh_keep_vertex_OR_sample_points="keep_vertex"))))

    scaling = FIXED_TARGET_SIZE / obj_pcd_size

    if vertical_only:
        scaling[0] = 1
        scaling[2] = 1
    transmat = add_scaling_to_transmat(np.eye(4), scaling)


    #print(np.linalg.norm(scaling))
    ai2thor_obj = transform_ai2thor_object(ai2thor_obj, transmat)
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
    obj_size = np.array(dict2xyztuple(pcd_bbox_size(pcd_or_mesh_to_np(ai2thor_obj, mesh_keep_vertex_OR_sample_points="keep_vertex"))))

    scale_factors = target_size / obj_size

    if mode == 'fit':
        uniform_scale = np.min(scale_factors)  # Ensures object fits within target
    elif mode == 'fill':
        uniform_scale = np.max(scale_factors)  # Ensures object fills target
    elif mode == "avg":
        uniform_scale = np.mean(scale_factors) # Best fit
    elif mode == 'weighted':
        weights = scale_factors
        uniform_scale = np.sum(weights * scale_factors) / np.sum(weights)
    else:
        raise ValueError("mode must be either 'fit' or 'fill'")

    scaling = np.array([uniform_scale, uniform_scale, uniform_scale])
    transmat = add_scaling_to_transmat(np.eye(4), scaling)

    from hippo.utils.spatial_utils import ai2thor_to_mesh
    import open3d as o3d
    #mesh = ai2thor_to_mesh(ai2thor_obj)
    #o3d.visualization.draw_geometries([mesh])

    ai2thor_obj = transform_ai2thor_object(ai2thor_obj, transmat)

    #mesh = ai2thor_to_mesh(ai2thor_obj)
    #o3d.visualization.draw_geometries([mesh])

    return ai2thor_obj

"""
src_center = (pcd_or_mesh_to_np(ai2thor_obj).min(axis=0) + pcd_or_mesh_to_np(ai2thor_obj).max(axis=0)) / 2
tgt_center = (np.array(target_pcd).min(axis=0) + np.array(target_pcd).max(axis=0)) / 2

translation = tgt_center - np.max(scaling) * src_center
transmat[:3, 3] = translation
"""