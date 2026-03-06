import numpy as np
def dict2xyztuple(dic):
    return (dic['x'], dic['y'], dic['z'])
from hippo.utils.spatial_utils import pcd_bbox_size, transform_ai2thor_object
from hippo.reconstruction.assetlookup.assetalign import pcd_or_mesh_to_np, add_scaling_to_transmat

#from diskcache import FanoutCache, Cache
#CACHEPATH = "/".join(__file__.split("/")[:-1]) + "/diskcache"
#cache = Cache(CACHEPATH)

#@cache.memoize()
def translate(ai2thor_obj, tgt_pcd):
    from hippo.utils.spatial_utils import pcd_bbox_size
    from hippo.reconstruction.assetlookup.assetalign import pcd_or_mesh_to_np, add_scaling_to_transmat

    def bbox(pcd):
        return np.array(pcd).min(axis=0), np.array(pcd).max(axis=0)

    src_bbox = bbox(pcd_or_mesh_to_np(ai2thor_obj, mesh_keep_vertex_OR_sample_points="keep_vertex"))
    target_bbox = bbox(tgt_pcd)

    #target_size = np.array(dict2xyztuple(pcd_bbox_size(target_pcd)))
    #obj_size = np.array(dict2xyztuple(pcd_bbox_size(pcd_or_mesh_to_np(ai2thor_obj))))

    #scale_factors = target_size / obj_size

    #if mode == 'fit':
    #    uniform_scale = np.min(scale_factors)  # Ensures object fits within target
    #elif mode == 'fill':
    #    uniform_scale = np.max(scale_factors)  # Ensures object fills target
    #else:
    #    raise ValueError("mode must be either 'fit' or 'fill'")

    #scaling = np.array([uniform_scale, uniform_scale, uniform_scale])

    transmat = np.eye(4) # add_scaling_to_transmat(, scaling)

    src_center = (src_bbox[0] + src_bbox[1]) / 2
    tgt_center = (target_bbox[0] + target_bbox[1]) / 2

    translation = tgt_center - src_center
    #translation[1] = 0
    transmat[:3, 3] = translation

    ai2thor_obj = transform_ai2thor_object(ai2thor_obj, transmat)

    return ai2thor_obj

def get_translate_position(ai2thor_obj, tgt_pcd):
    from hippo.utils.spatial_utils import pcd_bbox_size
    from hippo.reconstruction.assetlookup.assetalign import pcd_or_mesh_to_np, add_scaling_to_transmat

    def bbox(pcd):
        return np.array(pcd).min(axis=0), np.array(pcd).max(axis=0)

    src_bbox = bbox(pcd_or_mesh_to_np(ai2thor_obj, mesh_keep_vertex_OR_sample_points="keep_vertex"))
    target_bbox = bbox(tgt_pcd)

    #target_size = np.array(dict2xyztuple(pcd_bbox_size(target_pcd)))
    #obj_size = np.array(dict2xyztuple(pcd_bbox_size(pcd_or_mesh_to_np(ai2thor_obj))))

    #scale_factors = target_size / obj_size

    #if mode == 'fit':
    #    uniform_scale = np.min(scale_factors)  # Ensures object fits within target
    #elif mode == 'fill':
    #    uniform_scale = np.max(scale_factors)  # Ensures object fills target
    #else:
    #    raise ValueError("mode must be either 'fit' or 'fill'")

    #scaling = np.array([uniform_scale, uniform_scale, uniform_scale])

    transmat = np.eye(4) # add_scaling_to_transmat(, scaling)

    src_center = (src_bbox[0] + src_bbox[1]) / 2
    tgt_center = (target_bbox[0] + target_bbox[1]) / 2

    translation = tgt_center - src_center
    return translation
    #translation[1] = 0
    transmat[:3, 3] = translation

    ai2thor_obj = transform_ai2thor_object(ai2thor_obj, transmat)

    return ai2thor_obj


