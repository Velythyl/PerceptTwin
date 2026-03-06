import copy
import json
from pathlib import Path
import os
import numpy as np
import open3d as o3d

import numpy as np

from hippo.utils.o3d_np_v3v import v3v


def sample_prism_surface(center, dims, N):
    """
    Sample N points approximately uniformly on the surface of a rectangular prism.

    Parameters
    ----------
    center : array-like of shape (3,)
        (cx, cy, cz) — center of the prism.
    dims : array-like of shape (3,)
        (dx, dy, dz) — width (x), height (y), depth (z).
    N : int
        Number of points to sample.

    Returns
    -------
    points : ndarray of shape (N, 3)
        Sampled points.
    """
    cx, cy, cz = center
    dx, dy, dz = dims

    # Areas of each pair of opposite faces
    A_xy = dx * dy
    A_yz = dy * dz
    A_xz = dx * dz
    total_area = 2 * (A_xy + A_yz + A_xz)

    # Probabilities proportional to face areas
    probs = np.array([
        A_xy, A_xy,  # z = top/bottom
        A_yz, A_yz,  # x = left/right
        A_xz, A_xz  # y = front/back
    ]) / total_area

    # Sample which face each point goes to
    faces = np.random.choice(6, size=N, p=probs)

    points = np.zeros((N, 3))

    # Half-dimensions
    hx, hy, hz = dx / 2, dy / 2, dz / 2

    # For each face, sample points
    for i in range(6):
        mask = (faces == i)
        n_i = np.sum(mask)
        if n_i == 0:
            continue

        u = np.random.uniform(-1, 1, size=(n_i,))
        v = np.random.uniform(-1, 1, size=(n_i,))

        if i == 0:  # z = +hz
            points[mask] = np.column_stack((
                cx + u * hx,
                cy + v * hy,
                np.full(n_i, cz + hz)
            ))
        elif i == 1:  # z = -hz
            points[mask] = np.column_stack((
                cx + u * hx,
                cy + v * hy,
                np.full(n_i, cz - hz)
            ))
        elif i == 2:  # x = +hx
            points[mask] = np.column_stack((
                np.full(n_i, cx + hx),
                cy + u * hy,
                cz + v * hz
            ))
        elif i == 3:  # x = -hx
            points[mask] = np.column_stack((
                np.full(n_i, cx - hx),
                cy + u * hy,
                cz + v * hz
            ))
        elif i == 4:  # y = +hy
            points[mask] = np.column_stack((
                cx + u * hx,
                np.full(n_i, cy + hy),
                cz + v * hz
            ))
        elif i == 5:  # y = -hy
            points[mask] = np.column_stack((
                cx + u * hx,
                np.full(n_i, cy - hy),
                cz + v * hz
            ))

    return points


def pcd_visualize(pcd):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd)
    visualizer.poll_events()
    visualizer.update_renderer()

    #view_control = visualizer.get_view_control()
    #view_control.set_front([1, 0, 0])
    #view_control.set_up([0, 0, 1])
    #view_control.set_lookat([0, 0, 0])
    try:
        visualizer.run()
    except KeyboardInterrupt:
        pass
    visualizer.close()

def remove_outliers(pcd, nb_neighbors=500, std_ratio=0.005):
    """
    Remove outliers from a point cloud using statistical outlier removal.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        nb_neighbors (int): Number of neighbors to consider for each point.
        std_ratio (float): Threshold based on standard deviation.
                           Lower = stricter filtering.

    Returns:
        inlier_cloud (o3d.geometry.PointCloud): Point cloud with outliers removed.
        outlier_cloud (o3d.geometry.PointCloud): The removed outliers.
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    return inlier_cloud#, outlier_cloud

def load_point_cloud(cfg, path):
    path = Path(path)
    assert path.exists(), path
    pcd = o3d.io.read_point_cloud(str(path / "point_cloud.pcd"))



    if False:
        # Get points as numpy array
        points = np.asarray(pcd.points)
        # Mirror along X axis
        points[:, 0] *= -1  # Assign back
        pcd.points = v3v(points)

        points = np.asarray(pcd.points)
        prism = sample_prism_surface([0,0,0], [1,1,1], 1000)
        points = np.concatenate([points, prism])
        pcd.points = v3v(points)

    segments_anno = load_segments_anno(path)

    # Build a pcd with random colors
    pcd_o3d = []

    for ann in segments_anno["segGroups"]:
        obj = pcd.select_by_index(ann["segments"])
        obj = remove_outliers(obj, std_ratio=cfg.scene.statistical_pcd_denoise_std) if cfg.scene.statistical_pcd_denoise_std > 0 else obj
        pcd_o3d.append(obj)


    return pcd_o3d

def vis_cg(cg_pcds):
    for i, objpcd in enumerate(cg_pcds):
        np.random.seed(i)
        color_for_obj = np.random.choice(range(256), size=3)
        color_for_obj = np.repeat(color_for_obj[None], len(objpcd.points), axis=0)
        objpcd.colors = v3v(color_for_obj / 255)

    combined = cg_pcds[0]
    for other in cg_pcds[1:]:
        combined = combined + other

    o3d.visualization.draw_geometries([combined])

def load_clip_features(path):
    path = Path(path)
    with open(path / "clip_features.npy", "rb") as f:
        return np.load(f)

def load_segments_anno(path):
    path = Path(path)
    with open(path / "segments_anno.json", "r") as f:
        return json.load(f)

def load_conceptgraph(cfg, path,):
    pcd_dict = load_point_cloud(cfg, path)
    clip_features = load_clip_features(path)
    segments_anno = load_segments_anno(path)

    def setclip(grp, clip, pcd):
        grp["clip_features"] = clip
        grp["pcd"] = pcd
        return grp
    segments_anno["segGroups"] = [setclip(grp,clip,pcd) for grp, clip, pcd in zip(segments_anno["segGroups"], clip_features, pcd_dict)]

    

    from pathlib import Path
    for grp in segments_anno["segGroups"]:
        grp["paths"] = {
            "mask": [str(Path(f"{path}/segments/{grp['id']}/mask").resolve())],
            "rgb": [str(Path(f"{path}/segments/{grp['id']}/rgb").resolve())]
        }

    for grp in segments_anno["segGroups"]:
        grp["label"] = grp["label"].replace(": ", "").lower()

    def vis_id2obj():
        pcds = []
        for v in segments_anno["segGroups"]:
            pcds.append(v["pcd"])
        return vis_cg(pcds)


    """"""
    #vis_id2obj()
    #segments_anno = make_pcd_axis_aligned(segments_anno,
    #                                      0)  # align before filtering pcds because we will need the background pcds such as walls, doors, etc

    #vis_id2obj()
    #segments_anno = make_pcd_axis_aligned(segments_anno,
    #                                      1)  # align before filtering pcds because we will need the background pcds such as walls, doors, etc

    #vis_id2obj()
    #segments_anno = make_pcd_axis_aligned(segments_anno,2)    # align before filtering pcds because we will need the background pcds such as walls, doors, etc

    #vis_id2obj()

    rm_indices = path / "rm_indices.json"
    if rm_indices.exists():
        with open(rm_indices, "r") as f:
            rm_indices = json.load(f)
        print("Removing preprocessing indices:", rm_indices)
    else:
        rm_indices = []


    for i, grp in enumerate(copy.deepcopy(segments_anno["segGroups"])):
        if i in rm_indices:
            segments_anno["segGroups"][i] = None
            continue

        for to_remove in ["ceiling", "window", "window blinds", "blinds", "wall", "window blinds", "blinds", "water droplet", "carpet", "rug", "floor rug", "floor carpet", "background"]:#, "wooden sideboard"]: #, "door"]:
            if to_remove in grp["label"]:
                segments_anno["segGroups"][i] = None
                break
                


        #FOUND_KEEP = False
        #for only_keep in ["wooden cabinet", "grid window", "armchair", "boat photograph"]:
        #    if only_keep in grp["label"]:
        #        FOUND_KEEP = True
        #if not FOUND_KEEP:
        #    segments_anno["segGroups"][i] = None


    segments_anno["segGroups"] = list(filter(lambda x: x is not None, segments_anno["segGroups"]))
    for grp in segments_anno["segGroups"]:
        grp["label"] = grp["label"].replace(": ", "").lower()
    print([x["label"] for x in segments_anno["segGroups"]])



    #vis_id2obj()

    return segments_anno


def make_pcd_axis_aligned(segments_anno, axis_id):
    from hippo.reconstruction.assetlookup.assetIsPlane import is_single_plane
    from scipy.optimize import minimize
    from scipy.spatial.transform._rotation import Rotation

    # Extract point clouds and check if they're planes
    pcd_labels = []
    pcds = []
    pcd_is_plane = []
    pcd_plane_params = []
    for v in segments_anno["segGroups"]:
        print(v["label"])
        pcd_labels.append(v["label"])
        pcds.append(v["pcd"])

        plane_params = None
        try:
            is_plane, plane_params, ratio, normal_spread = is_single_plane(pcds[-1])


            if is_plane:
                if ratio < 0.02:
                    pass
                else:
                    from hippo.reconstruction.assetlookup.assetIsPlane import ask_llm_if_plane
                    llm_says_is_plane = ask_llm_if_plane(pcd_labels[-1])
                    if not llm_says_is_plane:
                        is_plane = False
        except:
            is_plane = False

        pcd_is_plane.append(is_plane)
        pcd_plane_params.append(plane_params)
        print("is plane:", is_plane)




    # Filter only planes and their parameters (normal vectors)
    plane_normals = []
    plane_pcds = []
    for is_plane, params, pcd in zip(pcd_is_plane, pcd_plane_params, pcds):
        if is_plane:
            # Assuming plane_params contains the normal vector (a, b, c) in ax + by + cz + d = 0
            normal = params[:3]  # First three components are the normal
            normal = normal / np.linalg.norm(normal)  # Normalize
            plane_normals.append(normal)
            plane_pcds.append(pcd)

    #vis_cg(plane_pcds)

    if not plane_normals:
        return segments_anno #raise ValueError("No planes found in the point cloud segments")

    # We'll find the rotation around the vertical (Z) axis that best aligns all planes
    # with either X or Y axes (for vertical planes) or with Z axis (for horizontal planes)

    def alignment_error(angle_deg):
        """Compute how much all plane normals disagree with being axis-aligned after rotation."""
        total_error = 0.0
        angle_rad = np.radians(angle_deg)

        # Create rotation matrix around vertical axis

        rot = [0,0,0]
        rot[axis_id] = angle_rad[0]
        rot = Rotation.from_rotvec(rot).as_matrix()

        rotated_pcds = []
        for normal, pcd in zip(plane_normals, plane_pcds):
            rotated_normal = rot @ normal

            # For each plane, compute how far it is from being axis-aligned
            # We consider both possibilities (aligned with X, Y, or Z)
            # The error is the minimum of the angles to any principal axis

            # Dot products with principal axes
            dot_x = abs(rotated_normal[0])
            dot_y = abs(rotated_normal[1])
            dot_z = abs(rotated_normal[2])

            # The best alignment is with the axis it's closest to
            max_dot = max(dot_x, dot_y, dot_z)

            # Error is 1 - max_dot (since dot product of aligned vectors is 1)
            error = 1 - max_dot
            total_error += error

            # Rotate the point cloud
            colors = pcd.colors
            pcd = np.array(pcd.points)
            rotated_pcd = (rot @ pcd.T).T  # Transpose to handle (N,3) shape
            #rotated_pcd = np.ascontiguousarray(rotated_pcd, dtype=np.float64)
            pcd = o3d.geometry.PointCloud()
            pcd.points = v3v(rotated_pcd)
            pcd.colors = colors
            rotated_pcds.append(pcd)

        #vis_cg(rotated_pcds)
        return total_error

    # Find the rotation angle that minimizes the alignment error

    try:
        result = minimize(alignment_error, x0=0, bounds=[(-45, 45)],method='Powell',  # Doesn't use gradient information
        options={'ftol': 10, 'xtol': 1.0})
        optimal_angle = result.x[0]
        print("Powell angle:", optimal_angle)

        if abs(optimal_angle) < 5:
            return segments_anno
        #result = minimize(alignment_error, x0=optimal_angle, bounds=[(-45, 45)],method='Powell',  # Doesn't use gradient information
        #options={'ftol': 1, 'xtol': 1.0, 'maxiter': 10})
        #optimal_angle = result.x[0]
        #print("Refined angle:", optimal_angle)

        # Apply the rotation to all point clouds
        rot = [0,0,0]
        rot[axis_id] = np.radians(optimal_angle)
        rot = Rotation.from_rotvec(rot).as_matrix()

        # Rotate all point clouds in the segments
        for segment in segments_anno["segGroups"]:
            pcd = segment["pcd"]

            points = np.array(pcd.points)
            points = (rot @ points.T).T

            pcd.points = v3v(points)
    except Exception as e:
        print("Plane alignment error:", e)
        print(f"Skipping plane alignment for axis {axis_id}")

    return segments_anno#, optimal_angle



if __name__ == "__main__":
    PATH = "../sacha_kitchen"
    #load_clip_features(PATH)

    x = load_conceptgraph(PATH)

    x = load_point_cloud(PATH)

    for _x in x:
        pcd_visualize(_x)