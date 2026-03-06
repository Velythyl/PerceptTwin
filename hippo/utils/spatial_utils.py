import copy

import open3d
import open3d as o3d
import numpy as np
from ai2thor.util.runtime_assets import load_existing_thor_asset_file, save_thor_asset_file

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR
from hippo.utils.o3d_np_v3v import v3v, v3iv


def get_bounding_box(points):

    if isinstance(points, o3d.geometry.MeshBase):
        bbox = points.get_axis_aligned_bounding_box()

        max_bounds = bbox.get_max_bound()
        min_bounds = bbox.get_min_bound()
    else:
        if isinstance(points, o3d.geometry.PointCloud):
            points = np.asarray(points.points)

        min_bounds, max_bounds = points.min(axis=0), points.max(axis=0)

    return *min_bounds, *max_bounds


def get_size(points, as_dict=True):
    bbox = get_bounding_box(points)
    min_bounds, max_bounds = bbox[:3], bbox[3:]

    dico = {k: max_bounds[i] - min_bounds[i] for i,k in enumerate(["x", "y", "z"])}

    if as_dict:
        return dico

    return dico["x"], dico["y"], dico["z"]


def calculate_volume(bbox):
    """Calculate the volume of an axis-aligned bounding box."""
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    return max(0, x_max - x_min) * max(0, y_max - y_min) * max(0, z_max - z_min)


def calculate_intersection_volume(bbox1, bbox2):
    """Calculate the volume of the intersection between two bounding boxes."""
    x_min1, y_min1, z_min1, x_max1, y_max1, z_max1 = bbox1
    x_min2, y_min2, z_min2, x_max2, y_max2, z_max2 = bbox2

    x_min_int = max(x_min1, x_min2)
    y_min_int = max(y_min1, y_min2)
    z_min_int = max(z_min1, z_min2)
    x_max_int = min(x_max1, x_max2)
    y_max_int = min(y_max1, y_max2)
    z_max_int = min(z_max1, z_max2)

    intersection_bbox = (x_min_int, y_min_int, z_min_int, x_max_int, y_max_int, z_max_int)
    return calculate_volume(intersection_bbox)


def is_fully_contained(bbox1, bbox2):
    """Check if bbox2 is fully contained in bbox1, or vice versa."""

    def contained(bbox_inner, bbox_outer):
        return (bbox_outer[0] <= bbox_inner[0] <= bbox_inner[3] <= bbox_outer[3] and
                bbox_outer[1] <= bbox_inner[1] <= bbox_inner[4] <= bbox_outer[4] and
                bbox_outer[2] <= bbox_inner[2] <= bbox_inner[5] <= bbox_outer[5])

    return contained(bbox2, bbox1) or contained(bbox1, bbox2)


def disambiguate(bbox1, cloud1, bbox2, cloud2, tresh_overlap=0.8, tresh_noice=0.5):
    # Step 1: Check for zero volumes
    volume1 = calculate_volume(bbox1)
    volume2 = calculate_volume(bbox2)

    if volume1 == 0 and volume2 == 0:
        return (False, False)  # Both are noise
    elif volume1 == 0:
        return (False, True)  # Only obj2 is real
    elif volume2 == 0:
        return (True, False)  # Only obj1 is real

    # Step 2: Check for full containment
    if is_fully_contained(bbox1, bbox2):
        return (True, False)  # obj2 is noise, fully contained in obj1
    if is_fully_contained(bbox2, bbox1):
        return (False, True)  # obj1 is noise, fully contained in obj2

    # Step 3: Calculate intersection volume
    intersection_volume = calculate_intersection_volume(bbox1, bbox2)

    if intersection_volume != 0:
        if len(cloud1) > len(cloud2):
            return True, False
        else:
            return False, True
    return True, True

    # Step 4: Calculate overlap ratios
    smaller_volume = min(volume1, volume2)
    larger_volume = max(volume1, volume2)
    overlap_ratio = intersection_volume / smaller_volume
    noise_ratio = intersection_volume / larger_volume

    # fixme requires tuning
    # Step 5: Decision logic for partial overlaps
    if overlap_ratio > 0.1:
        return (True, False) if volume1 > volume2 else (False, True)
    elif noise_ratio > 0.1:
        return (True, False) if volume1 > volume2 else (False, True)

    return True, True

from scipy.spatial import ConvexHull, KDTree

def disambiguate2(cloud1, cloud2, overlap_threshold=0.1, volume_threshold=1e-6):
    """
    Compare two point clouds and determine which to keep based on overlap and volume.

    Args:
        cloud1 (np.ndarray): First point cloud (N x 3)
        cloud2 (np.ndarray): Second point cloud (M x 3)
        overlap_threshold (float): Fraction of points that must overlap to consider clouds the same
        volume_threshold (float): Minimum volume difference to consider one cloud larger

    Returns:
        tuple: (keep_cloud1, keep_cloud2) booleans indicating which clouds to keep
    """
    # Check if either cloud is empty
    if len(cloud1) == 0 or len(cloud2) == 0:
        return (len(cloud1) > 0, len(cloud2) > 0)

    # Compute volumes using convex hull
    def compute_volume(points):
        if len(points) < 4:  # Need at least 4 points for a 3D convex hull
            return 0
        try:
            hull = ConvexHull(points)
            return hull.volume
        except:
            return 0

    vol1 = compute_volume(cloud1)
    vol2 = compute_volume(cloud2)

    # Check for overlap
    def check_overlap(cloud_a, cloud_b, threshold):
        if len(cloud_a) == 0 or len(cloud_b) == 0:
            return False

        # Build KDTree for cloud_b
        tree = KDTree(cloud_b)

        # Find nearest neighbor in cloud_b for each point in cloud_a
        distances, _ = tree.query(cloud_a, k=1)

        # Count points that are very close (consider them overlapping)
        close_points = np.sum(distances < 0.30)  # 0.01 is 1cm threshold for overlap
        overlap_fraction = close_points / len(cloud_a)

        return overlap_fraction >= threshold

    has_overlap = check_overlap(cloud1, cloud2, overlap_threshold) or \
                  check_overlap(cloud2, cloud1, overlap_threshold)

    # Determine which to keep
    if has_overlap:
        if abs(vol1 - vol2) < volume_threshold:
            # Essentially same volume, keep both
            return (True, False)
        elif vol1 > vol2:
            return (True, False)
        else:
            return (False, True)
    else:
        # No overlap, keep both
        return (True, True)

from diskcache import FanoutCache, Cache
CACHEPATH = "/".join(__file__.split("/")[:-1]) + "/diskcache"
cache = Cache(CACHEPATH)
@cache.memoize()
def pcd2mesh(cloud1):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = v3v(cloud1)


    # Create mesh from point clouds using ball pivoting
    # (or you could use other methods like Poisson reconstruction)
    # Estimate normals for both point clouds
    pcd1.estimate_normals()

    # Parameters for ball pivoting
    radii = [0.05, 0.1, 0.2, 0.4]  # adjust based on your point cloud density

    # Create meshes
    mesh1 = open3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd1, o3d.utility.DoubleVector(radii)))
    return mesh1


def disambiguate3(cloud1, cloud2, overlap_threshold=0.1, volume_threshold=1e-6):
    """
    Compare two point clouds and determine which to keep based on overlap and volume.

    Args:
        cloud1 (np.ndarray): First point cloud (N x 3)
        cloud2 (np.ndarray): Second point cloud (M x 3)
        overlap_threshold (float): Fraction of points that must overlap to consider clouds the same
        volume_threshold (float): Minimum volume difference to consider one cloud larger

    Returns:
        tuple: (keep_cloud1, keep_cloud2) booleans indicating which clouds to keep
    """
    # Check if either cloud is empty
    if len(cloud1) == 0 or len(cloud2) == 0:
        return (len(cloud1) > 0, len(cloud2) > 0)

    # Compute volumes using convex hull
    def compute_volume(points):
        if len(points) < 4:  # Need at least 4 points for a 3D convex hull
            return 0
        try:
            hull = ConvexHull(points)
            return hull.volume
        except:
            return 0

    vol1 = compute_volume(cloud1)
    vol2 = compute_volume(cloud2)


    #has_overlap = check_overlap(cloud1, cloud2, overlap_threshold) or \
    #              check_overlap(cloud2, cloud1, overlap_threshold)
    def check_point_clouds_overlap(cloud1, cloud2):
        """
        Check if two point clouds overlap by converting them to meshes and testing for intersection.

        Args:
            cloud1 (np.ndarray): First point cloud (N x 3)
            cloud2 (np.ndarray): Second point cloud (M x 3)

        Returns:
            bool: True if the point clouds overlap, False otherwise
        """
        # Return False if either cloud is empty
        if len(cloud1) == 0 or len(cloud2) == 0:
            return False

        # Convert numpy arrays to Open3D point clouds
        #pcd1 = o3d.geometry.PointCloud()
        #pcd1.points = o3d.utility.Vector3dVector(cloud1)

        #pcd2 = o3d.geometry.PointCloud()
        #pcd2.points = o3d.utility.Vector3dVector(cloud2)

        # Create mesh from point clouds using ball pivoting
        # (or you could use other methods like Poisson reconstruction)
        try:

            mesh1 = pcd2mesh(cloud1)
            mesh2 = pcd2mesh(cloud2)

            # Check for intersection
            intersection = mesh1.boolean_intersection(mesh2)
            return len(np.array(intersection.vertex.positions)) > 0
            #return intersection

        except Exception as e:
            print(f"Error in mesh creation/intersection: {e}")
            return False
    has_overlap = check_point_clouds_overlap(cloud1, cloud2) or check_point_clouds_overlap(cloud2, cloud1)


    # Determine which to keep
    if has_overlap:
        if abs(vol1 - vol2) < volume_threshold:
            # Essentially same volume, keep both
            return (True, False)
        elif vol1 > vol2:
            return (True, False)
        else:
            return (False, True)
    else:
        # No overlap, keep both
        return (True, True)


def rescale_asset(asset_dir, asset_id, scaling):
    import os
    asset_path = os.path.join(asset_dir, asset_id)
    if not os.path.exists(asset_path):
        raise FileNotFoundError(asset_path)

    o3d.io.read_triangle_mesh(asset_path)


def read_list_dictpoints(obj, key):
    return np.array([[p["x"], p["y"], p["z"]] for p in obj[key]])

def write_list_dictpoints(obj, key, value):
    #obj = copy.deepcopy(obj)
    obj[key] = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in value]
    return obj

def scale_ai2thor_object(obj, scale_factors):
    #obj = copy.deepcopy(obj)

    sx, sy, sz = scale_factors

    vertices = read_list_dictpoints(obj, "vertices")
    vertices *= np.array([sx, sy, sz])
    obj = write_list_dictpoints(obj, "vertices", vertices)

    # Scale visibility points if they exist
    if 'visibilityPoints' in obj:
        vis_points = read_list_dictpoints(obj, "visibilityPoints")
        vis_points *= np.array([sx, sy, sz])
        obj = write_list_dictpoints(obj, "visibilityPoints", vis_points)

    # Adjust normals for non-uniform scaling
    if sx != sy or sy != sz or sx != sz:
        normals = read_list_dictpoints(obj, "normals")
        scale_matrix = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1 / sz]])
        normals = normals @ scale_matrix.T
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize
        obj = write_list_dictpoints(obj, "normals", normals)

    new_colliders = []
    for collider in obj["colliders"]:
        vertices = read_list_dictpoints(collider, "vertices")
        vertices *= np.array([sx, sy, sz])
        collider = write_list_dictpoints(collider, "vertices", vertices)
        new_colliders.append(collider)

    obj["colliders"] = new_colliders
    return obj


import numpy as np

def make_trans_mat_from_axisscale(scaling):
    sx, sy, sz = scaling
    # Create a scaling matrix
    scale_matrix = np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])
    return scale_matrix

#from diskcache import FanoutCache, Cache
#CACHEPATH = "/".join(__file__.split("/")[:-1]) + "/diskcache"
#cache = Cache(CACHEPATH)
#@cache.memoize()
def transform_ai2thor_object(obj, transformation_matrix):
    """
    Apply a 4x4 transformation matrix to an AI2-THOR object.

    Args:
        obj: The AI2-THOR object to transform
        transformation_matrix: 4x4 numpy array representing the transformation
                              (should include homogeneous coordinates)
    Returns:
        The transformed object
    """
    # Extract the 3x3 linear transformation part for normals
    linear_transform = transformation_matrix[:3, :3]
    normal_transform = np.linalg.inv(linear_transform).T  # Correct transform for normals

    # Helper function to apply transformation to points
    def transform_points(points):
        # Convert to homogeneous coordinates
        homogeneous = np.column_stack([points, np.ones(len(points))])
        # Apply transformation
        transformed = homogeneous @ transformation_matrix.T
        # Convert back to 3D coordinates
        return transformed[:, :3]

    # Transform vertices
    vertices = read_list_dictpoints(obj, "vertices")
    obj = write_list_dictpoints(obj, "vertices", transform_points(vertices))

    # Transform visibility points if they exist
    if 'visibilityPoints' in obj:
        vis_points = read_list_dictpoints(obj, "visibilityPoints")
        obj = write_list_dictpoints(obj, "visibilityPoints", transform_points(vis_points))

    # Transform normals (special handling)
    if 'normals' in obj:
        normals = read_list_dictpoints(obj, "normals")
        normals = normals @ normal_transform.T
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize
        obj = write_list_dictpoints(obj, "normals", normals)

    # Transform colliders
    new_colliders = []
    for collider in obj["colliders"]:
        vertices = read_list_dictpoints(collider, "vertices")
        collider = write_list_dictpoints(collider, "vertices", transform_points(vertices))
        new_colliders.append(collider)

    obj["colliders"] = new_colliders
    return obj

def get_ai2thor_object_bbox(obj):

    vertices = read_list_dictpoints(obj, "vertices")

    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)

    return {k: maxs[i] - mins[i] for i, k in enumerate(['x', 'y', 'z'])}

def pcd_bbox_size(obj):
    mins = np.min(obj, axis=0)
    maxs = np.max(obj, axis=0)

    return {k: maxs[i] - mins[i] for i, k in enumerate(['x', 'y', 'z'])}

def ai2thor_to_mesh(ai2thor_obj):
    # Example: replace these with your actual data
    newpcs = []
    for p in ai2thor_obj["vertices"]:
        newpcs.append(np.array([p['x'], p['y'], p['z']]))
    vertices = np.vstack(newpcs)

    triangles = np.array(ai2thor_obj["triangles"], dtype=np.int32).reshape(-1, 3)

    normals = []
    for p in ai2thor_obj["normals"]:
        normals.append(np.array([p['x'], p['y'], p['z']]))
    normals = np.vstack(normals)

    # Create the TriangleMesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = v3v(vertices)
    mesh.triangles = v3iv(triangles)
    mesh.vertex_normals = v3v(normals)
    return mesh

def mesh_bbox(mesh):
    ai2thor_to_mesh


def vis_ai2thor_object(obj):
    combined_mesh = o3d.geometry.TriangleMesh()

    for collider in obj["colliders"]:
        # Load vertices and reshape
        vertices = read_list_dictpoints(collider, "vertices") # np.array(collider["vertices"], dtype=np.float32).reshape(-1, 3)

        # Load triangle indices and reshape
        triangles = np.array(collider["triangles"], dtype=np.int32).reshape(-1, 3)

        # Create a mesh for this collider
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = v3v(vertices)
        mesh.triangles = v3iv(triangles)

        # Merge with combined mesh
        combined_mesh += mesh

    o3d.visualization.draw_geometries([mesh])
    return


def filter_points_by_y_quartile(points, lower_percentile=5, upper_percentile=95, points_colors=None):
    """
    Filters points based on their y-axis values, removing points outside the specified percentile range.

    Parameters:
    - points (numpy.ndarray): (N, 3) array of 3D points.
    - lower_percentile (float): Lower bound percentile (e.g., 5 for the 5th percentile).
    - upper_percentile (float): Upper bound percentile (e.g., 95 for the 95th percentile).

    Returns:
    - filtered_points (numpy.ndarray): The points within the y-axis bounds.
    - bbox_min (numpy.ndarray): Min corner of the bounding box (x_min, y_min, z_min).
    - bbox_max (numpy.ndarray): Max corner of the bounding box (x_max, y_max, z_max).
    """
    # Extract y-values from the points
    y_values = points[:, 1]

    # Calculate percentiles for the y-axis
    y_lower = np.percentile(y_values, lower_percentile)
    y_upper = np.percentile(y_values, upper_percentile)

    # Filter out points based on y-axis bounds
    mask = (y_values >= y_lower) & (y_values <= y_upper)
    filtered_points = points[mask]

    # Compute bounding box
    #bbox_min = np.min(filtered_points, axis=0)
    #bbox_max = np.max(filtered_points, axis=0)

    if points_colors is not None:
        return filtered_points, points_colors[mask]

    return filtered_points #, bbox_min, bbox_max



if __name__ == '__main__':
    x = load_existing_thor_asset_file(OBJATHOR_ASSETS_DIR, "546bb213065c45b4aa3267e85d03e725/546bb213065c45b4aa3267e85d03e725")

    save_thor_asset_file(x, "../temp.pkl.gz")

    vis_ai2thor_object(x)
    exit()
    scale_ai2thor_object(x, (1, 0.8, 2))


    print(x.keys())

    rescale_asset(OBJATHOR_ASSETS_DIR, "9c219d7d26ac47beaa46539a9a4fa8da/9c219d7d26ac47beaa46539a9a4fa8da.pkl.gz", None)

