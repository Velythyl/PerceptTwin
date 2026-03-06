import copy
import functools
from typing import Dict

import jax.numpy as jnp
import numpy as np
import jax

from hippo.utils.o3d_np_v3v import v3v


def add_axis_rot_to_trans_mat(transmat, theta, axis='y'):
    """
    Add rotation to a transformation matrix around a specified axis.

    Parameters:
    transmat (np.array): 4x4 transformation matrix
    theta (float): rotation angle in radians
    axis (str): rotation axis ('x', 'y', or 'z')

    Returns:
    np.array: new transformation matrix with rotation applied
    """

    def rot_mat_from_theta(theta, axis):
        c = np.cos(theta)
        s = np.sin(theta)
        if axis == 'x':
            return np.array([
                [1, 0, 0, 0],
                [0, c, -s, 0],
                [0, s, c, 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'y':
            return np.array([
                [c, 0, s, 0],
                [0, 1, 0, 0],
                [-s, 0, c, 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'z':
            return np.array([
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

    return transmat @ rot_mat_from_theta(theta, axis)

def add_scaling_to_transmat(transmat, sx, sy=None, sz=None):
    def mk_scaling(sx, sy, sz):
        return np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ])
    if isinstance(sx, (int, float)):
        assert sy is not None and sz is not None
        scaling = mk_scaling(sx, sy, sz)

    elif len(sx.shape) == 1:
        assert sz is None and sy is None
        sx, sy, sz = sx
        scaling = mk_scaling(sx, sy, sz)
    else:
        scaling = sx

    return transmat @ scaling




def rotate_point_cloud_y_axis(points, theta):
    """
    Rotate a point cloud around the y-axis by a given angle in degrees.

    Args:
        points: (N, 3) numpy array or jax array representing the point cloud.
        degrees: Rotation angle in degrees (positive for counter-clockwise rotation).

    Returns:
        Rotated point cloud as a jax array of shape (N, 3).
    """

    # Compute rotation matrix for y-axis
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    rotation_matrix = jnp.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    # Apply rotation
    rotated_points = jnp.dot(points, rotation_matrix.T)  # Transpose for correct multiplication

    rotated_points = center_point_cloud(rotated_points)

    return rotated_points

def rotate_point_cloud_z_axis(points, theta):
    """
    Rotate a point cloud around the z-axis by a given angle in degrees.

    Args:
        points: (N, 3) numpy array or jax array representing the point cloud.
        degrees: Rotation angle in degrees (positive for counter-clockwise rotation).

    Returns:
        Rotated point cloud as a jax array of shape (N, 3).
    """

    # Compute rotation matrix for z-axis
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    rotation_matrix = jnp.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    # Apply rotation
    rotated_points = jnp.dot(points, rotation_matrix.T)  # Transpose for correct multiplication

    rotated_points = center_point_cloud(rotated_points)

    return rotated_points


def pairwise_distances(points):
    """
    Compute pairwise Euclidean distances between 3D points.

    Args:
        points: Array of shape (n, 3) containing n 3D points

    Returns:
        Distance matrix of shape (n, n) where D[i,j] is the distance between points i and j
    """
    # Compute squared differences between all pairs of points
    diff = points[:, None, :] - points[None, :, :]  # Shape: (n, n, 3)
    squared_diffs = diff ** 2

    # Sum along the last dimension (x,y,z coordinates) and take square root
    distances = jnp.sqrt(jnp.sum(squared_diffs, axis=-1))

    return distances

def point2point_dist(p1, p2):
    return jnp.sqrt(jnp.sum((p2-p1)**2))

def get_dists_for_p1(p1, target_pcd):
    return jax.vmap(functools.partial(point2point_dist, p1))(target_pcd)

def get_mindist_for_p1(target_pcd, p1):
    ret = get_dists_for_p1(p1, target_pcd)#.min()
    return -jax.lax.top_k(-ret, 3)[0]

def get_loss(pcd_to_rotate, pcd_to_match, rad):
    pcd_to_rotate = rotate_point_cloud_y_axis(pcd_to_rotate, rad)

    # Distance from rotated to target
    mindists_rot_to_target = jax.vmap(functools.partial(get_mindist_for_p1, pcd_to_match))(pcd_to_rotate)

    # Distance from target to rotated (important for symmetry)
    mindists_target_to_rot = jax.vmap(functools.partial(get_mindist_for_p1, pcd_to_rotate))(pcd_to_match)

    return (mindists_rot_to_target.mean() + mindists_target_to_rot.mean()) / 2

def get_score_OLD(pcd_to_rotate, pcd_to_match, rad):
    pcd_to_rotate = rotate_point_cloud_z_axis(pcd_to_rotate, rad)
    mindists = jax.vmap(functools.partial(get_mindist_for_p1, pcd_to_match))(pcd_to_rotate)

    return mindists.mean()
from tqdm import tqdm


def center_point_cloud(points):
    """
    Center a point cloud so its centroid is at (0, 0, 0).

    Parameters:
    points : numpy.ndarray
        Array of shape (N, 3) containing 3D points

    Returns:
    numpy.ndarray
        Centered point cloud
    """
    # Calculate the centroid (mean along each axis)
    centroid = np.mean(points, axis=0)

    # Center the points by subtracting the centroid
    centered_points = points - centroid

    return centered_points

def pcd_or_mesh_to_np(pcd_or_mesh, NUM_POINTS_TO_KEEP=1000, mesh_keep_vertex_OR_sample_points="sample_points"):
    if isinstance(pcd_or_mesh, jnp.ndarray) or isinstance(pcd_or_mesh, tuple) or isinstance(pcd_or_mesh, list):
        return pcd_or_mesh_to_np(np.array(pcd_or_mesh))

    if isinstance(pcd_or_mesh, np.ndarray):
        if pcd_or_mesh.shape[0] > NUM_POINTS_TO_KEEP:
            pcd_or_mesh = pcd_or_mesh[np.random.choice(np.arange(pcd_or_mesh.shape[0]), size=NUM_POINTS_TO_KEEP, replace=False)]
        return center_point_cloud(pcd_or_mesh.copy())

    if isinstance(pcd_or_mesh, o3d.geometry.PointCloud):
        #pcd = pcd_or_mesh.voxel_down_sample(voxel_size=0.05)
        pcd = np.asarray(pcd_or_mesh.points)
        return pcd_or_mesh_to_np(pcd)

    if isinstance(pcd_or_mesh, Dict):
        if mesh_keep_vertex_OR_sample_points == "keep_vertex":

            newpcs = []
            for p in pcd_or_mesh["vertices"]:
                newpcs.append(np.array([p['x'], p['y'], p['z']]))
            pcd = np.vstack(newpcs)
            return pcd_or_mesh_to_np(pcd)

        else:
            from hippo.utils.spatial_utils import ai2thor_to_mesh
            mesh = ai2thor_to_mesh(pcd_or_mesh)
            pcd = mesh.sample_points_poisson_disk(number_of_points=NUM_POINTS_TO_KEEP)
            return pcd_or_mesh_to_np(pcd)


    if isinstance(pcd_or_mesh, o3d.geometry.TriangleMesh):
        return pcd_or_mesh_to_np(pcd_or_mesh.sample_points_uniformly(number_of_points=NUM_POINTS_TO_KEEP))

def global_align(pcd_to_rotate: np.array, pcd_to_match: np.array, get_best_score=False):
    trials = jnp.linspace(0,  2*np.pi, 100)

    BATCH_SIZE = 100
    BATCH_SIZE = min(BATCH_SIZE, len(trials))

    scores = []
    for b_trials in tqdm(trials.reshape(-1 ,BATCH_SIZE)):
        scores.append(jax.vmap(functools.partial(get_loss, pcd_to_rotate, pcd_to_match))(b_trials))
        #scores.append(get_score(pcd_to_rotate, pcd_to_match, trial))
    scores = jnp.concatenate(scores)
    #scores = jax.vmap(functools.partial(get_score, pcd_to_rotate, pcd_to_match))(trials)
    best_rot = scores.argmin()
    found_rot = trials[best_rot]

    if get_best_score:
        return found_rot, scores[best_rot]
    return found_rot

def swap_yz(pcd: np.array):
    return pcd[:,[0,2,1]]

DISABLE_VIS = False
def vis(xyz, disable=DISABLE_VIS):
    if disable:
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = v3v(xyz)
    o3d.visualization.draw_geometries([pcd])


def make_z_trans_init_matrix(theta):
    """
    Create a 4x4 homogeneous transformation matrix for z-axis rotation.

    Args:
        theta: Rotation angle in radians

    Returns:
        4x4 numpy array representing the transformation
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return np.array([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta, cos_theta, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def make_y_trans_init_matrix(theta):
    """
    Create a 4x4 homogeneous transformation matrix for y-axis rotation.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return np.array([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]
    ])

from scipy.spatial.transform import Rotation as R
def euler_to_matrix_4x4(x_rot, y_rot, z_rot, degrees: bool, order='xyz'):
    """
    Create a 4x4 transformation matrix from Euler angles.

    Parameters:
    - x_rot, y_rot, z_rot: Rotations around the X, Y, and Z axes
    - degrees: If True, interprets angles in degrees; otherwise in radians
    - order: Rotation order, default is 'xyz'

    Returns:
    - 4x4 numpy ndarray
    """
    # Create a rotation object from Euler angles
    rotation = R.from_euler(order, [x_rot, y_rot, z_rot], degrees=degrees)

    # Convert to 3x3 rotation matrix
    rot_matrix = rotation.as_matrix()

    # Create 4x4 homogeneous transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    return transform


def fine_tune(pcd_to_align, target_pcd, initial_zrot):
    #initted_pcd = rotate_point_cloud_z_axis(pcd_to_align, initial_zrot)
    trans_init = euler_to_matrix_4x4(0, initial_zrot, 0, degrees=False)

    pcd_to_align = transform_point_cloud(pcd_to_align, trans_init)
    _pcd_to_align = o3d.geometry.PointCloud()
    _pcd_to_align.points = v3v(pcd_to_align)
    pcd_to_align = _pcd_to_align

    _target_pcd = o3d.geometry.PointCloud()
    _target_pcd.points = v3v(target_pcd)
    target_pcd = _target_pcd

    fine_tuned = o3d.pipelines.registration.registration_icp(
        pcd_to_align, target_pcd, 0.02, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return fine_tuned.transformation




def transform_point_cloud(points, transformation_matrix):
    """
    Apply a 4x4 transformation matrix to a 3D point cloud.

    Args:
        points: Array of shape (N, 3) containing N 3D points
        transformation_matrix: 4x4 affine transformation matrix

    Returns:
        Transformed points of shape (N, 3)
    """
    vis(points, disable=True)

    # Convert points to homogeneous coordinates (N, 4)
    homogeneous_points = jnp.concatenate([
        points,
        jnp.ones((points.shape[0], 1))  # Add homogeneous coordinate of 1
    ], axis=1)

    # Apply transformation (N, 4) = (N, 4) @ (4, 4)
    transformed_points = homogeneous_points @ transformation_matrix.T

    # Convert back to Cartesian coordinates by dividing by w (N, 3)
    ret = transformed_points[:, :3] / transformed_points[:, 3:4]
    vis(ret, disable=True)
    return ret


import numpy as np
import math


def keep_only_y_rotation(matrix_4x4):
    """
    Takes a 4x4 transformation matrix and returns a new matrix with only the Y-axis rotation component.
    All other rotations (X and Z) are removed, while translation and scale are preserved.

    Args:
        matrix_4x4: 4x4 numpy array representing a transformation matrix

    Returns:
        4x4 numpy array with only Y-axis rotation
    """
    # Make sure we're working with a numpy array
    m = np.array(matrix_4x4, dtype=np.float64, copy=True)

    # Extract the 3x3 rotation part of the matrix
    rotation = m[:3, :3]

    # Calculate the Y rotation angle from the rotation matrix
    # Using arcsin of the -m[2][0] (sin(theta)) but with safety checks
    y_angle = math.atan2(rotation[2, 0], rotation[0, 0])

    # Rebuild a rotation matrix with only the Y rotation
    cy = math.cos(y_angle)
    sy = math.sin(y_angle)

    new_rotation = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])

    # Create the new 4x4 matrix
    result = np.identity(4)
    result[:3, :3] = new_rotation

    # Preserve the translation components
    result[:3, 3] = m[:3, 3]

    # Preserve the bottom row (typically [0, 0, 0, 1] for affine transforms)
    result[3, :] = m[3, :]

    return result

def np_to_pcd(nppcd, source_OR_target=None):
    if isinstance(nppcd, o3d.geometry.PointCloud):
        return nppcd

    pcd = o3d.geometry.PointCloud()
    pcd.points = v3v(nppcd)

    if source_OR_target is not None:
        if source_OR_target is True:
            pcd.paint_uniform_color([1, 0.706, 0])
        elif source_OR_target is False:
            pcd.paint_uniform_color([0, 0.651, 0.929])

    return pcd

def draw_registration_result(source, target, transformation=np.identity(4)):
    if not isinstance(transformation, np.ndarray):
        try:
            transformation = transformation.transformation
        except:
            pass

    source = copy.deepcopy(source)
    target = copy.deepcopy(target)
    if not isinstance(source, o3d.geometry.PointCloud):
        source = np_to_pcd(np.array(source), source_OR_target=True)
    if not isinstance(target, o3d.geometry.PointCloud):
        target = np_to_pcd(np.array(target), source_OR_target=False)

    source = source.paint_uniform_color([1, 0, 0])
    target = target.paint_uniform_color([0, 0, 1])

    source.transform(transformation)
    o3d.visualization.draw_geometries([source, target])#,
                                      #zoom=0.4559,
                                      #front=[0.6452, -0.3036, -0.7011],
                                      #lookat=[1.9892, 2.0208, 1.8945],
                                      #up=[-0.2779, -0.9482, 0.1556])

def execute_global_registration(pcd_to_align, pcd_to_match, voxel_size=0.05, init_rots=False):
    def preprocess_point_cloud(pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh



    def prepare_dataset(pcd_to_align, pcd_to_match, voxel_size):
        print(":: Load two point clouds and disturb initial pose.")
        #draw_registration_result(pcd_to_align, pcd_to_match)

        source_down, source_fpfh = preprocess_point_cloud(pcd_to_align, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(pcd_to_match, voxel_size)
        return pcd_to_align, pcd_to_match, source_down, target_down, source_fpfh, target_fpfh



    def slow_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result


    pcd_to_align = np_to_pcd(pcd_to_align)
    pcd_to_match = np_to_pcd(pcd_to_match)

    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)

    if isinstance(init_rots, np.ndarray) or isinstance(init_rots, list) or isinstance(init_rots, tuple):  # noqa
        pcd_to_align = np_to_pcd(transform_point_cloud(pcd_or_mesh_to_np(pcd_to_align), euler_to_matrix_4x4(*init_rots, degrees=True)))

    _, _, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pcd_to_align, pcd_to_match, voxel_size)

    result = slow_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    draw_registration_result(pcd_to_align, pcd_to_match, result)
    return np.array(result.transformation)

from sklearn.neighbors import KDTree

def extract_salient_features(pcd, k_neighbors=30, angle_threshold=np.pi / 4,
                             center_radius_ratio=0.1, include_extrema=True):
    """
    Extract salient features (edges/boundaries + extrema + center points) from a point cloud.

    Args:
        pcd: Open3D point cloud
        k_neighbors: Number of neighbors to consider for normal estimation
        angle_threshold: Angle threshold (in radians) to consider a point as edge
        center_radius_ratio: Ratio of bounding box diagonal to use as center region radius
        include_extrema: Whether to include min/max points in each dimension

    Returns:
        Open3D point cloud containing only salient features
    """
    # Convert to numpy array
    points = np.asarray(pcd.points)

    # Estimate normals if not already present
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(k_neighbors))
    normals = np.asarray(pcd.normals)

    # Use KDTree for efficient neighbor searches
    tree = KDTree(points)

    # For each point, find its neighbors and compute normal variation
    edge_indices = set()
    for i in range(len(points)):
        # Get k nearest neighbors (including the point itself)
        _, indices = tree.query(points[i].reshape(1, -1), k=k_neighbors)
        indices = indices[0]  # Remove extra dimension

        # Compute angles between current normal and neighbor normals
        angles = np.arccos(np.clip(np.dot(normals[indices], normals[i]), -1.0, 1.0))

        # If maximum angle exceeds threshold, consider this point salient
        if np.max(angles) > angle_threshold:
            edge_indices.add(i)

    # Add extrema points (min and max in each dimension)
    extrema_indices = set()
    if include_extrema:
        for dim in range(3):
            min_idx = np.argmin(points[:, dim])
            max_idx = np.argmax(points[:, dim])
            extrema_indices.add(min_idx)
            extrema_indices.add(max_idx)

    # Add points near the center
    center_indices = set()
    if center_radius_ratio > 0:
        # Compute bounding box and its diagonal
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        center = (min_pt + max_pt) / 2
        radius = center_radius_ratio * np.linalg.norm(max_pt - min_pt)

        # Find points within radius of center
        center_indices = set(tree.query_radius(center.reshape(1, -1), radius)[0])

    # Combine all indices
    all_salient_indices = list(edge_indices.union(extrema_indices).union(center_indices))

    # Extract salient points
    salient_pcd = pcd.select_by_index(all_salient_indices)

    return salient_pcd


import numpy as np

def rescale_point_cloud(source_pts, target_pts):

    # Method 2: Bounding box diagonal
    source_extent = np.linalg.norm(np.ptp(source_pts, axis=0))
    target_extent = np.linalg.norm(np.ptp(target_pts, axis=0))

    scale_factor = target_extent / source_extent
    rescaled_source = source_pts * scale_factor
    return rescaled_source


import open3d as o3d
import numpy as np

import open3d as o3d
import numpy as np


def curvature_based_downsample(pcd, k_neighbors=30, curvature_threshold=0.1):
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(k_neighbors))

    # Build KDTree for nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    curvatures = []
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    for i in range(len(points)):
        # Find k-nearest neighbors
        [k, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], k_neighbors)

        # Compute variance of normals as curvature proxy
        neighbor_normals = normals[idx]
        normal_variance = np.var(neighbor_normals, axis=0)
        curvature = np.linalg.norm(normal_variance)
        curvatures.append(curvature)

    curvatures = np.array(curvatures)

    # Select points with high curvature
    high_curvature_idx = np.where(curvatures > curvature_threshold)[0]
    return pcd.select_by_index(high_curvature_idx)


def edge_preserving_downsample(pcd, k_neighbors=30, angle_threshold=0.5):
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(k_neighbors))

    # Build KDTree
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    edge_points = []
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    for i in range(len(points)):
        # Find k-nearest neighbors
        [k, idx, _] = kdtree.search_knn_vector_3d(points[i], k_neighbors)

        # Check angle between normals
        for j in idx:
            if i == j:
                continue
            angle = np.dot(normals[i], normals[j]) / (np.linalg.norm(normals[i]) * np.linalg.norm(normals[j]))
            if np.arccos(np.clip(angle, -1, 1)) > angle_threshold:
                edge_points.append(i)
                break

    return pcd.select_by_index(list(set(edge_points)))


def feature_preserving_poisson_disk(pcd, min_density=0.5, k_neighbors=10):
    # Estimate normals and curvature
    pcd.estimate_normals()
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    curvatures = []
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    for i in range(len(points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(points[i], k_neighbors)
        neighbor_normals = normals[idx]
        normal_variance = np.var(neighbor_normals, axis=0)
        curvature = np.linalg.norm(normal_variance)
        curvatures.append(curvature)

    curvatures = np.array(curvatures)
    curvatures = (curvatures - curvatures.min()) / (curvatures.max() - curvatures.min())

    # Perform importance sampling
    probabilities = min_density + (1 - min_density) * curvatures
    random_values = np.random.random(len(points))
    selected_indices = np.where(random_values < probabilities)[0]

    return pcd.select_by_index(selected_indices)

def voxel_downsample(pcd, divisions=20):
    pcd = pcd_or_mesh_to_np(pcd)
    extent = np.linalg.norm(np.ptp(pcd, axis=0))

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(np_to_pcd(pcd),
                                                                voxel_size=extent / divisions)
    point_cloud_np = np.asarray(
        [voxel_grid.origin + pt.grid_index * voxel_grid.voxel_size for pt in voxel_grid.get_voxels()])
    return point_cloud_np


def remove_translation_from_transmat(matrix):
    """
    Remove translation components from a 4x4 transformation matrix.

    Args:
        matrix (np.ndarray): 4x4 transformation matrix

    Returns:
        np.ndarray: Matrix with translation removed
    """
    # Make a copy to avoid modifying the original
    result = matrix.copy()

    # Set translation components to zero (last column, top 3 elements)
    result[:3, 3] = 0

    return result



def try_rescue_planes(pcd_to_align, target_pcd):
    # rescues cases where a plane is generated flat on the ground, yet should be standing

    #from hippo.reconstruction.assetlookup.assetIsPlane import is_plane_object
    #if not (is_plane_object(target_pcd) and is_plane_object(pcd_to_align)):
    #    print("Is not a plane, skipping rescue attempt")
    #    return pcd_to_align, np.array([0,0,0])
    #print("Is a plane, attempting rescue")

    rots_to_try = [(0,0,0), (90, 0, 0), (0, 0, 90)] # , (90,0,90)]

    

    def try_rot(rot):
        try_pcd = transform_point_cloud(pcd_to_align, euler_to_matrix_4x4(*rot, degrees=True))
        loss = global_align(try_pcd, target_pcd, get_best_score=True)[1]
        return loss

    losses = np.array([try_rot(r) for r in rots_to_try])
    best_rot = losses.argmin()
    return transform_point_cloud(pcd_to_align, euler_to_matrix_4x4(*rots_to_try[best_rot], degrees=True)), np.array(rots_to_try[best_rot])



#from diskcache import FanoutCache, Cache
#CACHEPATH = "/".join(__file__.split("/")[:-1]) + "/diskcache"
#cache = Cache(CACHEPATH)
#@cache.memoize()
def align(pcd_to_align, spoof_rad=None, target_pcd=None, do_fine_tune=False, rough_scaling=True, downscale_using_voxel_divisions=25, do_global_tune=False, round_rot=None, is_planar_according_to_llm=None):
    if target_pcd is None:
        target_pcd = pcd_to_align
        assert spoof_rad is not None
    if spoof_rad is not None:
        pcd_to_align = rotate_point_cloud_y_axis(pcd_or_mesh_to_np(pcd), spoof_rad)

    pcd_to_align = pcd_or_mesh_to_np(pcd_to_align, mesh_keep_vertex_OR_sample_points="sample_points")
    target_pcd = pcd_or_mesh_to_np(target_pcd)



    DISABLE_VIS = True
    if not DISABLE_VIS:
        print("Original PCDs")
        draw_registration_result(pcd_to_align, target_pcd)

    if rough_scaling:
        pcd_to_align = rescale_point_cloud(pcd_to_align, target_pcd)
        #pcd_to_align = copy.deepcopy(pcd_to_align) * rough_scaling

        if not DISABLE_VIS:
            print("Rescaled source PCD")
            draw_registration_result(pcd_to_align, target_pcd)

    if downscale_using_voxel_divisions:
        pcd_to_align = voxel_downsample(pcd_to_align, divisions=downscale_using_voxel_divisions) #np.array(pcd_to_align.points)
        target_pcd = voxel_downsample(target_pcd, divisions=downscale_using_voxel_divisions) #np.array(target_pcd.points)

        pcd_to_align = pcd_or_mesh_to_np(np_to_pcd(pcd_to_align))   # quick recenter
        target_pcd = pcd_or_mesh_to_np(np_to_pcd(target_pcd))

        if not DISABLE_VIS:
            print("Downscaled to edges")
            draw_registration_result(pcd_to_align, target_pcd)

    if is_planar_according_to_llm:
        pcd_to_align, rescue_rots = try_rescue_planes(pcd_to_align, target_pcd) # these are degrees
    else:
        rescue_rots = np.zeros(3)
    found_rot = global_align(pcd_to_align, target_pcd)
    if not DISABLE_VIS:
        print("Gross rotation")
        draw_registration_result(rotate_point_cloud_y_axis(pcd_to_align, found_rot), target_pcd)

    if not do_global_tune and not do_fine_tune:
        def round_angle(angle, N):
            return N * round(angle / N)
        if round_rot is not None:
            found_rot = math.radians(round_angle(math.degrees(found_rot), round_rot))

        #offset = target_pcd.min(axis=0)[1] - rotate_point_cloud_y_axis(pcd_to_align, found_rot).min(axis=0)[1]
        transmat = euler_to_matrix_4x4(math.radians(rescue_rots[0]), math.radians(rescue_rots[1]) + found_rot, math.radians(rescue_rots[2]), degrees=False)
        #transmat[1,3] = offset

        if not DISABLE_VIS:
            print("Fix offset")
            draw_registration_result(transform_point_cloud(pcd_to_align, transmat), target_pcd)
        return (np.array([rescue_rots[0], rescue_rots[1] + math.degrees(found_rot), rescue_rots[2]]),
                transmat)

    init_rots = np.array([rescue_rots[0], rescue_rots[1] + math.degrees(found_rot), rescue_rots[2]]) # degrees
    #raise NotImplementedError("Need to handle rescue rots")


    if do_global_tune:
        transmat = execute_global_registration(pcd_to_align, target_pcd, voxel_size=0.05, init_rots=init_rots)
        init_rots = list(map(np.radians, init_rots))
        transmat = add_axis_rot_to_trans_mat(add_axis_rot_to_trans_mat(add_axis_rot_to_trans_mat(transmat, init_rots[0], axis='x'), init_rots[1], axis='y'), init_rots[2], axis='z')
        transmat = remove_translation_from_transmat(transmat)

        if not DISABLE_VIS:
            print("Global registration")
            #transform_point_cloud(pcd_to_align, transmat)
            draw_registration_result(transform_point_cloud(pcd_to_align, transmat), target_pcd)

        return transmat_to_euler(transmat, degrees=True), transmat

    if do_fine_tune:
        #print(make_y_trans_init_matrix(found_rot))

        tuned_transform = fine_tune(pcd_to_align, target_pcd, found_rot)


        #tuned_transform = keep_only_y_rotation(tuned_transform)
        #print(tuned_transform)

        vis(transform_point_cloud(pcd_to_align, tuned_transform), disable=DISABLE_VIS)

        if not DISABLE_VIS:
            print("Local registration")
            #transform_point_cloud(pcd_to_align, transmat)
            draw_registration_result(transform_point_cloud(pcd_to_align, tuned_transform), target_pcd)

        return np.array([0, transmat_to_euler(tuned_transform, degrees=True)[1], 0]), euler_to_matrix_4x4(0, transmat_to_euler(tuned_transform, degrees=False)[1], 0, degrees=False) #tuned_transform

    raise AssertionError("Why did the function not return? lol")


from scipy.spatial.transform import Rotation as Rscipy
def transmat_to_euler(trans_mat, degrees: bool):
    if trans_mat.shape == (4,4):
        trans_mat = trans_mat[:3,:3]

    r = Rscipy.from_matrix(trans_mat)
    euler_xyz = r.as_euler('xyz', degrees=degrees)  # or use 'zyx' if you prefer that order
    return euler_xyz

def euler_to_transmat(euler_angles, degrees=True):
    """
    Converts Euler angles to a 4x4 transformation matrix (rotation only).

    Parameters:
        euler_angles (array-like): The Euler angles in 'xyz' order.
        degrees (bool): True if the input angles are in degrees, False for radians.

    Returns:
        np.ndarray: A 4x4 transformation matrix with rotation applied.
    """
    r = Rscipy.from_euler('xyz', euler_angles, degrees=degrees)
    rot_mat = r.as_matrix()

    # Create 4x4 transformation matrix
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot_mat
    return trans_mat

if __name__ == "__main__":
    from hippo.conceptgraph.conceptgraph_intake import load_conceptgraph

    # 21
    #cg = load_conceptgraph("/home/charlie/Desktop/Holodeck/hippo/datasets/replica_room0_cg-detector_2025-04-04-18-03-58")
    #pcd = cg["segGroups"][21]['pcd']
    #pcd = pcd_or_mesh_to_np(pcd)
    #align(pcd, rad=np.pi/2)

    # 24
    mesh = "/home/charlie/TRELLIS/datasets/replica_room0_cg-detector_2025-04-04-18-03-58/segments/24/masked/convert/c036e0c8"
    from ai2thor.util.runtime_assets import load_existing_thor_asset_file
    obj = load_existing_thor_asset_file("/home/charlie/TRELLIS/datasets/replica_room0_cg-detector_2025-04-04-18-03-58/segments/24/masked/convert",
                                        f"c036e0c8/c036e0c8")  # load_existing_thor_asset_file(OBJATHOR_ASSETS_DIR, f"{assetId}/{assetId}")
    cg = load_conceptgraph(
        "/home/charlie/Desktop/Holodeck/hippo/datasets/replica_room0_cg-detector_2025-04-04-18-03-58")
    pcd = cg["segGroups"][24]['pcd']
    obj = swap_yz(pcd_or_mesh_to_np(obj))
    align(obj, spoof_rad=None, target_pcd=pcd)

