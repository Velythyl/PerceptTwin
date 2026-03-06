import jax
import jax.numpy as jnp
import numpy as np
import open3d as o3d

def ask_llm_if_plane(label):
    from llmqueries.llm import LLM
    prompt = f"""
    I need your help to detect "planar" objects. If the object is planar, please output OBJECT_IS_PLANAR anywhere in your answer. Otherwise, output OBJECT_IS_NOT_PLANAR.

    Keep your answers brief.

    Examples of planar objects: door, photograph, window, ceiling, etc.
    Examples of non-planar objects: sofa, armchair, table, cylindrical objects, etc.

    ---

    Now consider this object: {label}
            """
    _, resp = LLM(prompt=prompt.strip(), gpt_version="gpt-4.1-mini-2025-04-14")

    is_planar = "OBJECT_IS_PLANAR" in resp
    return is_planar

def is_single_plane(points, noise_threshold=0.05, normal_spread_threshold=0.1, min_points=10, k_normals=20):
    """
    Check if a point cloud is approximately a single plane, with curvature rejection.

    Args:
        points: (N, 3) array of 3D points or Open3D point cloud
        noise_threshold: PCA eigenvalue ratio threshold (lower = more tolerant)
        normal_spread_threshold: Std deviation of normals magnitude to reject curved surfaces
        min_points: Minimum number of points required for plane detection
        k_normals: Nearest neighbors for normal estimation

    Returns:
        is_plane: bool
        plane_params: (4,) array of plane equation coefficients
        ratio: smallest eigenvalue ratio
        normal_spread: std deviation of normals
    """
    if isinstance(points, o3d.geometry.PointCloud):
        o3d_pc = points
        points = np.asarray(points.points)
    else:
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(np.asarray(points))

    if len(points) < min_points:
        return False, jnp.array([0., 0., 0., 0.]), None, None

    # Center the points
    centroid = jnp.mean(points, axis=0)
    centered = points - centroid

    # Compute covariance matrix
    cov = jnp.cov(centered, rowvar=False)

    # PCA via SVD
    with jax.default_device(jax.devices('cpu')[0]):
        _, s, vh = jnp.linalg.svd(cov)
    eigenvalues = s
    eigenvectors = vh.T

    # Smallest eigenvalue ratio
    ratio = float(eigenvalues[2] / (eigenvalues.sum() + 1e-10))
    is_plane_pca = ratio < noise_threshold

    # Plane equation from PCA
    normal = eigenvectors[:, 2]
    d = -jnp.dot(normal, centroid)
    plane_params = jnp.concatenate([normal, jnp.array([d])])

    # --- NEW: normal vector consistency check ---
    o3d_pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=min(k_normals, len(points)-1))
    )
    normals = np.asarray(o3d_pc.normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # normalize
    # Mean angle deviation from average normal
    mean_normal = normals.mean(axis=0)
    mean_normal /= np.linalg.norm(mean_normal)
    deviations = np.linalg.norm(normals - mean_normal, axis=1)
    normal_spread = float(np.std(deviations))

    is_plane_normals = normal_spread < normal_spread_threshold

    # Final decision: must pass both tests
    is_plane = is_plane_pca and is_plane_normals

    return is_plane, plane_params, ratio, normal_spread


# JIT compile for better performance
#is_single_plane_jitted = jit(is_single_plane)

# Example usage:
if __name__ == "__main__":
    # Generate test data - a plane with some noise
    key = jax.random.PRNGKey(0)
    num_points = 1000

    # Create a plane in xy direction with some noise
    plane_points = jax.random.uniform(key, (num_points, 2), minval=-1, maxval=1)
    z_noise = jax.random.normal(key, (num_points,)) * 0.02  # Small noise
    points = jnp.column_stack([plane_points, z_noise])

    # Check if it's a plane
    is_plane, plane_eq, _ = is_single_plane(points)
    print(f"Is plane: {is_plane}")
    print(f"Plane equation: {plane_eq}")

    # Create non-planar data (a cube)
    cube_points = jax.random.uniform(key, (num_points, 3), minval=-1, maxval=1)
    is_plane_cube, _ = is_single_plane(cube_points)
    print(f"Cube is plane: {is_plane_cube}")

    # Create non-planar data (a sphere)
    # Generate random points on a sphere surface
    key, subkey = jax.random.split(key)
    sphere_points = jax.random.normal(subkey, (num_points, 3))
    sphere_points = sphere_points / jnp.linalg.norm(sphere_points, axis=1, keepdims=True)
    # Add some noise to make it more realistic
    key, subkey = jax.random.split(key)
    sphere_points += jax.random.normal(subkey, (num_points, 3)) * 0.01
    is_plane_sphere, _ = is_single_plane(sphere_points)
    print(f"Sphere is plane: {is_plane_sphere}")