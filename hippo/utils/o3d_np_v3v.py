import numpy as np
import open3d as o3d

def v3v(points):
    points = np.ascontiguousarray(points, dtype=np.float64)
    return o3d.utility.Vector3dVector(points)

def v3iv(points):
    points = np.ascontiguousarray(points, dtype=np.float64)
    return o3d.utility.Vector3iVector(points)