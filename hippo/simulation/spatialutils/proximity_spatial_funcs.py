import jax.numpy as jnp
from jax import jit
import jax

def _bool_ifelse_elementwise(cond, iftrue, iffalse):
    return iftrue * cond + iffalse * (1-cond)

@jax.jit
def bool_ifelse(cond, iftrue, iffalse):
    cond = jnp.atleast_1d(cond)
    iftrue = jnp.atleast_1d(iftrue)
    iffalse = jnp.atleast_1d(iffalse)

    MAIN_SHAPE = cond.shape[0]

    if len(iffalse.shape) == 0:
        iffalse = jnp.ones_like(cond) * iffalse

    if len(iftrue.shape) == 0:
        iftrue = jnp.ones_like(cond) * iftrue

    if iffalse.shape[0] != MAIN_SHAPE:
        iffalse = jnp.repeat(iffalse[None], MAIN_SHAPE, axis=0)
    if iftrue.shape[0] != MAIN_SHAPE:
        iftrue = jnp.repeat(iftrue[None], MAIN_SHAPE, axis=0)

    cond = cond.astype(int)

    return jax.vmap(_bool_ifelse_elementwise)(cond, iftrue, iffalse)

def isOnTop_OLD(pos1, size1, pos2, size2, tol_overlap=0.5, tol_dist=0.1):
    """Returns True if obj1 is on top of obj2 within given tolerances."""
    # Compute XY overlap as the intersection over the area of obj1
    overlap_x = jnp.maximum(0, jnp.minimum(pos1[0] + size1[0] / 2, pos2[0] + size2[0] / 2) -
                jnp.maximum(pos1[0] - size1[0] / 2, pos2[0] - size2[0] / 2))
    overlap_y = jnp.maximum(0, jnp.minimum(pos1[1] + size1[1] / 2, pos2[1] + size2[1] / 2) -
                jnp.maximum(pos1[1] - size1[1] / 2, pos2[1] - size2[1] / 2))
    overlap_area = overlap_x * overlap_y
    obj1_area = size1[0] * size1[1]

    xy_overlap_ratio = bool_ifelse(obj1_area>0, overlap_area / obj1_area, 0)

    # Check if obj1's bottom is close to obj2's top
    top_z_obj2 = pos2[2] + size2[2] / 2
    bottom_z_obj1 = pos1[2] - size1[2] / 2
    z_close = jnp.abs(bottom_z_obj1 - top_z_obj2) <= tol_dist

    return (xy_overlap_ratio >= tol_overlap) & z_close


def isOnTop(pos1, size1, pos2, size2, tol_overlap=0.1, tol_dist=0.2):
    """Returns True if obj1 is on top of obj2 within given tolerances.

    Args:
        pos1: (x,y,z) position of obj1 (center coordinates)
        size1: (dx,dy,dz) size of obj1 (full widths/extents)
        pos2: (x,y,z) position of obj2 (center coordinates)
        size2: (dx,dy,dz) size of obj2 (full widths/extents)
        tol_overlap: minimum required XY overlap ratio (of obj1's area)
        tol_dist: maximum allowed vertical gap between objects

    Returns:
        True if obj1 is on top of obj2 according to tolerances
    """
    # Compute XY overlap as the intersection over the area of obj1
    overlap_x = jnp.maximum(0, jnp.minimum(pos1[0] + size1[0] / 2, pos2[0] + size2[0] / 2) -
                            jnp.maximum(pos1[0] - size1[0] / 2, pos2[0] - size2[0] / 2))
    overlap_y = jnp.maximum(0, jnp.minimum(pos1[1] + size1[1] / 2, pos2[1] + size2[1] / 2) -
                            jnp.maximum(pos1[1] - size1[1] / 2, pos2[1] - size2[1] / 2))
    overlap_area = overlap_x * overlap_y
    obj1_area = size1[0] * size1[1]
    #obj2_area = size2[0] * size2[1]

    # Use minimum of both areas for more robust overlap checking
    xy_overlap_ratio = overlap_area / obj1_area # jnp.maximum(1e-6, jnp.minimum(obj1_area, obj2_area))

    # Check vertical relationship
    top_z_obj2 = pos2[2] + size2[2] / 2
    bottom_z_obj1 = pos1[2] - size1[2] / 2
    z_close = (bottom_z_obj1 >= top_z_obj2 - tol_dist) & (bottom_z_obj1 <= top_z_obj2 + tol_dist)

    # Check if obj1 is above obj2 (using centers is not ideal, but better than nothing)
    is_above = bottom_z_obj1 > pos2[2] - size2[2] / 2  # obj1's bottom is above obj2's middle

    return (xy_overlap_ratio >= tol_overlap) & z_close & is_above

def isInside(pos1, size1, pos2, size2, tol=1e-6):
    """
    Strict containment check if obj1 is completely inside obj2.

    Args:
        pos1: (x,y,z) position of obj1 (center coordinates)
        size1: (dx,dy,dz) size of obj1 (full widths/extents)
        pos2: (x,y,z) position of obj2 (center coordinates)
        size2: (dx,dy,dz) size of obj2 (full widths/extents)
        tol: tolerance for boundary comparisons

    Returns:
        True if obj1 is completely inside obj2 (with tolerance), False otherwise
    """
    # Calculate the min and max bounds for each object
    obj1_min = pos1 - size1 / 2
    obj1_max = pos1 + size1 / 2
    obj2_min = pos2 - size2 / 2
    obj2_max = pos2 + size2 / 2

    # Check if all of obj1's bounds are within obj2's bounds with tolerance
    return (jnp.all(obj1_min >= obj2_min - tol) & jnp.all(obj1_max <= obj2_max + tol))

def isInside2(pos1, size1, pos2, size2, tol_overlap=0.99,  tol_dist=None):
    """Returns True if obj1 is inside obj2 within the overlap tolerance."""
    # Compute XYZ overlap volume
    overlap_x = jnp.maximum(0, jnp.minimum(pos1[0] + size1[0] / 2, pos2[0] + size2[0] / 2) -
                jnp.maximum(pos1[0] - size1[0] / 2, pos2[0] - size2[0] / 2))
    overlap_y = jnp.maximum(0, jnp.minimum(pos1[1] + size1[1] / 2, pos2[1] + size2[1] / 2) -
                jnp.maximum(pos1[1] - size1[1] / 2, pos2[1] - size2[1] / 2))
    overlap_z = jnp.maximum(0, jnp.minimum(pos1[2] + size1[2] / 2, pos2[2] + size2[2] / 2) -
                jnp.maximum(pos1[2] - size1[2] / 2, pos2[2] - size2[2] / 2))

    overlap_volume = overlap_x * overlap_y * overlap_z
    obj1_volume = size1[0] * size1[1] * size1[2]

    overlap_ratio = bool_ifelse(obj1_volume > 0, overlap_volume / obj1_volume, 0)

    return overlap_ratio >= tol_overlap


def isInside3(pos1, size1, pos2, size2, tol_overlap=0.99, tol_dist=None):
    """Returns True if obj1 is inside obj2 within the overlap tolerance."""


    # Compute XYZ overlap ranges
    min1_x, max1_x = pos1[0] - size1[0] / 2, pos1[0] + size1[0] / 2
    min2_x, max2_x = pos2[0] - size2[0] / 2, pos2[0] + size2[0] / 2
    overlap_x = jnp.maximum(0, jnp.minimum(max1_x, max2_x) - jnp.maximum(min1_x, min2_x))

    min1_y, max1_y = pos1[1] - size1[1] / 2, pos1[1] + size1[1] / 2
    min2_y, max2_y = pos2[1] - size2[1] / 2, pos2[1] + size2[1] / 2
    overlap_y = jnp.maximum(0, jnp.minimum(max1_y, max2_y) - jnp.maximum(min1_y, min2_y))

    min1_z, max1_z = pos1[2] - size1[2] / 2, pos1[2] + size1[2] / 2
    min2_z, max2_z = pos2[2] - size2[2] / 2, pos2[2] + size2[2] / 2
    overlap_z = jnp.maximum(0, jnp.minimum(max1_z, max2_z) - jnp.maximum(min1_z, min2_z))

    # Calculate volumes
    overlap_volume = overlap_x * overlap_y * overlap_z
    obj1_volume = size1[0] * size1[1] * size1[2]

    # Handle zero volume case
    if tol_dist is not None:
        # Alternative distance-based check could go here
        pass

    return (obj1_volume > 0) & (overlap_volume / obj1_volume >= tol_overlap)


def isInside_OLD(pos1, size1, pos2, size2):
    """
    Check if obj1 is completely inside obj2.

    Args:
        pos1: position of obj1 (center coordinates)
        size1: size of obj1 (half-widths/extents)
        pos2: position of obj2 (center coordinates)
        size2: size of obj2 (half-widths/extents)

    Returns:
        True if obj1 is completely inside obj2, False otherwise
    """
    # Calculate the min and max bounds for each object
    obj1_min = pos1 - size1
    obj1_max = pos1 + size1
    obj2_min = pos2 - size2
    obj2_max = pos2 + size2

    # Check if all of obj1's bounds are within obj2's bounds
    return jnp.all(obj1_min >= obj2_min) & jnp.all(obj1_max <= obj2_max)


def isBeside(pos1, size1, pos2, size2, tol_dist=0.5):
    """Check if objects overlap in one axis and are close in the other."""
    same_surface = jnp.abs((pos1[2] - size1[2] / 2) - (pos2[2] - size2[2] / 2)) <= (tol_dist / 4)

    # Overlap in X and proximity in Y (or vice versa)
    overlap_x = (jnp.abs(pos1[0] - pos2[0]) <= (size1[0] + size2[0]) / 2)
    proximity_y = (jnp.abs(pos1[1] - pos2[1]) <= (size1[1] + size2[1]) / 2 + tol_dist)

    # Overlap in Y and proximity in X (or vice versa)
    overlap_y = (jnp.abs(pos1[1] - pos2[1]) <= (size1[1] + size2[1]) / 2)
    proximity_x = (jnp.abs(pos1[0] - pos2[0]) <= (size1[0] + size2[0]) / 2 + tol_dist)

    return same_surface & ((overlap_x & proximity_y) | (overlap_y & proximity_x))

def isNear(pos1, size1, pos2, size2,tol_overlap=None, tol_dist=0.5):
    """Returns True if obj1 is near obj2 within the distance tolerance."""
    distance = jnp.linalg.norm(pos1 - pos2)
    return distance <= tol_dist

def distance(pos1, size1, pos2, size2, tol_overlap=None, tol_dist=None):
    """Returns the Euclidean distance between obj1 and obj2."""
    return jnp.linalg.norm(pos1 - pos2)

def bbox_dist(pos1, size1, pos2, size2, tol_overlap=None, tol_dist=None):
    """
    Compute the minimal distance between two axis-aligned bounding boxes.

    Args:
        pos1: Center position of first box (shape: [d])
        size1: Size of first box (shape: [d])
        pos2: Center position of second box (shape: [d])
        size2: Size of second box (shape: [d])
        tol_overlap: If boxes overlap by more than this, return 0 (optional)
        tol_dist: If boxes are separated by more than this, return this value (optional)

    Returns:
        The minimal distance between the boxes, with optional thresholds applied
    """
    # Calculate the distance between centers
    delta = jnp.abs(pos1 - pos2)

    size1 = size1 / 2
    size2 = size2 / 2

    # Calculate the sum of half-extents
    sum_half_extents = size1 + size2

    # Compute separation in each dimension
    separation = delta - sum_half_extents

    # The minimal distance is the maximum separation across dimensions
    # (negative values indicate overlap)
    max_separation = jnp.max(separation)

    # If boxes overlap in all dimensions, the minimal distance is negative
    # (we take the maximum separation as the penetration depth)
    min_dist = jnp.where(max_separation < 0,
                         max_separation,  # penetration depth (negative)
                         jnp.linalg.norm(jnp.maximum(separation, 0)))  # positive distance

    # Apply tolerance for overlap if provided
    if tol_overlap is not None:
        min_dist = jnp.where(min_dist < -tol_overlap, 0.0, min_dist)

    # Apply tolerance for distance if provided
    if tol_dist is not None:
        min_dist = jnp.where(min_dist > tol_dist, tol_dist, min_dist)

    return min_dist