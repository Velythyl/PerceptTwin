from typing import Dict, Any

import ai2thor.controller
from shapely import Polygon
from shapely.ops import triangulate


def get_rooms_polymap(house: Dict[str, Any]):
    room_poly_map = {}

    # NOTE: Map the rooms
    for i, room in enumerate(house["rooms"]):
        room_poly_map[room["id"]] = Polygon(
            [(p["x"], p["z"]) for p in room["floorPolygon"]]
        )

    return room_poly_map


def get_candidate_points_in_room(
    room_id: str,
    room_poly_map: Dict[str, Polygon],
):
    polygon = room_poly_map[room_id]

    room_triangles = triangulate(polygon)

    candidate_points = [
        ((t.centroid.x, t.centroid.y), t.area) for t in room_triangles  # type:ignore
    ]

    # We sort the triangles by size so we try to go to the center of the largest triangle first
    candidate_points.sort(key=lambda x: x[1], reverse=True)
    candidate_points = [p[0] for p in candidate_points]

    # The centroid of the whole room polygon need not be in the room when the room is concave. If it is,
    # let's make it the first point we try to navigate to.
    if polygon.contains(polygon.centroid):
        candidate_points.insert(0, (polygon.centroid.x, polygon.centroid.y))

    return candidate_points

def try_find_collision_free_starting_position(
    house: Dict[str, Any],
    controller: ai2thor.controller.Controller,
    room_poly_map: Dict[str, Polygon],
):
    teleport_success = False
    for room_id in sorted(room_poly_map.keys()):
        candidate_points = get_candidate_points_in_room(room_id, room_poly_map)
        for cand in candidate_points:
            event = controller.step(
                action="TeleportFull",
                position={
                    "x": float(cand[0]),
                    "y": house["metadata"]["agent"]["position"]["y"],
                    "z": float(cand[1]),
                },
                rotation=house["metadata"]["agent"]["rotation"],
                standing=True,
                horizon=30,
                forceAction=True,
            )
            print(event.metadata["errorMessage"])
            if event:
                teleport_success = True
                break

        if teleport_success:
            break

    if teleport_success:
        return True
    else:
        return False

def get_all_reachable_positions(
    house: Dict[str, Any],
    controller: ai2thor.controller.Controller,
    room_poly_map: Dict[str, Polygon],
):
    reachable_positions = []


    teleport_success = False
    for room_id in sorted(room_poly_map.keys()):
        candidate_points = get_candidate_points_in_room(room_id, room_poly_map)
        for cand in candidate_points:
            event = controller.step(
                action="TeleportFull",
                position={
                    "x": float(cand[0]),
                    "y": house["metadata"]["agent"]["position"]["y"],
                    "z": float(cand[1]),
                },
                rotation=house["metadata"]["agent"]["rotation"],
                standing=True,
                horizon=30,
                forceAction=True,
            )
            print(event.metadata["errorMessage"])
            if event:
                teleport_success = True
                reachable_positions.append(cand)
                break

        if teleport_success:
            break

    return reachable_positions

    if teleport_success:
        return True
    else:
        return False