import copy
import dataclasses
import functools

import networkx as nx
import numpy as np
import jax.numpy as jnp
import jax


from hippo.simulation.spatialutils.filter_positions import filter_reachable_positions, draw_grid_graph_2d


def astar(requested_start_node, requested_end_node, full_reachability_graph, runtime_container):
    filtered = filter_reachable_positions(full_reachability_graph, runtime_container)

    def dist(p1, p2):
        return jnp.linalg.norm(p1 - p2)

    all_nodes = jnp.array(filtered.nodes)
    def shunt(pos, universal_y):
        pos = jnp.array(pos)
        pos = pos.at[1].set(universal_y)

        dists = jnp.nan_to_num(jax.vmap(functools.partial(dist, p2=pos))(all_nodes), neginf=jnp.inf, nan=jnp.inf, posinf=jnp.inf)
        argret = dists.argmin()
        return full_reachability_graph.clean_convert(all_nodes[argret])

    universal_y = list(filtered.nodes)[0][1]
    start_node = shunt(requested_start_node, universal_y=universal_y)
    end_node = shunt(requested_end_node, universal_y=universal_y)

    assert start_node in full_reachability_graph
    assert end_node in full_reachability_graph

    path = nx.astar_path(filtered, start_node, end_node)[1:]

    DEBUG = False
    if DEBUG:
        draw_grid_graph_2d(filtered, path)

    for node in path:
        yield node

class AStar:
    def __init__(self, end_node, full_reachability_graph, runtime_container):
        self.end_node = end_node
        self.full_reachability_graph = copy.deepcopy(full_reachability_graph)
        self.runtime_container = runtime_container

        self.generated_nodes = []

    def generate_path(self, start_node):
        full_reachability_graph = copy.deepcopy(self.full_reachability_graph)
        full_reachability_graph.remove_nodes_from(self.generated_nodes)

        filtered = filter_reachable_positions(self.full_reachability_graph, self.runtime_container)

        def dist(p1, p2):
            return jnp.linalg.norm(p1 - p2)

        all_nodes = jnp.array(filtered.nodes)

        def shunt(pos):
            pos = jnp.array(pos)

            argret = jax.vmap(functools.partial(dist, p2=pos))(all_nodes).argmin()
            return tuple(np.array(all_nodes[argret]).tolist())

        start_node = shunt(start_node)
        end_node = shunt(self.end_node)

        path = nx.astar_path(filtered, start_node, end_node)
        for node in path:
            self.generated_nodes.append(node)
            yield node

def find_4_quadrants_closest_nodes_for_aabb(target_object_aabb, full_reachability_graph, runtime_container):
    # for each face of the bounding box
    # generate closest point in the reachability graph
    pass

def GoToObject(self, robot, dest_obj):
    # todo https://chat.deepseek.com/a/chat/s/411a780c-246e-4909-ac93-48ad5f66e14f
    print("Going to ", dest_obj)

    dest_obj_id = self._get_object_id(dest_obj)

    self.push_action(
        {
            'action': 'GoToObject_PreConditionCheck',
            'agent_id': self._get_robot_id(robot),
            'objectId': dest_obj_id
        }
    )

    from hippo.simulation.ai2thor_metadata_reader import get_object_position_from_controller
    dest_obj_pos = get_object_position_from_controller(self.controller, dest_obj_id)
    # dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']]

    dest_obj_aabb = {obj["objectId"]: obj["axisAlignedBoundingBox"] for obj in
                     self.controller.last_event.metadata["objects"]}.get(dest_obj_id, None)['size']
    dest_obj_aabb = (dest_obj_aabb['x'], dest_obj_aabb['y'], dest_obj_aabb['z'])

    def dist_to_goal(robot):
        # Robot position and half-size
        robot_center = get_robot_position_from_controller(self.controller, self._get_robot_id(robot))
        robot_center = (robot_center[0], robot_center[2])
        robot_center = np.array(robot_center)
        robot_half_size = np.array([0.3, 0.3])  # Replace with actual half-size

        # Object position and half-size
        obj_center = (dest_obj_pos[0], dest_obj_pos[2])
        obj_center = np.array(obj_center)
        obj_half_size = (dest_obj_aabb[0], dest_obj_aabb[2])
        obj_half_size = np.array(obj_half_size) / 2  # Replace with actual half-size

        # Calculate the distance between centers
        delta = robot_center - obj_center

        # Calculate the overlap in each dimension
        overlap_x = abs(delta[0]) - (robot_half_size[0] + obj_half_size[0])
        overlap_y = abs(delta[1]) - (robot_half_size[1] + obj_half_size[1])

        # If rectangles overlap, distance is negative (you might want to handle this case differently)
        if overlap_x < 0 and overlap_y < 0:
            return max(overlap_x, overlap_y)  # negative value indicates penetration depth
        elif overlap_x < 0:
            return overlap_y
        elif overlap_y < 0:
            return overlap_x
        else:
            return np.sqrt(overlap_x ** 2 + overlap_y ** 2)

    def dist_robot_2_node(robot, node):
        return np.linalg.norm(
            np.array(get_robot_position_from_controller(self.controller, self._get_robot_id(robot))) - np.array(
                node))

    def rotate_to_face_node(robot, node):
        # align the robot once goal is reached
        # compute angle between robot heading and object
        robot_location = self._get_robot_location_dict(robot)

        robot_object_vec = [node[0] - robot_location['x'],
                            node[2] - robot_location['z']]
        y_axis = [0, 1]
        unit_y = y_axis / np.linalg.norm(y_axis)
        unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)

        angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
        angle = 360 * angle / (2 * np.pi)
        angle = (angle + 360) % 360
        rot_angle = angle - robot_location['rotation']

        if rot_angle > 0:
            self.push_action({'action': 'RotateRight', 'degrees': abs(rot_angle),
                              'agent_id': self._get_robot_id(robot)})
        else:
            self.push_action({'action': 'RotateLeft', 'degrees': abs(rot_angle),
                              'agent_id': self._get_robot_id(robot)})

    goal_thresh = 0.75

    def get_generators():
        new_generators = []
        for ia, robot in enumerate(robots):
            if dist_to_goal(robot) > goal_thresh:
                from hippo.simulation.spatialutils.motion_planning import astar
                new_generators.append(
                    astar(get_robot_position_from_controller(self.controller, self._get_robot_id(robot)),
                          dest_obj_pos, self.full_reachability_graph,
                          self.current_object_container))
            else:
                new_generators.append(None)
        return new_generators

    generators = get_generators()

    def are_we_done():
        ALL_DONE = True
        for ia, robot in enumerate(robots):
            d = dist_to_goal(robot)
            print(f"Going to {dest_obj_id}, distance:", d)
            if not d < goal_thresh:
                ALL_DONE = False
        return ALL_DONE

    if are_we_done():
        print(f"Was going to {dest_obj_id}, but already next to object. Will only adjust camera.")
    else:
        while True:
            try:
                for ia, robot in enumerate(robots):
                    if generators[ia] is not None:
                        node = next(generators[ia])

                        moveMagnitude = dist_robot_2_node(robot, node)
                        rotate_to_face_node(robot, node)
                        self._lock_robot(robot)
                        self.push_action(
                            {'action': 'MoveAhead', 'moveMagnitude': moveMagnitude,
                             'agent_id': self._get_robot_id(robot)})
                        self._lock_robot(robot)
                        self._release_robot(robot)
            except StopIteration:
                assert are_we_done(), "Possible motion planning failure, fix the astar path planner"

    for ia, robot in enumerate(robots):
        rotate_to_face_node(robot, dest_obj_pos)

    def LookUpDownAtObject(robot, agent_id):
        # todo make this its own function and call it after every object interaction...
        robot_location = self._get_robot_location_dict(robot)
        dy = dest_obj_pos[1] - robot_location["y"]
        # Compute yaw rotation
        dx = dest_obj_pos[0] - robot_location["x"]
        dz = dest_obj_pos[2] - robot_location["z"]

        horizontal_dist = math.sqrt(dx ** 2 + dz ** 2)
        pitch = math.degrees(math.atan2(dy, horizontal_dist))

        # Adjust camera pitch
        current_horizon = robot_location["horizon"]
        if pitch > current_horizon:
            self.push_action({"action": "LookUp", "agent_id": agent_id})
        else:
            self.push_action({"action": "LookDown", "agent_id": agent_id})

    def get_dest_obj(agent_id):
        for obj in self.controller.last_event.events[agent_id].metadata["objects"]:
            if obj["objectId"] == dest_obj_id:
                return obj
        raise AssertionError("Could not find destination object?!")

    for ia, robot in enumerate(robots):
        NUM_TRIES = 0
        MAX_NUM_TRIES = 10
        while not get_dest_obj(self._get_robot_id(robot))["visible"]:
            self._lock_robot(robot)
            LookUpDownAtObject(robot, self._get_robot_id(robot))
            self._lock_robot(robot)
            self._release_robot(robot)
            if NUM_TRIES > MAX_NUM_TRIES:
                break

    for ia, robot in enumerate(robots):
        self.push_action(
            {
                'action': 'GoToObject_PostConditionCheck',
                'agent_id': self._get_robot_id(robot),
                'objectId': dest_obj_id
            }
        )
    print("Reached: ", dest_obj)