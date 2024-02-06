from typing import Any, Dict

import numpy as np
from gym import spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from habitat_extensions.shortest_path_follower import (
    ShortestPathFollowerCompat,
)
from habitat_extensions.task import VLNExtendedEpisode
import pickle
import gzip
import json

@registry.register_sensor(name="GlobalGPSSensor")
class GlobalGPSSensor(Sensor):
    r"""The agents current location in the global coordinate frame

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
                to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """

    cls_uuid: str = "globalgps"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._dimensionality,),
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, **kwargs: Any):
        return self._sim.get_agent_state().position.astype(np.float32)


@registry.register_sensor
class ShortestPathSensor(Sensor):
    r"""Sensor for observing the action to take that follows the shortest path
    to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    cls_uuid: str = "shortest_path_sensor"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        if config.USE_ORIGINAL_FOLLOWER:
            self.follower = ShortestPathFollowerCompat(
                sim, config.GOAL_RADIUS, return_one_hot=False
            )
            self.follower.mode = "geodesic_path"
        else:
            self.follower = ShortestPathFollower(
                sim, config.GOAL_RADIUS, return_one_hot=False
            )
        # self._sim = sim
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        return np.array(
            [
                best_action
                if best_action is not None
                else HabitatSimActions.STOP
            ]
        )


@registry.register_sensor
class VLNOracleProgressSensor(Sensor):
    r"""Sensor for observing how much progress has been made towards the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    cls_uuid: str = "progress"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        # TODO: what is the correct sensor type?
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )


        if "geodesic_distance" not in episode.info.keys():
            distance_from_start = self._sim.geodesic_distance(
                episode.start_position, episode.goals[0].position
            )
            episode.info["geodesic_distance"] = distance_from_start

        distance_from_start = episode.info["geodesic_distance"]

        progress =  (distance_from_start - distance_to_target) / distance_from_start

        return np.array(progress, dtype = np.float32)

@registry.register_sensor
class VLNOracleActionGeodesicSensor(Sensor):
    r"""Sensor for observing the optimal action to take. Does not rely on the shortest path to the Goal.
    Instead observes the next waypoint and the best action towards that based on the shortest path.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        gt_path = config.GT_PATH.format(split=config.SPLIT)
        with gzip.open(gt_path, "rt") as f:
            self.gt_waypoint_locations = json.load(f)
        
        self.is_sparse = getattr(config, "IS_SPARSE", True)
        self.num_inter_waypoints = getattr(config, "NUM_WAYPOINTS", 0)   # Set number of intermediate waypoints
        self.goal_radius = getattr(config, "GOAL_RADIUS", 0.5)

        super().__init__(config=config)
        self.follower = ShortestPathFollower(
            self._sim,
            # all goals can be navigated to within 0.5m.
            goal_radius=getattr(config, "GOAL_RADIUS", 0.5),
            return_one_hot=False,
        )
        self.follower.mode = "geodesic_path"
        self.reference_path_action = None
        self.next_way_action = 0
        self.episode_id_action = None

        self.possible_actions= [HabitatSimActions.MOVE_FORWARD, HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT]

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "vln_law_action_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        
        if self.num_inter_waypoints > 0:
            locs = self.gt_waypoint_locations[str(episode.episode_id)]["locations"] #episode.reference_path
            ep_path_length = self._sim.geodesic_distance(locs[0], episode.goals[0].position)
            
            way_locations = [locs[0]]
            count = 0
            dist = ep_path_length / (self.num_inter_waypoints+1)
            for way in locs[:-1]:
                d = self._sim.geodesic_distance(locs[0], way)
                if d >= dist:
                    way_locations.append(way)
                    if count >= (self.num_inter_waypoints-1):
                        break
                    count += 1
                    dist += ep_path_length / (self.num_inter_waypoints+1)

            way_locations.append(episode.goals[0].position)

        else:
            if self.is_sparse:
                # Sparse supervision of waypoints
                way_locations = episode.reference_path
            else:
                # Dense supervision of waypoints
                way_locations = self.gt_waypoint_locations[str(episode.episode_id)]["locations"]

        current_position = self._sim.get_agent_state().position.tolist()
        
        nearest_dist = float("inf")
        nearest_way = way_locations[-1]
        nearest_way_count = len(way_locations)-1
        # way locations 就是参考路径
        for ind, way in reversed(list(enumerate(way_locations))):
            distance_to_way = self._sim.geodesic_distance(
                current_position, way
            )
            if distance_to_way > self.goal_radius and distance_to_way < nearest_dist:
                dist_way_to_goal = self._sim.geodesic_distance(
                    way, episode.goals[0].position
                )
                dist_agent_to_goal = self._sim.geodesic_distance(
                    current_position, episode.goals[0].position
                )
                if dist_agent_to_goal > dist_way_to_goal:
                    nearest_dist = distance_to_way
                    nearest_way = way
                    nearest_way_count = ind

        best_action = self.follower.get_next_action(nearest_way)

        if best_action is None:
            while best_action is None:
                # if the agent has reached the current waypoint then update the next waypoint
                if nearest_way_count == (len(way_locations)-1):
                    return np.array([HabitatSimActions.STOP])
                
                nearest_way_count = nearest_way_count + 1
                nearest_way = way_locations[nearest_way_count]
                best_action = self.follower.get_next_action(nearest_way)
            
            return np.array([nearest_way])
            
        return np.array([nearest_way])

@registry.register_sensor
class RxRInstructionSensor(Sensor):

    cls_uuid: str = "rxr_instruction"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self.features_path = config.features_path
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(512, 768),
            dtype=np.float32,
        )

    def get_observation(
        self,
        observations: Dict[str, "Observations"],
        episode: VLNExtendedEpisode,
        **kwargs,
    ):
        features = np.load(
            self.features_path.format(
                split=episode.instruction.split,
                id=int(episode.instruction.instruction_id),
                lang=episode.instruction.language.split("-")[0],
            ),
        )
        feats = np.zeros((512, 768), dtype=np.float32)
        s = features["features"].shape
        feats[: s[0], : s[1]] = features["features"]
        return feats


@registry.register_sensor
class VLNMapWaypointSensor(Sensor):
    r"""Sensor for observing waypoint on MP3D map.
        返回路点的距离与转角
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    cls_uuid: str = "map_waypoint"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        with open(self._config.GRAPHS_FILE, "rb") as f:
            self._conn_graphs = pickle.load(f)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2, 5), # 不超过3m距离的点
            dtype=np.float32,
        )

    def get_observation(
        self,
        observations: Dict[str, "Observations"],
        episode: VLNExtendedEpisode,
        **kwargs,
    ):
        waypoints = np.zeros((13, 2), dtype=np.float32)
        self._scene_id = episode.scene_id.split("/")[-2]
        # 机器人当前位置
        agent_position = self._sim.get_agent_state().position
        agent_angle = self.get_polar_angle()
        scene_id = episode.scene_id.split("/")[-1].split(".")[0]
        
        # 距离agent最近的 MP3D START NODE 
        self._nearest_node = maps.get_nearest_node(
            self._conn_graphs[scene_id], np.take(agent_position, (0, 2))
        )
        # 该节点的位置(图)
        nn_position = self._conn_graphs[self._scene_id].nodes[
            self._nearest_node
        ]["position"]
        # 该节点的邻接信息
        nn_conj = self._conn_graphs[self._scene_id].adj[
            self._nearest_node
        ]
        i = 0
        for wp in nn_conj:
            dis = nn_conj[wp]["weight"] # 邻接的边长
            wp_pos = self._conn_graphs[self._scene_id].nodes[wp]["position"]
            r, phi = self.compute_dist_curpos2wp(agent_position, wp_pos, agent_angle)
            if(r < 3):
                waypoints[i][0] = r
                waypoints[i][1] = phi
                i += 1
        return waypoints 

    def compute_dist_curpos2wp(self, agent_pos, wp_pos, agent_angle):
        # 计算wp_pos距离agent_pos的距离及角度
        agent_pos = np.take(agent_pos, (0, 2))
        wp_pos = np.take(wp_pos, (0, 2))
        r = np.linalg.norm(
            np.array(agent_pos) - np.array(wp_pos), ord=2
        )
        wp_angle = np.arctan2(agent_pos[0] - wp_pos[0], agent_pos[1] - wp_pos[1])
        wp_angle = np.pi - wp_angle
        # 从polar坐标系 相对角 转为 agent坐标系相对角
        phi = (wp_angle - agent_angle + 2 * np.pi) % (2 * np.pi)
        
        return r, phi


    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        ) # 四元数 转 旋转向量

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip