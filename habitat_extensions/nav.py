from typing import Any, List, Optional, Tuple

import math
import numpy as np

from habitat.core.embodied_task import (
    SimulatorTaskAction,
)
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


# @registry.register_task_action
# class MoveForwardByDistanceAction(SimulatorTaskAction):
#     def step(self, *args: Any, distance: float,**kwargs: Any):
#         r"""Update ``_metric``, this method is called from ``Env`` on each
#         ``step``.
#         """
#         original_amount = self._sim.get_agent(0).agent_config.action_space[1].actuation.amount
#         self._sim.get_agent(0).agent_config.action_space[1].actuation.amount = distance
#         output = self._sim.step(HabitatSimActions.MOVE_FORWARD)
#         self._sim.get_agent(0).agent_config.action_space[1].actuation.amount = original_amount
#         return output


@registry.register_task_action
class MoveHighToLowAction(SimulatorTaskAction):
    def step(self, *args: Any, 
            angle: float, distance: float,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        init_state = self._sim.get_agent_state()
        
        collision = False
        # left_action = HabitatSimActions.TURN_LEFT
        forward_action = HabitatSimActions.MOVE_FORWARD
        # init_left = self._sim.get_agent(0).agent_config.action_space[
        #     left_action].actuation.amount
        init_forward = self._sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        # self._sim.get_agent(0).agent_config.action_space[
        #     left_action].actuation.amount = angle * 180 / math.pi
        # output = self._sim.step(left_action)
        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        
        self._sim.set_agent_state(init_state.position, rotation)

        # self._sim.get_agent(0).agent_config.action_space[
        #     forward_action].actuation.amount = distance
        
        ksteps = int(distance//init_forward)
        for k in range(ksteps):
            if k == ksteps - 1:
                output = self._sim.step(forward_action)
            else:
                self._sim.step_without_obs(forward_action)
            if self._sim.previous_step_collided:
                collision = True
        
        # self._sim.get_agent(0).agent_config.action_space[
        #     left_action].actuation.amount = init_left
        # self._sim.get_agent(0).agent_config.action_space[
        #     forward_action].actuation.amount = init_forward

        output['collision'] = collision

        return output


@registry.register_task_action
class MoveHighToLowActionEval(SimulatorTaskAction):
    # 高级动作转为低级 记录 位置 朝向以及是否碰撞
    def step(self, *args: Any, 
            angle: float, distance: float,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        init_state = self._sim.get_agent_state()

        positions = []
        collisions = []
        headings = []
        # left_action = HabitatSimActions.TURN_LEFT
        forward_action = HabitatSimActions.MOVE_FORWARD
        # init_left = self._sim.get_agent(0).agent_config.action_space[
        #     left_action].actuation.amount
        init_forward = self._sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        # self._sim.get_agent(0).agent_config.action_space[
        #     left_action].actuation.amount = angle * 180 / math.pi
        # output = self._sim.step(left_action)
        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)

        self._sim.set_agent_state(init_state.position, rotation)
        # positions.append(init_state.position)
        # self._sim.get_agent(0).agent_config.action_space[
        #     forward_action].actuation.amount = distance

        ksteps = int(distance//init_forward)
        for k in range(ksteps):
            if k == ksteps - 1:
                output = self._sim.step(forward_action)
            else:
                self._sim.step_without_obs(forward_action)
            positions.append(self._sim.get_agent_state().position)
            collisions.append(self._sim.previous_step_collided)
            # 计算朝向
            heading_vector = quaternion_rotate_vector(
                self._sim.get_agent_state().rotation.inverse(), np.array([0, 0, -1])
            )
            heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
            headings.append(heading)

        # self._sim.get_agent(0).agent_config.action_space[
        #     left_action].actuation.amount = init_left
        # self._sim.get_agent(0).agent_config.action_space[
        #     forward_action].actuation.amount = init_forward
        output['positions'] = positions
        output['collisions'] = collisions
        output['headings'] = headings

        return output
