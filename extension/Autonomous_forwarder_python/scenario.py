# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import carb.eventdispatcher
import numpy as np
import random as r
import carb
import asyncio
import omni
import time
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.articulations import Articulation, ArticulationView
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils import distance_metrics
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.motion_generation import ArticulationMotionPolicy, RmpFlow
from omni.isaac.motion_generation.interface_config_loader import load_supported_motion_policy_config
from omni.isaac.core.objects import FixedCuboid
from pxr import PhysxSchema
from omni.isaac.dynamic_control import _dynamic_control
import omni.kit.commands
from pxr import Gf
import omni.replicator.core as rep


class AutonomousForwarder:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self._articulation = None
        self._target = None

        self._script_generator = None

        self._joint_indices = None
        self._stage = None
        self.dc = None

        self._goal_blocks = []

        # pid stuff
        # Constants for PID tuning
        self.Kp = 1.5
        self.Ki = 0.00001
        self.Kd = 0.0001
        
        # Setpoint for desired turning angle
        self.setpoint = 0
        
        # Create PID controller
        self.pid = PIDController(self.Kp, self.Ki, self.Kd, self.setpoint)

        #lidar
        self.sensor = None
        #test lidar
        self.lidarPath = "/front/front_lidar/front_lidar"
        print("Init")

    def load_assets(self):
        """Load assets onto the stage and return them so they can be registered with the
        core.World.

        This function is called from ui_builder._setup_scene()

        The position in which things are loaded is also the position to which
        they will be returned on reset.
        """
        # Fowarder
        robot_prim_path = "/forwarder"
        stage_utils.add_reference_to_stage(usd_path="H:/autonomous_forwarder/dev_forwarder.usd", prim_path=robot_prim_path)
        self._articulation = Articulation(prim_path="/forwarder/front", name="forwarder")

        # Terrain cube
        terrain_prim_path = "/World/terrain"
        FixedCuboid(terrain_prim_path, scale=[200,200,0.1])
        
        # setting up lidars for the forwarder
        lidarPath = "/front/front_lidar/front_lidar"

        # ! would be better to use the rtx lidar but idk how to get the ouput from that shit
        """result, lidarPrim = omni.kit.commands.execute(
                    "RangeSensorCreateLidar",
                    path=lidarPath,
                    parent=robot_prim_path,
                    min_range=0.4,
                    max_range=100.0,
                    draw_points=True,
                    draw_lines=True,
                    horizontal_fov=180.0,
                    vertical_fov=10.0,
                    horizontal_resolution=0.4,
                    vertical_resolution=4.0,
                    rotation_rate=0,
                    high_lod=True,
                    yaw_offset=0.0,
                    enable_semantics=False
                )"""
        

        # 1. Create The Camera
        _, self.sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=lidarPath,
            parent=robot_prim_path,
            config="OS0_128ch20hz1024res"
        )
        # 2. Create and Attach a render product to the camera
        render_product = rep.create.render_product(self.sensor.GetPath(), [1, 1])

        # 3. Create a Replicator Writer that "writes" points into the scene for debug viewing
        writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
        writer.attach(render_product)

        # 4. Create Annotator to read the data from with annotator.get_data()
        annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
        annotator.attach(render_product)

        self._goal_block = FixedCuboid(
            name="goal_cube",
            position=np.array([2.5, -10, 0.1]),
            prim_path="/World/goal_cube",
            size=0.1,
            color=np.array([1, 0, 0]),
        )
        self._goal_blocks.append(self._goal_block)

        self._goal_block2 = FixedCuboid(
            name="goal_cube2",
            position=np.array([-2.5, -30, 0.1]),
            prim_path="/World/goal_cube2",
            size=0.1,
            color=np.array([1, 0, 0]),
        )
        self._goal_blocks.append(self._goal_block2)

        self._goal_block3 = FixedCuboid(
            name="goal_cube3",
            position=np.array([2.5, -50, 0.1]),
            prim_path="/World/goal_cube3",
            size=0.1,
            color=np.array([1, 0, 0]),
        )
        self._goal_blocks.append(self._goal_block3)

        self._goal_block4 = FixedCuboid(
            name="goal_cube4",
            position=np.array([-2.5, -70, 0.1]),
            prim_path="/World/goal_cube4",
            size=0.1,
            color=np.array([1, 0, 0]),
        )
        self._goal_blocks.append(self._goal_block4)

        """
        # ? Might be usefull for obsticle avoidance
        self._obstacles = [
            FixedCuboid(
                name="ob1",
                prim_path="/World/obstacle_1",
                scale=np.array([0.03, 1.0, 0.3]),
                position=np.array([0.25, 0.25, 0.15]),
                color=np.array([0.0, 0.0, 1.0]),
            ),
            FixedCuboid(
                name="ob2",
                prim_path="/World/obstacle_2",
                scale=np.array([0.5, 0.03, 0.3]),
                position=np.array([0.5, 0.25, 0.15]),
                color=np.array([0.0, 0.0, 1.0]),
            ),
        ]"""


        self.dc=_dynamic_control.acquire_dynamic_control_interface()
        # Return assets that were added to the stage so that they can be registered with the core.World
        return True

    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """
        # Set a camera view that looks good
        set_camera_view(eye=[0, 1, 210], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

        # Create a script generator to execute my_script().
        self._script_generator = self.my_script()

    def reset(self):
        """
        This function is called when the reset button is pressed.
        In this example the core.World takes care of all necessary resetting
        by putting everything back in the position it was in when loaded.

        In more complicated scripts, e.g. scripts that modify or create USD properties
        or attributes at runtime, the user will need to implement necessary resetting
        behavior to ensure their script runs deterministically.
        """
        # Start the script over by recreating the generator.
        self._script_generator = self.my_script()

    """
    The following two functions demonstrate the mechanics of running code in a script-like way
    from a UI-based extension.  This takes advantage of Python's yield/generator framework.  

    The update() function is tied to a physics subscription, which means that it will be called
    one time on every physics step (usually 60 frames per second).  Each time it is called, it
    queries the script generator using next().  This makes the script generator execute until it hits
    a yield().  In this case, no value need be yielded.  This behavior can be nested into subroutines
    using the "yield from" keywords.
    """

    def update(self, step: float):
        try:
            result = next(self._script_generator)
        except StopIteration:
            return True

    def my_script(self):
        # driving to pos
        for end in range(len(self._goal_blocks)):
            yield from self.drive_to_pos(6, end)

        #yield from self.get_lidar_param()
        
        #yield from self.drive_to_pos(5, 0)
        """yield from self.drive_to_pos(5, 1)
        yield from self.drive_to_pos(5, 2)
        yield from self.drive_to_pos(5, 3)"""
       
        """# Notice that subroutines can still use return statements to exit.  goto_position() returns a boolean to indicate success.
        success = yield from self.goto_position(
            translation_target, orientation_target, self._articulation, self._rmpflow, timeout=200
        )

        if not success:
            print("Could not reach target position")
            return

        yield from self.open_gripper_franka(self._articulation)

        # Visualize the new target.
        lower_translation_target = np.array([0.4, 0, 0.04])
        self._target.set_world_pose(lower_translation_target, orientation_target)

        success = yield from self.goto_position(
            lower_translation_target, orientation_target, self._articulation, self._rmpflow, timeout=250
        )

        yield from self.close_gripper_franka(self._articulation, close_position=np.array([0.02, 0.02]), atol=0.006)

        high_translation_target = np.array([0.4, 0, 0.4])
        self._target.set_world_pose(high_translation_target, orientation_target)

        success = yield from self.goto_position(
            high_translation_target, orientation_target, self._articulation, self._rmpflow, timeout=200
        )

        next_translation_target = np.array([0.4, 0.4, 0.4])
        self._target.set_world_pose(next_translation_target, orientation_target)

        success = yield from self.goto_position(
            next_translation_target, orientation_target, self._articulation, self._rmpflow, timeout=200
        )

        next_translation_target = np.array([0.4, 0.4, 0.25])
        self._target.set_world_pose(next_translation_target, orientation_target)

        success = yield from self.goto_position(
            next_translation_target, orientation_target, self._articulation, self._rmpflow, timeout=200
        )

        yield from self.open_gripper_franka(self._articulation)"""
        print("my_script")

    ################################### Functions

    def goto_position(
        self,
        translation_target,
        orientation_target,
        articulation,
        rmpflow,
        translation_thresh=0.01,
        orientation_thresh=0.1,
        timeout=500,
    ):
        """
        Use RMPflow to move a robot Articulation to a desired task-space position.
        Exit upon timeout or when end effector comes within the provided threshholds of the target pose.
        """

        """articulation_motion_policy = ArticulationMotionPolicy(articulation, rmpflow, 1 / 60)
        rmpflow.set_end_effector_target(translation_target, orientation_target)

        for i in range(timeout):
            ee_trans, ee_rot = rmpflow.get_end_effector_pose(
                articulation_motion_policy.get_active_joints_subset().get_joint_positions()
            )

            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, translation_target)
            rotation_target = quats_to_rot_matrices(orientation_target)
            rot_dist = distance_metrics.rotational_distance_angle(ee_rot, rotation_target)

            done = trans_dist < translation_thresh and rot_dist < orientation_thresh

            if done:
                return True

            rmpflow.update_world()
            action = articulation_motion_policy.get_next_articulation_action(1 / 60)
            articulation.apply_action(action)

            # If not done on this frame, yield() to pause execution of this function until
            # the next frame.
            yield ()

        return False"""
        print("goto_position")
        return False

    # * speed is given in km/h but isaac sim uses rads/min i think
    def drive_to_pos(self, speed, goal):
        """
        joint_indices 0 = ?_bogie
        joint_indices 1 = ?_bogie
        joint_indices 2 = back_joint
        joint_indices 3 = BRB_wheel
        joint_indices 4 = BRF_wheel
        joint_indices 5 = BLF_wheel
        joint_indices 6 = BLB_wheel
        joint_indices 7 = ?_bogie
        joint_indices 8 = ?_bogie
        joint_indices 9 = FRF_wheel
        joint_indices 10 = FRB_wheel
        joint_indices 11 = FLB_wheel
        joint_indices 12 = FLF_wheel
        """
        drive_action = ArticulationAction(joint_velocities=np.array([speed]), joint_indices=np.array([3,4,5,6,9,10,11,12]))
        self._articulation.apply_action(drive_action)

        goal_block = self._goal_blocks[goal]
        goal_position, _ = goal_block.get_local_pose()
        position, _ = self._articulation.get_local_pose()
        
        while not (self.reached_end(goal_position, position, 4)):
            goal_position, _ = goal_block.get_local_pose()
            position, _ = self._articulation.get_local_pose()
            self.turn_to_pos(goal_position, position)
            yield ()

        # stopping
        drive_action = ArticulationAction(joint_velocities=np.array([0]), joint_indices=np.array([3,4,5,6,9,10,11,12]))
        self._articulation.apply_action(drive_action)
        return True


    def turn_to_pos(self, end, current):
        """alpha=arctan(x/y) = np.arctan2(y,x)"""

        pos = [(end[0] - current[0]),(end[1] - current[1])]
        angle = np.arctan2(pos[0], pos[1])

        if angle < 0:
            angle = angle + np.pi
        elif angle > 0:
            angle = angle - np.pi
        else:
            pass

        back = self.dc.get_rigid_body("/forwarder/back")
        back_pose = self.dc.get_rigid_body_pose(back)

        front = self.dc.get_rigid_body("/forwarder/front")
        front_pose = self.dc.get_rigid_body_pose(front)

        desired_angle = (angle + back_pose.r[2]) + front_pose.r[2]

        control_output = self.pid.compute(desired_angle)
        carb.log_error(control_output)
        turn_action = ArticulationAction(joint_positions=control_output, joint_indices=2)
        self._articulation.apply_action(turn_action)
        return True

    def reached_end(self, end, current, acc=2):
        """Check if robot has reached end position with given accuracy"""
        return np.sqrt(np.sum((end - current) ** 2)) <= acc
    
    def get_lidar_param(self):
        time.sleep(1)

        self.timeline.pause()

        pointcloud = omni.isaac.sensor.get_pointcloud("/forwarder/front/front_lidar/front_lidar")

        #semantics = self.lidarInterface.get_semantic_data("/forwarder/front/front_lidar/front_lidar")

        print("Point Cloud", pointcloud)

        #print("Semantic ID", semantics)

        carb.log_error(pointcloud)
        yield()


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        
        self.prev_error = 0
        self.integral = 0

    def compute(self, feedback):
        error = self.setpoint - feedback
        self.integral += error
        derivative = error - self.prev_error
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        self.prev_error = error
        
        return output * -1
    
