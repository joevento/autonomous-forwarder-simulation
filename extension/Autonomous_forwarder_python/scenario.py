# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
import omni.kit.app
import os
import omni.usd
from pxr import Gf
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.articulations import Articulation, ArticulationView
import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep
from omni.isaac.core.objects import FixedCuboid

import omni.isaac.core.utils.stage as stage_utils

from .global_variables import EXTENSION_NAME
from .robot import RobotHandler

class AutonomousForestryScript:
    def __init__(self):
        self._articulation = None
        self._articulation_view = None
        self._xform = None
        # waypoints which the robot will try to hit
        self._targets = [
            [0, 0, 0.85],
            [-2, -10, 0.85],
            [2, -20, 0.85],
            [0, -30, 0.85]
        ]

        self._script_generator = None

    def load_assets(self):
        """Load assets onto the stage and return them so they can be registered with the
        core.World.

        This function is called from ui_builder._setup_scene()

        The position in which things are loaded is also the position to which
        they will be returned on reset.
        """
        self._terrain_prim_path = "/terrain"

        terrain_prim_path = "/World/terrain"
        FixedCuboid(terrain_prim_path, scale=[200,200,0.1])

        return True
    
    def load_robot(self):
        # Fowarder
        robot_prim_path = "/forwarder"
        stage_utils.add_reference_to_stage(usd_path="/home/discoflower8890/autonomous-forwarder-simulation/dev_forwarder.usd", prim_path=robot_prim_path)
        self._articulation = Articulation(prim_path="/forwarder/front", name="forwarder")
        prim_utils.set_prim_property(robot_prim_path, "xformOp:translate", Gf.Vec3d(0, 0, 0))

        # setting up lidars for the forwarder
        lidarPath = "/front/front_lidar/front_lidar"

        # xform for waypoint detection
        self._xform = XFormPrim(robot_prim_path + "/front/Xform", name="target")
        prim_utils.set_prim_property(robot_prim_path + "/front/Xform", "xformOp:translate", Gf.Vec3d(0, -3, 0))

        # ! would be better to use the rtx lidar but idk how to get the ouput from that shit
        result, lidarPrim = omni.kit.commands.execute(
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
                )
        

        # 1. Create The Camera
        """_, self.sensor = omni.kit.commands.execute(
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
        annotator.attach(render_product)"""

    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """
        # Set a camera view that looks good
        set_camera_view(eye=[6.5, 6.5, 6.5], target=[0.0, 0.0, 1.0], camera_prim_path="/OmniverseKit_Persp")

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
        self._robot = RobotHandler()
        yield from self._robot.drive_spline(1.5, self._targets, 2, self._xform)
        