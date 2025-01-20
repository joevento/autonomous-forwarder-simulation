import numpy as np
import math
import os
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils

from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.prims import XFormPrim
from scipy.interpolate import splprep, splev
from omni.isaac.core.articulations import Articulation, ArticulationView
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from omni.isaac.core.utils.nucleus import get_assets_root_path

from . import global_variables

# for waypoint debug
from omni.isaac.core.objects import VisualSphere

class SplineInterpolator:
    def __init__(self, waypoints, num_points=50, k=1):
        self.waypoints = waypoints
        self.k = k
        self.num_points = num_points
        self.spline = self.create_spline()

    def create_spline(self):
        x = [point[0] for point in self.waypoints]
        y = [point[1] for point in self.waypoints]
        z = [point[2] for point in self.waypoints]
        tck, u = splprep([x, y, z], s=0, k=3)
        return tck

    def interpolate(self):
        u_new = np.linspace(0, 1, self.num_points)
        x_new, y_new, z_new = splev(u_new, self.spline)
        return np.vstack((x_new, y_new, z_new)).T

class RobotHandler:
    def __init__(self):
        self._articulation = Articulation(prim_path="/forwarder/front", name="Robot")
        self._articulation.initialize()
        #global
        global_variables.articulation = self._articulation
        global_variables.articulation.initialize()

        self._articulation_view = ArticulationView(prim_paths_expr="/forwarder/front")
        self._articulation_view.initialize()
        global_variables.articulation_view = self._articulation_view

        self.world = global_variables.world_instance
        print(self.world.scene)
        print(self.world.scene.get_object(name="forwarder").get_articulation_controller())
        self._articulation_controller = self.world.scene.get_object("forwarder").get_articulation_controller()
        self._articulation_controller.initialize(global_variables.articulation_view)
        #global
        global_variables.articulation_controller = self._articulation_controller
        global_variables.articulation_controller.initialize(self._articulation_view)

        # rmpflow
        # TODO: Make all the paths dynamic
        extension_path = "H:/autonomous_forwarder/extension/Autonomous_forwarder_python"

        rmp_config_dir = os.path.join(extension_path, "motion_policy_configs")

        print(rmp_config_dir)
        # init rmpflow object
        self._rmpflow = RmpFlow(
            robot_description_path= "H:/autonomous_forwarder/extension/Autonomous_forwarder_python/motion_policy_configs/crane_RDF.yaml",
            urdf_path= "H:/autonomous_forwarder/extension/Autonomous_forwarder_python/motion_policy_configs/dev_forwarder_arm.urdf",
            rmpflow_config_path= "H:/autonomous_forwarder/extension/Autonomous_forwarder_python/motion_policy_configs/crane_rmpflow_common.yaml",
            end_effector_frame_name= "arm_end_effector",
            maximum_substep_size=0.00334
        )
        # global
        global_variables.rmpflow = self._rmpflow
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation, self._rmpflow)
        #global
        global_variables.articulation_rmpflow = self._articulation_rmpflow
        
        # TargetPos
        stage_utils.add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._arm_target = XFormPrim("/World/target", scale=[.04, .04, .04])
        self._arm_target.set_world_pose(np.array([.5,0,.7]),euler_angles_to_quats([0,np.pi,0]))
        # global
        global_variables.arm_target = self._arm_target

        # https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_motion_generation_rmpflow.html#generating-motions-with-an-rmpflow-instance

    def move_arm(self, step):
        target_pos, target_orient = self._arm_target.get_world_pose()
        
        self._rmpflow.set_end_effector_target(
            target_pos, target_orient
        )

        # Track any movements of the cube obstacle
        self._rmpflow.update_world()

        #Track any movements of the robot base
        robot_base_translation,robot_base_orientation = self._articulation.get_world_pose()
        self._rmpflow.set_robot_base_pose(robot_base_translation,robot_base_orientation)
        action = self._articulation_rmpflow.get_next_articulation_action(step)
        self._articulation_controller.apply_action(action)
        yield()

    # Driving stuffs
    def drive_spline(self, speed, waypoints, accuracy=0.01, wp_xform=None):
        interpolator = SplineInterpolator(waypoints, k=3)
        path_points = interpolator.interpolate()
        
        # waypoint debug balls
        for i, point in enumerate(path_points):
            prim_path = f"/World/Xform/Cube_{i}"
            VisualSphere(prim_path=prim_path, position=point, color=np.array([1.0, 0.0, 0.0]), radius=accuracy)

        for point in path_points:
            yield from self.drive_to_pos(speed, point, accuracy, wp_xform)
            print("done")
        
        print("Spline drive complete")
        return True

    def drive_to_pos(self, speed, end_pos, accuracy=0.01, wp_xform=None):
        self._articulation_view.apply_action(
            ArticulationAction(
                joint_velocities=np.array(speed),
                joint_indices=np.array(global_variables.WHEEL_INDICES)
            )
        )
        position, rotation = self._articulation.get_world_pose()

        wp_trigger_pos, _ = wp_xform.get_world_pose()
        
        while not self.reached_end(end_pos, wp_trigger_pos, accuracy):
            position, rotation = self._articulation.get_world_pose()
            wp_trigger_pos, _ = wp_xform.get_world_pose()
            yield from self.turn_to_pos(end_pos, wp_trigger_pos, rotation, speed)
            yield ()
        
        self._articulation_view.apply_action(
            ArticulationAction(
                joint_velocities=np.array([0]),
                joint_indices=np.array(global_variables.WHEEL_INDICES)
            )
        )
        return True

    def turn_to_pos(self, end, current, rotation, v):
        end_l = end - current
        current_l = current - current # :D
        z_rot = (self.quaternion_to_euler(rotation)[2])

        forward = [current_l[0], current_l[1]-1, current_l[2]]
        #backward = [current_l[0] - 1, current_l[1], current_l[2]] # Currently not supported, need to get the back parts rotation for this.

        # Define the rotation matrix for Z-axis
        R_z = np.array([
            [np.cos(z_rot), -np.sin(z_rot), 0],
            [np.sin(z_rot), np.cos(z_rot), 0],
            [0, 0, 1]
        ])
        forward = np.dot(R_z, forward)
        #backward = np.dot(R_z, backward) # doesnt support backing up

        if v > 0:
            cross_product = np.cross(end_l, forward)
            z_component = cross_product[2]
            print(np.rad2deg(z_component))
            self._articulation_view.apply_action(
                ArticulationAction(
                    joint_positions=np.array(z_component),
                    joint_indices=np.array(global_variables.MIDDLE_JOINT)
                )
            )
            yield ()

        else: # Negative speeds aka reversing is not supported currently
            yield ()

    def reached_end(self, end, current, acc=2):
        return np.linalg.norm((end[:-1] - current[:-1])) <= acc

    def quaternion_to_euler(self, q):
        x = q[1]
        y = q[2]
        z = q[3]
        w = q[0]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians