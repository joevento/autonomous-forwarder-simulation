import numpy as np
import math
import omni.usd

from scipy.interpolate import splprep, splev
from omni.isaac.core.articulations import Articulation, ArticulationView
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Sdf, Usd, UsdGeom

from .global_variables import WHEEL_INDICES, MIDDLE_JOINT

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
        self._articulation_view = ArticulationView(prim_paths_expr="/forwarder/front")
        self._articulation_view.initialize()

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
                joint_indices=np.array(WHEEL_INDICES)
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
                joint_indices=np.array(WHEEL_INDICES)
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
                    joint_indices=np.array(MIDDLE_JOINT)
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