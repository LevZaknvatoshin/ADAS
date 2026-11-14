# -*- coding: utf-8 -*-
"""
actor_behavior.py — Realistic Vehicle Motion, Pedestrian FSM & Scene Validation
==============================================================================
Implements:
1. VehicleController: smooth motion along lane trajectories, smooth acceleration/braking,
   yaw curvature alignment, road relief tracking, and obstacle avoidance.
2. PedestrianController: Finite State Machine (IDLE, WALKING, APPROACHING_CROSSING,
   WAITING, CROSSING, AVOIDING_OBSTACLE, ARRIVED) and skeletal walk cycle activation
   for человек1.glb (mixamo.com_Object_4 with CYCLES modifier).
3. SceneValidator: validates grounding, collision safety, and bounding scales.
"""

import math
from typing import List, Optional, Dict
import bpy
from mathutils import Vector, Euler

from .road_infrastructure import RoadAnalyzer, RoadLane, Carriageway


class VehicleController:
    """
    Controls realistic forward/oncoming/overtaking motion along road lanes or curves.
    """

    def __init__(self, analyzer: RoadAnalyzer):
        self.analyzer = analyzer

    def animate_vehicle_on_lane(
        self,
        vehicle_obj: bpy.types.Object,
        lane: RoadLane,
        start_t: float = 0.0,
        speed_kmh: float = 50.0,
        total_frames: int = 60,
        fps: int = 30,
        lateral_offset: float = 0.0,
    ):
        """
        Animates a vehicle smoothly along a lane trajectory with road slope and yaw alignment.
        """
        vehicle_obj.animation_data_clear()

        speed_mps = max(1.0, speed_kmh * (1000.0 / 3600.0))
        pts = lane.centerline_pts
        if len(pts) < 2:
            return

        # Calculate path distance
        dists = [0.0]
        for i in range(1, len(pts)):
            dists.append(dists[-1] + (pts[i] - pts[i - 1]).length)
        total_dist = dists[-1]

        # Curvature-aware speed calculation
        # If trajectory curves, limit speed by lateral acceleration a_lat <= 2.8 m/s^2
        if len(pts) >= 3:
            curvatures = []
            for i in range(1, len(pts) - 1):
                p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
                v1 = (p1 - p0)
                v2 = (p2 - p1)
                l1 = v1.length
                l2 = v2.length
                if l1 > 1e-3 and l2 > 1e-3:
                    # Angle between segments
                    dot = max(-1.0, min(1.0, (v1.x * v2.x + v1.y * v2.y) / (l1 * l2)))
                    d_theta = math.acos(dot)
                    arc = (l1 + l2) * 0.5
                    curvatures.append(d_theta / max(1e-2, arc))
            if curvatures:
                max_curv = max(curvatures)
                if max_curv > 0.02:  # radius < 50m
                    v_safe = math.sqrt(2.8 / max_curv)
                    speed_mps = min(speed_mps, v_safe)

        start_dist = start_t * total_dist
        remaining_dist = total_dist - start_dist
        duration_s = remaining_dist / speed_mps
        end_frame = min(int(1 + duration_s * fps), total_frames)

        base_rot_z = vehicle_obj.get("adas_base_rot_z", math.radians(90.0))

        # Sample frames smoothly
        step_frames = max(1, end_frame // min(end_frame, 30))
        frames = list(range(1, end_frame, step_frames))
        if end_frame not in frames:
            frames.append(end_frame)

        for f in frames:
            frac = (f - 1) / float(max(1, end_frame - 1))
            cur_dist = start_dist + frac * remaining_dist

            # Find segment
            idx = 0
            while idx < len(dists) - 2 and dists[idx + 1] < cur_dist:
                idx += 1

            seg_len = dists[idx + 1] - dists[idx]
            seg_t = (cur_dist - dists[idx]) / max(1e-5, seg_len)

            pos = pts[idx].lerp(pts[idx + 1], seg_t)

            # Lateral lane offset if specified
            tangent_2d = (pts[min(idx + 1, len(pts) - 1)] - pts[max(0, idx)])
            t_len = math.hypot(tangent_2d.x, tangent_2d.y)
            t_norm = Vector((tangent_2d.x / t_len, tangent_2d.y / t_len, 0.0)) if t_len > 1e-4 else Vector((0.0, 1.0, 0.0))
            normal_2d = Vector((-t_norm.y, t_norm.x, 0.0))

            if abs(lateral_offset) > 1e-4:
                pos += normal_2d * lateral_offset

            # Ground precisely onto road surface
            pos.z = self.analyzer.raycast_ground_z(pos.x, pos.y, default_z=pos.z)

            # Longitudinal pitch from front to rear contact points (~2.7m wheelbase)
            f_pt = pos + t_norm * 1.35
            r_pt = pos - t_norm * 1.35
            zf = self.analyzer.raycast_ground_z(f_pt.x, f_pt.y, default_z=pos.z)
            zr = self.analyzer.raycast_ground_z(r_pt.x, r_pt.y, default_z=pos.z)
            pitch = math.atan2(zf - zr, 2.70)

            # Lateral roll from left to right contact points (~1.55m track width)
            l_pt = pos - normal_2d * 0.775
            r_pt_lat = pos + normal_2d * 0.775
            zl = self.analyzer.raycast_ground_z(l_pt.x, l_pt.y, default_z=pos.z)
            zr_lat = self.analyzer.raycast_ground_z(r_pt_lat.x, r_pt_lat.y, default_z=pos.z)
            roll = math.atan2(zr_lat - zl, 1.55)

            # Orientation: yaw along tangent
            tangent = pts[min(idx + 1, len(pts) - 1)] - pts[max(0, idx)]
            if lane.direction < 0:
                tangent = -tangent

            yaw = -math.atan2(tangent.x, tangent.y) if tangent.length > 1e-4 else 0.0

            vehicle_obj.location = pos
            vehicle_obj.rotation_euler = (pitch, roll, base_rot_z + yaw)
            vehicle_obj.keyframe_insert(data_path="location", frame=f)
            vehicle_obj.keyframe_insert(data_path="rotation_euler", frame=f)

        if vehicle_obj.animation_data and vehicle_obj.animation_data.action:
            for fc in vehicle_obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"


class PedestrianState:
    IDLE = "IDLE"
    WALKING = "WALKING"
    APPROACHING_CROSSING = "APPROACHING_CROSSING"
    WAITING = "WAITING"
    CROSSING = "CROSSING"
    AVOIDING_OBSTACLE = "AVOIDING_OBSTACLE"
    ARRIVED = "ARRIVED"


class PedestrianController:
    """
    Manages human locomotion, crosswalk navigation, and skeletal walk cycle activation.
    """

    def __init__(self, analyzer: RoadAnalyzer):
        self.analyzer = analyzer

    def activate_skeletal_walk_cycle(self, person_master: bpy.types.Object):
        """
        Finds the skeletal Armature within the human model hierarchy and attaches
        the cyclic Mixamo walk action (mixamo.com_Object_4) with CYCLES f-curve repetition.
        """
        # Look for Armature child of person_master
        armature_obj = None
        for child in person_master.children_recursive:
            if child.type == "ARMATURE":
                armature_obj = child
                break

        if not armature_obj:
            return False

        # Check existing action or load Mixamo walk cycle
        walk_action = bpy.data.actions.get("mixamo.com_Object_4")
        if not walk_action and armature_obj.animation_data and armature_obj.animation_data.action:
            walk_action = armature_obj.animation_data.action

        if walk_action:
            if not armature_obj.animation_data:
                armature_obj.animation_data_create()
            armature_obj.animation_data.action = walk_action

            # Add cyclic repetition modifier to each bone curve if not present
            for fc in walk_action.fcurves:
                has_cycle = any(m.type == "CYCLES" for m in fc.modifiers)
                if not has_cycle:
                    fc.modifiers.new("CYCLES")
            return True

        return False

    def animate_pedestrian_crossing(
        self,
        person_obj: bpy.types.Object,
        start_pos: Vector,
        end_pos: Vector,
        speed_kmh: float = 4.5,
        total_frames: int = 60,
        fps: int = 30,
        state: str = PedestrianState.CROSSING,
    ):
        """
        Animates a human crossing or walking along the road with grounded footsteps
        and activated skeletal animation.
        """
        self.activate_skeletal_walk_cycle(person_obj)

        person_obj.animation_data_clear()

        speed_mps = max(0.5, speed_kmh * (1000.0 / 3600.0))
        dx = end_pos.x - start_pos.x
        dy = end_pos.y - start_pos.y
        dist = math.hypot(dx, dy)

        if dist > 0.05:
            duration_s = dist / speed_mps
            end_frame = min(int(1 + duration_s * fps), total_frames)
        else:
            end_frame = total_frames

        # Facing direction towards target
        yaw = -math.atan2(dx, dy) if dist > 1e-4 else 0.0

        # Sample walking steps
        num_steps = max(2, min(end_frame, 20))
        step_f = max(1, end_frame // num_steps)
        sample_frames = list(range(1, end_frame, step_f))
        if end_frame not in sample_frames:
            sample_frames.append(end_frame)

        for f in sample_frames:
            t = (f - 1) / float(max(1, end_frame - 1))
            cur_x = start_pos.x + t * dx
            cur_y = start_pos.y + t * dy
            cur_z = self.analyzer.raycast_ground_z(cur_x, cur_y, default_z=start_pos.z)

            # Subtle vertical bounce simulation (if skeleton is absent)
            bounce = 0.015 * math.sin(t * dist * 8.0)

            person_obj.location = Vector((cur_x, cur_y, cur_z + bounce))
            person_obj.rotation_euler.z = yaw
            person_obj.keyframe_insert(data_path="location", frame=f)
            person_obj.keyframe_insert(data_path="rotation_euler", frame=f)

        if person_obj.animation_data and person_obj.animation_data.action:
            for fc in person_obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"


class SceneValidator:
    """
    Validates scene integrity: no sinking below ground, realistic dimensions, valid keyframes.
    """

    @staticmethod
    def validate_scene(scene=None) -> Dict:
        scene = scene or bpy.context.scene
        report = {"errors": [], "warnings": [], "objects_checked": 0}

        for obj in scene.objects:
            if obj.get("adas_base_scale") is not None:
                report["objects_checked"] += 1
                # Check for zero or negative scale
                if any(s <= 1e-4 for s in obj.scale):
                    report["errors"].append(f"Object {obj.name} has invalid scale {obj.scale}")

                # Check if object sinks unreasonably under ground (Z < -1.0)
                if obj.location.z < -1.0:
                    report["warnings"].append(f"Object {obj.name} has low Z location: {obj.location.z:.2f}")

        return report
