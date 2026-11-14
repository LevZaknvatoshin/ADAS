# -*- coding: utf-8 -*-
"""
physics_controller.py — Rigid Body Simulation, Falling Cones & Scale Normalization
==================================================================================
Handles real-world physical dynamics for obstacles (traffic cones, barrels, boxes, tires)
falling onto the 3D road surface under gravity, colliding realistically, and baking
clean deterministic keyframes.

Also estimates real-world metric scales (cars ~4.5m, humans ~1.75m, cones ~0.7m).
"""

import math
from typing import List, Optional, Dict
import bpy
from mathutils import Vector, Euler


class SceneScaleEstimator:
    """
    Standardizes 3D object dimensions to real-world metric scales in Blender coordinates.
    """

    REFERENCE_DIMS = {
        "VEHICLE": {
            "SEDAN": {"length": 4.60, "width": 1.85, "height": 1.45},
            "SUV":   {"length": 4.90, "width": 2.00, "height": 1.75},
            "TRUCK": {"length": 6.50, "width": 2.20, "height": 2.50},
        },
        "PERSON": {"height": 1.75, "width": 0.50, "depth": 0.28},
        "OBSTACLE": {
            "CONE":   {"height": 0.70, "radius": 0.22},
            "BARREL": {"height": 0.90, "radius": 0.30},
            "BOX":    {"height": 0.50, "width": 0.50, "depth": 0.50},
            "TIRE":   {"height": 0.25, "radius": 0.35},
        },
    }

    @classmethod
    def get_target_scale(
        cls,
        obj_type: str,
        subtype: str = "DEFAULT",
        current_dims: Optional[Vector] = None,
        lane_width: Optional[float] = None,
    ) -> float:
        """
        Returns uniform scale factor to ensure object conforms to metric real-world dimensions
        and matches the detected lane width of the reconstructed road scene.
        """
        # Lane-width normalization factor: standard highway/urban lane is ~3.5 meters
        lane_ratio = 1.0
        if lane_width is not None and lane_width > 0.5:
            # Scale proportionally to lane width (bounded between 0.6x and 1.8x for safety)
            lane_ratio = max(0.6, min(1.8, lane_width / 3.5))

        if obj_type == "VEHICLE":
            ref = cls.REFERENCE_DIMS["VEHICLE"].get(subtype, cls.REFERENCE_DIMS["VEHICLE"]["SEDAN"])
            if current_dims and current_dims.y > 0.1:
                return float(ref["length"] / current_dims.y) * lane_ratio
            return 0.075 * lane_ratio  # Calibrated for машина.glb (width ~1.54m, ~52% of lane)

        elif obj_type == "PERSON":
            ref = cls.REFERENCE_DIMS["PERSON"]
            if current_dims and current_dims.z > 0.1:
                return float(ref["height"] / current_dims.z) * lane_ratio
            return 1.0 * lane_ratio  # Calibrated for человек1.glb

        elif obj_type == "OBSTACLE":
            st = subtype.upper()
            ref = cls.REFERENCE_DIMS["OBSTACLE"].get(st, cls.REFERENCE_DIMS["OBSTACLE"]["CONE"])
            if current_dims and current_dims.z > 0.05:
                return float(ref["height"] / current_dims.z) * lane_ratio
            defaults = {"CONE": 1.3, "BARREL": 0.70, "BOX": 0.10, "TIRE": 0.25}
            base = defaults.get(st, 0.10 if st == "DEFAULT" else 0.4)
            return base * lane_ratio

        return 1.0 * lane_ratio


class RigidBodyController:
    """
    Simulates physical dynamics for traffic cones, obstacles, vehicles and pedestrians
    falling onto road under gravity, colliding realistically, and baking clean deterministic keyframes.
    Supports both live viewport simulation (Play / Spacebar) and baked timeline animation.
    """

    def __init__(self, scene=None):
        self.scene = scene or bpy.context.scene

    def ensure_rigidbody_world(self):
        """Ensures the scene has an active RigidBodyWorld configured for stable simulations."""
        if not self.scene.rigidbody_world:
            try:
                bpy.ops.rigidbody.world_add()
            except Exception:
                pass
        if self.scene.rigidbody_world:
            self.scene.rigidbody_world.enabled = True
            self.scene.rigidbody_world.substeps_per_frame = 10
            self.scene.rigidbody_world.solver_iterations = 20

    def ensure_road_collision(self):
        """Ensures all road geometry has passive mesh collision so objects never fall through."""
        self.ensure_rigidbody_world()
        road_names = [
            "RoadCam_2.5D_SceneMesh",
            "RoadScene_2.5D_Reconstruction",
            "ADAS_Road_StaticCollider",
            "ADAS_Carriageway_Mesh",
        ]
        found_road = False
        for name in road_names:
            obj = bpy.data.objects.get(name)
            if obj and obj.type == "MESH":
                self._make_passive_collider(obj)
                found_road = True

        for obj in self.scene.objects:
            if obj.get("role") == "roadcam_2.5d_reconstruction" and obj.type == "MESH":
                self._make_passive_collider(obj)
                found_road = True

        return found_road

    def _make_passive_collider(self, obj):
        try:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            if not obj.rigid_body:
                bpy.ops.rigidbody.object_add(type="PASSIVE")
            rb = obj.rigid_body
            rb.collision_shape = "MESH"
            rb.friction = 0.90
            rb.restitution = 0.10
            obj.select_set(False)
            if self.scene.rigidbody_world and self.scene.rigidbody_world.collection:
                if obj.name not in self.scene.rigidbody_world.collection.objects:
                    self.scene.rigidbody_world.collection.objects.link(obj)
        except Exception as e:
            print(f"[ADAS Physics] Error setting passive road collider on {obj.name}: {e}")

    def setup_object_ragdoll(
        self,
        obj: bpy.types.Object,
        obj_type: str = "OBSTACLE",
        subtype: str = "DEFAULT",
        drop_height: Optional[float] = None,
        initial_tilt_deg: float = 0.0,
    ):
        """
        Equips any scene object (vehicle, person, obstacle) with active Rigid Body dynamics
        and positions it slightly above ground so it falls and settles naturally on the asphalt.
        """
        self.ensure_road_collision()

        # Target mesh for rigid body
        phys_target = obj
        if obj.type != "MESH":
            for child in obj.children_recursive:
                if child.type == "MESH":
                    phys_target = child
                    break

        # Elevation above road surface
        if drop_height is None:
            if obj_type == "VEHICLE":
                drop_height = 0.40
            elif obj_type == "PERSON":
                drop_height = 0.25
            else:
                drop_height = 0.60

        obj.location.z += max(0.15, drop_height)
        if initial_tilt_deg != 0.0:
            obj.rotation_euler.x += math.radians(initial_tilt_deg)

        # Clear prior keyframes
        obj.animation_data_clear()
        phys_target.animation_data_clear()

        # Add active Rigid Body
        try:
            bpy.context.view_layer.objects.active = phys_target
            phys_target.select_set(True)
            if not phys_target.rigid_body:
                bpy.ops.rigidbody.object_add(type="ACTIVE")
            rb = phys_target.rigid_body
            rb.enabled = True

            if obj_type == "VEHICLE":
                rb.collision_shape = "CONVEX_HULL"
                rb.mass = 1500.0        # Realistic 1.5 tonne car
                rb.friction = 0.85
                rb.restitution = 0.08    # Low bounciness
                rb.linear_damping = 0.05
                rb.angular_damping = 0.10

            elif obj_type == "PERSON":
                rb.collision_shape = "CAPSULE" if phys_target.type == "MESH" else "CONVEX_HULL"
                rb.mass = 75.0          # Realistic human weight ~75kg
                rb.friction = 0.75
                rb.restitution = 0.10
                rb.linear_damping = 0.10
                rb.angular_damping = 0.20

            else:  # OBSTACLE
                st = subtype.upper() if subtype else "DEFAULT"
                if st == "CONE":
                    rb.collision_shape = "CONVEX_HULL"
                    rb.mass = 2.5
                    rb.friction = 0.85
                    rb.restitution = 0.15
                    obj.rotation_euler.x += math.radians(3.0)
                elif st == "TIRE":
                    rb.collision_shape = "CONVEX_HULL"
                    rb.mass = 10.0
                    rb.friction = 0.90
                    rb.restitution = 0.25
                elif st == "BARREL":
                    rb.collision_shape = "CYLINDER"
                    rb.mass = 25.0
                    rb.friction = 0.80
                    rb.restitution = 0.10
                elif st == "BOX":
                    rb.collision_shape = "BOX"
                    rb.mass = 5.0
                    rb.friction = 0.75
                    rb.restitution = 0.10
                else:
                    rb.collision_shape = "CONVEX_HULL"
                    rb.mass = 4.0
                    rb.friction = 0.80
                    rb.restitution = 0.15

            phys_target.select_set(False)
            obj["adas_phys_target"] = phys_target.name

            # Ensure linked to physics world collection
            if self.scene.rigidbody_world and self.scene.rigidbody_world.collection:
                if phys_target.name not in self.scene.rigidbody_world.collection.objects:
                    self.scene.rigidbody_world.collection.objects.link(phys_target)

        except Exception as e:
            print(f"[ADAS Physics] Error adding active rigid body to {obj.name}: {e}")

    def setup_falling_cone(
        self,
        cone_obj: bpy.types.Object,
        target_x: float,
        target_y: float,
        road_surface_z: float,
        drop_height: float = 1.2,
        initial_tilt_deg: float = 3.0,
    ):
        """
        Positions a cone or obstacle above the target road coordinate with subtle tilt
        and equips it with active Blender Rigid Body dynamics.
        """
        # Position slightly above road
        start_z = road_surface_z + max(0.5, drop_height)
        cone_obj.location = Vector((target_x, target_y, start_z))
        if initial_tilt_deg != 0.0:
            cone_obj.rotation_euler.x += math.radians(initial_tilt_deg)
        self.setup_object_ragdoll(cone_obj, obj_type="OBSTACLE", subtype="CONE", drop_height=0.0)

    def bake_simulation_to_keyframes(
        self,
        physics_objects: List[bpy.types.Object],
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
        step: int = 1,
    ):
        """
        Bakes the rigid body simulation into clean LocRot keyframes on the objects,
        then cleanly disables the active rigid body so the scene is deterministic,
        lightweight, and 100% reproducible on any machine.
        """
        if not physics_objects:
            return

        f_start = frame_start or self.scene.frame_start
        f_end = frame_end or self.scene.frame_end

        # Ensure RigidBodyWorld cache matches requested range
        if self.scene.rigidbody_world:
            self.scene.rigidbody_world.point_cache.frame_start = f_start
            self.scene.rigidbody_world.point_cache.frame_end = f_end

        print(f"[ADAS Physics] Baking physics simulation for {len(physics_objects)} object(s) across frames {f_start}..{f_end}...")

        # Bake rigid body world point cache
        try:
            bpy.ops.ptcache.bake_all(bake=True)
        except Exception as e:
            print(f"[ADAS Physics] ptcache.bake_all notice: {e}")

        # Record world matrix transforms into keyframes
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for f in range(f_start, f_end + 1, step):
            self.scene.frame_set(f)
            depsgraph.update()
            for obj in physics_objects:
                target_name = obj.get("adas_phys_target")
                phys_obj = bpy.data.objects.get(target_name) if target_name else obj
                if phys_obj and phys_obj != obj:
                    eval_child = phys_obj.evaluated_get(depsgraph)
                    mat_world = eval_child.matrix_world.copy()
                else:
                    eval_obj = obj.evaluated_get(depsgraph)
                    mat_world = eval_obj.matrix_world.copy()
                loc, rot, sc = mat_world.decompose()
                obj.location = loc
                obj.rotation_euler = rot.to_euler()
                obj.keyframe_insert(data_path="location", frame=f)
                obj.keyframe_insert(data_path="rotation_euler", frame=f)

        # Smooth linear keyframe transitions
        for obj in physics_objects:
            if obj.animation_data and obj.animation_data.action:
                for fc in obj.animation_data.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "LINEAR"

        # Safely disable Rigid Body from dynamic objects so baked keyframes take full control
        for obj in physics_objects:
            target_name = obj.get("adas_phys_target")
            phys_obj = bpy.data.objects.get(target_name) if target_name else obj
            if phys_obj and phys_obj.rigid_body:
                phys_obj.rigid_body.enabled = False
            if obj.rigid_body:
                obj.rigid_body.enabled = False

        self.scene.frame_set(f_start)
        print("[ADAS Physics] Successfully baked and keyframed physical objects!")

    def clear_ragdoll(self, objects: List[bpy.types.Object]):
        """Clears baked keyframes and removes rigid bodies from objects, resetting them to rest."""
        for obj in objects:
            obj.animation_data_clear()
            target_name = obj.get("adas_phys_target")
            phys_obj = bpy.data.objects.get(target_name) if target_name else obj
            if phys_obj and phys_obj.rigid_body:
                try:
                    bpy.context.view_layer.objects.active = phys_obj
                    phys_obj.select_set(True)
                    bpy.ops.rigidbody.object_remove()
                    phys_obj.select_set(False)
                except Exception:
                    phys_obj.rigid_body.enabled = False
            if obj.rigid_body:
                try:
                    bpy.context.view_layer.objects.active = obj
                    obj.select_set(True)
                    bpy.ops.rigidbody.object_remove()
                    obj.select_set(False)
                except Exception:
                    obj.rigid_body.enabled = False
