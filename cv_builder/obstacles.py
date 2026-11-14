"""
Obstacle Manager — Spawning + animation of scenario actors
==========================================================
Handles operator-defined obstacles, pedestrians, vehicles, and custom models.
Each spawned object gets ADAS ground-truth metadata for the annotator.
"""

import bpy
import bmesh
import math
import os
import random
from mathutils import Vector, Euler
from mathutils.bvhtree import BVHTree

from .road_infrastructure import RoadAnalyzer, LaneDetector, CollisionBuilder
from .physics_controller import RigidBodyController, SceneScaleEstimator
from .actor_behavior import VehicleController, PedestrianController, PedestrianState, SceneValidator


# ---------------------------------------------------------------------------
# Metric mesh normalization
# ---------------------------------------------------------------------------

def normalize_mesh_to_metric(obj, target_size_m, axis="X"):
    """
    Uniformly scales *obj* so that its local-space bounding-box extent along
    *axis* equals *target_size_m* metres.

    Parameters
    ----------
    obj : bpy.types.Object
        The top-level object (usually an Empty parent of imported meshes).
    target_size_m : float
        Desired real-world size in metres along the chosen axis.
    axis : str
        ``"X"`` for width, ``"Y"`` for length, ``"Z"`` for height.

    The function computes the bounding box from *all* recursive mesh children
    (if *obj* itself has no mesh data) so that imported GLB hierarchies are
    measured correctly.

    After scaling it stores ``obj["adas_metric_width"]`` (or ``_height`` /
    ``_length``) so downstream code can query the true metric size.
    """
    # Collect all mesh-child world-space coordinates to cover the full hierarchy
    coords = []
    if obj.type == "MESH" and obj.data and obj.data.vertices:
        mw = obj.matrix_world
        coords.extend(mw @ v.co for v in obj.data.vertices)
    for child in obj.children_recursive:
        if child.type == "MESH" and child.data and child.data.vertices:
            mw = child.matrix_world
            coords.extend(mw @ v.co for v in child.data.vertices)

    if not coords:
        # Nothing measurable — store target and return
        obj["adas_metric_width"] = target_size_m
        return

    axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(axis.upper(), 0)
    vals = [c[axis_idx] for c in coords]
    bbox_extent = max(vals) - min(vals)

    if bbox_extent < 1e-6:
        obj["adas_metric_width"] = target_size_m
        return

    ratio = target_size_m / bbox_extent
    obj.scale *= ratio
    obj["adas_base_scale"] = float(obj.scale.x)

    # Persist the metric size for downstream queries (surface alignment, etc.)
    prop_name = {"X": "adas_metric_width", "Y": "adas_metric_length", "Z": "adas_metric_height"}.get(axis.upper(), "adas_metric_width")
    obj[prop_name] = target_size_m


OBJ_COLLECTION = "ADAS_Objects"

CLASS_IDS = {
    "PERSON": 0,
    "VEHICLE": 1,
    "OBSTACLE": 2,
    "CUSTOM": 2,  # default for custom; user can override
}

CLASS_NAMES = {
    "PERSON": "person",
    "VEHICLE": "vehicle",
    "OBSTACLE": "obstacle",
    "CUSTOM": "obstacle",
}

# Color palettes
CAR_COLORS = [
    (0.85, 0.05, 0.05, 1), (0.05, 0.15, 0.75, 1), (0.92, 0.92, 0.94, 1),
    (0.08, 0.08, 0.08, 1), (0.40, 0.42, 0.45, 1), (0.02, 0.35, 0.15, 1),
]
CLOTH_COLORS = [
    (0.1, 0.1, 0.15, 1), (0.7, 0.2, 0.1, 1), (0.2, 0.5, 0.3, 1),
    (0.8, 0.8, 0.8, 1), (0.15, 0.25, 0.5, 1), (0.6, 0.1, 0.1, 1),
]


def cleanup_orphan_imported_objects():
    """
    Detects and deletes any leftover imported meshes/empties sitting orphaned at origin (0, 0, 0)
    when their parent container was deleted or unlinked.
    Protected collections: ADAS_Environment, ADAS_Road, ADAS_Lights, ADAS_Cameras.
    """
    prefixes = (
        "sketchfab", "object_", "box.", "rootnode", "car01", "frontwheel", "windows",
        "mixamo", "mesh", "cleaner", "materialmerger"
    )
    protected_cols = {"ADAS_Environment", "ADAS_Road", "ADAS_Lights", "ADAS_Cameras"}

    for obj in list(bpy.data.objects):
        try:
            # 1. Never delete environment or road models
            if (
                obj.name.startswith("ADAS_Env_")
                or obj.name.startswith("ADAS_Road")
                or obj.name.startswith("RoadCam_")
                or obj.name.startswith("RoadScene_")
                or obj.name.startswith("PhotoDepth_")
                or obj.get("adas_is_environment")
                or obj.get("role") in ("road_surface", "road_surface_3d", "roadcam_2.5d_reconstruction", "environment")
            ):
                continue

            # 2. Never delete objects residing in protected collections
            if any(c.name in protected_cols for c in obj.users_collection):
                continue

            if obj.parent is None and obj.location.length < 0.15:
                name_low = obj.name.lower()
                if any(name_low.startswith(p) for p in prefixes):
                    for child in list(obj.children_recursive):
                        if any(c.name in protected_cols for c in child.users_collection):
                            continue
                        if child.get("adas_is_environment") or child.name.startswith("ADAS_Env_"):
                            continue
                        try:
                            m = child.data if child.type == "MESH" else None
                            bpy.data.objects.remove(child, do_unlink=True)
                            if m and m.users == 0:
                                bpy.data.meshes.remove(m)
                        except (ReferenceError, KeyError):
                            pass
                    m = obj.data if obj.type == "MESH" else None
                    bpy.data.objects.remove(obj, do_unlink=True)
                    if m and m.users == 0:
                        bpy.data.meshes.remove(m)
        except (ReferenceError, KeyError):
            pass


def remove_item_objects(item_index: int):
    """
    Completely and cleanly removes all objects (master container and all recursive children,
    meshes, and materials) associated with scenario item `item_index`.
    Prevents orphan meshes from lingering at (0, 0, 0).
    """
    cleanup_orphan_imported_objects()
    targets = []

    # Find master empties or tagged objects with this index
    for obj in bpy.data.objects:
        try:
            if obj.get("adas_is_environment") or obj.name.startswith("ADAS_Env_"):
                continue
            if obj.get("adas_item_index") == item_index or obj.get("adas_parent_index") == item_index:
                targets.append(obj)
                targets.extend(list(obj.children_recursive))
        except (ReferenceError, KeyError):
            continue

    # Also check name patterns like Obstacle_0, Vehicle_0, Person_0, Custom_0
    prefixes = (f"Obstacle_{item_index}", f"Vehicle_{item_index}", f"Person_{item_index}", f"Custom_{item_index}")
    for obj in bpy.data.objects:
        try:
            if obj.get("adas_is_environment") or obj.name.startswith("ADAS_Env_"):
                continue
            if any(obj.name == p or obj.name.startswith(p + "_") for p in prefixes):
                if obj not in targets:
                    targets.append(obj)
                    targets.extend(list(obj.children_recursive))
        except (ReferenceError, KeyError):
            continue

    # Deduplicate preserving order
    seen = set()
    unique_targets = []
    for o in targets:
        if o.name in bpy.data.objects and o not in seen:
            if not o.name.startswith("ADAS_Env_") and not o.get("adas_is_environment"):
                seen.add(o)
                unique_targets.append(o)

    # Delete children first, then parents
    for o in reversed(unique_targets):
        try:
            if o.name in bpy.data.objects:
                mesh_data = o.data if o.type == "MESH" else None
                bpy.data.objects.remove(o, do_unlink=True)
                if mesh_data and mesh_data.users == 0:
                    bpy.data.meshes.remove(mesh_data)
        except (ReferenceError, KeyError):
            pass

    cleanup_orphan_imported_objects()


class ObstacleManager:

    def __init__(self):
        if OBJ_COLLECTION not in bpy.data.collections:
            c = bpy.data.collections.new(OBJ_COLLECTION)
            bpy.context.scene.collection.children.link(c)
        self.col = bpy.data.collections[OBJ_COLLECTION]
        cleanup_orphan_imported_objects()

    def clear(self):
        """Safely and recursively removes all scenario objects and orphan meshes."""
        cleanup_orphan_imported_objects()
        for obj in list(self.col.objects):
            try:
                if obj.name in bpy.data.objects:
                    for child in list(obj.children_recursive):
                        try:
                            if child.name in bpy.data.objects:
                                m = child.data if child.type == "MESH" else None
                                bpy.data.objects.remove(child, do_unlink=True)
                                if m and m.users == 0:
                                    bpy.data.meshes.remove(m)
                        except (ReferenceError, KeyError):
                            pass
                    m = obj.data if obj.type == "MESH" else None
                    bpy.data.objects.remove(obj, do_unlink=True)
                    if m and m.users == 0:
                        bpy.data.meshes.remove(m)
            except (ReferenceError, KeyError):
                pass

        for obj in list(bpy.data.objects):
            try:
                if ("adas_item_index" in obj or "adas_parent_index" in obj) and not obj.get("adas_is_environment") and not obj.name.startswith("ADAS_Env_"):
                    bpy.data.objects.remove(obj, do_unlink=True)
            except (ReferenceError, KeyError):
                pass
        cleanup_orphan_imported_objects()

    def get_road_z(self, x: float, y: float) -> float:
        """
        Finds the road surface height Z at position (x, y).
        Uses BVH raycasting directly on road meshes in local coordinates.
        """
        road_objs = [
            o for o in bpy.context.scene.objects
            if o.type == "MESH" and (
                o.name.startswith("RoadCam_") or
                o.name.startswith("RoadScene_") or
                o.name.startswith("ADAS_Road") or
                o.name.startswith("ADAS_Carriageway") or
                o.name.startswith("PhotoDepth") or
                o.name.startswith("ADAS_Env_") or
                o.name.startswith("ADAS_Environment") or
                o.get("role") in ("roadcam_2.5d_reconstruction", "road_surface")
            )
        ]
        if not road_objs:
            return 0.0

        depsgraph = bpy.context.evaluated_depsgraph_get()
        best_z = None
        for obj in road_objs:
            try:
                bvh = BVHTree.FromObject(obj, depsgraph)
                mw = obj.matrix_world
                mw_inv = mw.inverted()
                local_origin = mw_inv @ Vector((x, y, 50.0))
                local_dir = mw_inv.to_3x3() @ Vector((0.0, 0.0, -1.0))
                hit, normal, index, dist = bvh.ray_cast(local_origin, local_dir)
                if hit is not None:
                    world_hit = mw @ hit
                    if best_z is None or world_hit.z > best_z:
                        best_z = world_hit.z
            except Exception:
                continue

        if best_z is not None:
            return best_z

        # Fallback to nearest vertex on road mesh
        for obj in road_objs:
            try:
                bvh = BVHTree.FromObject(obj, depsgraph)
                mw = obj.matrix_world
                mw_inv = mw.inverted()
                nearest, normal, index, dist = bvh.find_nearest(mw_inv @ Vector((x, y, 0.0)))
                if nearest is not None:
                    return (mw @ nearest).z
            except Exception:
                continue

        return 0.0

    def align_to_road_surface(self, obj, item=None):
        """
        Aligns the object to the 3D road surface slope.
        For vehicles: uses a longitudinal pitch along heading so wheels
        hug the road surface with zero hover or clipping.
        For non-vehicles: aligns Z height to ground and maintains stable contact.
        """
        if not obj:
            return

        pos = obj.location
        yaw = obj.rotation_euler.z

        obj_type = obj.get("adas_class_name", "obstacle")
        if obj_type == "vehicle" or (item and getattr(item, "obj_type", "") == "VEHICLE"):
            # Measure ground directly under vehicle
            gz = self.get_road_z(pos.x, pos.y)
            obj.location.z = gz

            # Longitudinal pitch along heading (direction vehicle is facing)
            # At yaw=0, vehicle faces +Y. Heading vector is (-sin_y, cos_y)
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            front_x = pos.x - 1.2 * sin_y
            front_y = pos.y + 1.2 * cos_y
            rear_x = pos.x + 1.2 * sin_y
            rear_y = pos.y - 1.2 * cos_y

            z_front = self.get_road_z(front_x, front_y)
            z_rear = self.get_road_z(rear_x, rear_y)
            dz = z_front - z_rear
            pitch = -math.atan2(dz, 2.40)
            max_rad = math.radians(12.0)
            pitch = max(-max_rad, min(max_rad, pitch))

            obj.rotation_euler.x = pitch
            obj.rotation_euler.y = 0.0 # Clean zero roll: prevents crazy sideways capsizing
            obj.rotation_euler.z = yaw
        else:
            gz = self.get_road_z(pos.x, pos.y)
            obj.location.z = gz

    def spawn_item(self, item, index=0, total_frames=60, fps=30):
        """
        Spawn a single ADAS_SceneItem into the 3D scene.
        `item` is an ADAS_SceneItem PropertyGroup instance.
        """
        obj = None

        if item.obj_type == "CUSTOM":
            obj = self._import_custom(item.custom_model_path, f"Custom_{index}")
        elif item.obj_type == "PERSON":
            obj = self._make_person(f"Person_{index}")
        elif item.obj_type == "VEHICLE":
            target_w = None
            try:
                from .road_infrastructure import RoadAnalyzer, LaneDetector
                det = LaneDetector(RoadAnalyzer())
                lane_w = det.get_orthogonal_width_at(item.pos_x, item.pos_y)
                if 1.0 <= lane_w <= 12.0:
                    target_w = 0.53 * lane_w
            except Exception:
                target_w = None
            obj = self._make_vehicle(f"Vehicle_{index}", target_width=target_w)
        elif item.obj_type == "OBSTACLE":
            stype = item.subtype if item.subtype in ("CONE","TIRE","BOX","BARREL") else None
            obj = self._make_obstacle(f"Obstacle_{index}", stype)

        if not obj:
            return

        # Position on the horizontal road surface
        road_z = self.get_road_z(item.pos_x, item.pos_y)
        obj.location = Vector((item.pos_x, item.pos_y, road_z))
        base_rot_z = obj.get("adas_base_rot_z", 0.0)
        obj.rotation_euler.z = base_rot_z + math.radians(item.rotation_z)

        # Constant metric scale — Blender's camera handles perspective foreshortening
        base_scale = obj.get("adas_base_scale", 1.0)
        total_scale = base_scale * item.scale_factor
        obj.scale = Vector((total_scale, total_scale, total_scale))
        obj["adas_base_scale"] = float(base_scale)

        # 4-wheel kinematic ground alignment (pitch & roll to hug road terrain)
        self.align_to_road_surface(obj, item)

        # Ground truth tags
        obj["adas_item_index"] = index
        obj["adas_class_id"] = CLASS_IDS.get(item.obj_type, 2)
        obj["adas_class_name"] = CLASS_NAMES.get(item.obj_type, "obstacle")
        for child in obj.children_recursive:
            child["adas_item_index"] = index
            child["adas_parent_index"] = index

        # Animation: Pre-cooked procedural animation is disabled in favor of
        # manual keyframing and baking in Tab 3 ("Анимация & Запекание").
        # if item.animated:
        #     self._animate(obj, item, total_frames, fps)

        return obj

    # ───────────────────── animation ─────────────────────

    def _animate(self, obj, item, total_frames, fps):
        obj.animation_data_clear()
        beh = getattr(item, "scenario_behavior", "CUSTOM")

        # 1. Pedestrian Walking & Crossing with skeletal animation
        if item.obj_type == "PERSON":
            analyzer = RoadAnalyzer(bpy.context.scene)
            ped_ctrl = PedestrianController(analyzer)
            start_p = Vector((item.pos_x, item.pos_y, self.get_road_z(item.pos_x, item.pos_y)))
            end_p = Vector((item.end_x, item.end_y, self.get_road_z(item.end_x, item.end_y)))
            ped_ctrl.animate_pedestrian_crossing(
                obj, start_p, end_p,
                speed_kmh=item.speed_kmh or 4.5,
                total_frames=total_frames,
                fps=fps,
            )
            return

        # 2. Author-defined motion (along road curve if FOLLOW_ROUTE, else direct linear path)
        route_obj = bpy.data.objects.get("ADAS_Vehicle_Route") if beh == "FOLLOW_ROUTE" else None
        if route_obj and route_obj.type == "CURVE":
            self._animate_along_route(obj, route_obj, item, total_frames, fps)
        else:
            self._animate_linear(obj, item, total_frames, fps)

    def _animate_linear(self, obj, item, total_frames, fps):
        scene = bpy.context.scene
        obj.animation_data_clear()

        bounds = get_road_bounds(scene)
        lane_detector = bounds.get("lane_detector")
        lane_detector = get_road_bounds(bpy.context.scene).get("lane_detector")
        base_scale = obj.get("adas_base_scale", 1.0)
        base_rot_z = obj.get("adas_base_rot_z", 0.0)

        beh = getattr(item, "scenario_behavior", "CUSTOM")

        # Driving scenarios span total_frames so vehicle visibly moves continuously along the lane
        if item.obj_type == "VEHICLE" or beh in ("DRIVE_FORWARD", "ONCOMING", "OVERTAKE", "FOLLOW_ROUTE", "ANIMATED"):
            end_frame = max(30, total_frames)
        else:
            speed_mps = (item.speed_kmh or 5.0) * (1000.0 / 3600.0)
            dx_m = item.end_x - item.pos_x
            dy_m = item.end_y - item.pos_y
            dist_m = math.sqrt(dx_m * dx_m + dy_m * dy_m)
            if speed_mps > 0 and dist_m > 0:
                # 2.5D perspective depth compression factor (~25x)
                effective_dist = dist_m * 25.0
                duration_s = effective_dist / speed_mps
                end_frame = max(15, min(int(1 + duration_s * fps), total_frames))
            else:
                end_frame = total_frames

        # Sample evenly-spaced points along trajectory from start to finish
        num_samples = min(16, end_frame)
        sample_steps = [i / float(num_samples - 1) for i in range(num_samples)]
        dx = item.end_x - item.pos_x
        dy = item.end_y - item.pos_y

        for t in sample_steps:
            f = int(round(1 + t * (end_frame - 1)))
            scene.frame_set(f)

            cur_x = item.pos_x + t * dx
            cur_y = item.pos_y + t * dy
            cur_z = self.get_road_z(cur_x, cur_y)

            # Subtle hop when obstacle is flying/rolling in
            if "FLY_INTO" in beh and 0.1 < t < 0.9:
                cur_z += 0.25 * math.sin(t * math.pi)

            obj.location = Vector((cur_x, cur_y, cur_z))

            # Orientation: align heading and slope without roll artifacts
            obj.rotation_euler.y = 0.0
            if item.obj_type == "VEHICLE" and (dx * dx + dy * dy) > 1e-4:
                motion_yaw = -math.atan2(dx, dy)
                obj.rotation_euler.z = base_rot_z + motion_yaw + math.radians(item.rotation_z)
                step_dt = 0.05
                ahead_x = cur_x + step_dt * dx
                ahead_y = cur_y + step_dt * dy
                ahead_z = self.get_road_z(ahead_x, ahead_y)
                dist_step = math.hypot(ahead_x - cur_x, ahead_y - cur_y)
                pitch = math.atan2(ahead_z - cur_z, max(1e-4, dist_step))
                obj.rotation_euler.x = max(-math.radians(15.0), min(math.radians(15.0), pitch))
            else:
                cur_rot = base_rot_z + math.radians(item.rotation_z)
                if "FLY_INTO" in beh:
                    cur_rot += math.radians(360.0 * t)
                obj.rotation_euler.z = cur_rot
                obj.rotation_euler.x = 0.0

            # Adaptive lane scaling: maintains 53% lane width along perspective trajectory
            s_base = base_scale
            if getattr(item, "obj_type", "") == "VEHICLE" and lane_detector:
                try:
                    cur_lw = lane_detector.get_orthogonal_width_at(cur_x, cur_y)
                    if 1.0 <= cur_lw <= 12.0:
                        s_base = (0.53 * cur_lw) / 20.4902
                except Exception:
                    pass
            s = s_base * item.scale_factor
            obj.scale = Vector((s, s, s))

            obj.keyframe_insert(data_path="location", frame=f)
            obj.keyframe_insert(data_path="rotation_euler", frame=f)
            obj.keyframe_insert(data_path="scale", frame=f)

        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"

    def _animate_along_route(self, obj, route_obj, item, total_frames, fps):
        """Animates a vehicle along the 3D road centerline curve ADAS_Vehicle_Route."""
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_route = route_obj.evaluated_get(depsgraph)
        eval_mesh = eval_route.to_mesh()
        eval_pts = [eval_route.matrix_world @ v.co for v in eval_mesh.vertices]
        eval_route.to_mesh_clear()

        if len(eval_pts) < 2:
            return self._animate_linear(obj, item, total_frames, fps)

        dists = [0.0]
        for i in range(1, len(eval_pts)):
            dists.append(dists[-1] + (eval_pts[i] - eval_pts[i - 1]).length)
        total_len = dists[-1]

        speed_mps = (item.speed_kmh or 50.0) * (1000.0 / 3600.0)
        duration_s = total_len / max(1.0, speed_mps)
        end_frame = min(int(1 + duration_s * fps), total_frames)

        lane_detector = get_road_bounds(bpy.context.scene).get("lane_detector")
        base_scale = obj.get("adas_base_scale", 1.0)
        base_rot_z = obj.get("adas_base_rot_z", 0.0)

        step = max(1, end_frame // min(end_frame, 24))
        sample_frames = list(range(1, end_frame, step))
        if end_frame not in sample_frames:
            sample_frames.append(end_frame)

        for f in sample_frames:
            bpy.context.scene.frame_set(f)
            t = (f - 1) / float(max(1, end_frame - 1))
            target_dist = t * total_len

            idx = 0
            while idx < len(dists) - 2 and dists[idx + 1] < target_dist:
                idx += 1

            seg_len = dists[idx + 1] - dists[idx]
            seg_t = (target_dist - dists[idx]) / max(1e-5, seg_len)
            pos = eval_pts[idx].lerp(eval_pts[idx + 1], seg_t)

            tangent = eval_pts[min(idx + 1, len(eval_pts) - 1)] - eval_pts[max(0, idx)]
            yaw = -math.atan2(tangent.x, tangent.y) if tangent.length > 1e-4 else 0.0

            obj.location = pos
            obj.rotation_euler.z = base_rot_z + yaw
            s_base = base_scale
            if getattr(item, "obj_type", "") == "VEHICLE" and lane_detector:
                try:
                    cur_lw = lane_detector.get_orthogonal_width_at(pos.x, pos.y)
                    if 1.0 <= cur_lw <= 12.0:
                        s_base = (0.53 * cur_lw) / 20.4902
                except Exception:
                    pass
            s = s_base * item.scale_factor
            obj.scale = Vector((s, s, s))
            obj.keyframe_insert(data_path="location", frame=f)
            obj.keyframe_insert(data_path="rotation_euler", frame=f)
            obj.keyframe_insert(data_path="scale", frame=f)

        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"

    # ───────────────────── model discovery & loader ──────

    def _find_model_file(self, filename: str) -> str | None:
        """Finds model path in cv_builder/models or root 'модели'."""
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(curr_dir, "models", filename),
            os.path.join(curr_dir, "..", "модели", filename),
            os.path.join(curr_dir, "..", "models", filename),
            os.path.join(bpy.path.abspath("//"), "модели", filename),
            os.path.join(bpy.path.abspath("//"), "models", filename),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return None

    def _load_glb(self, filename: str, name: str, scale=1.0, rot_z_deg=0.0, offset_z=0.0, remove_names=None):
        """Loads a GLB model with proper scaling, centering, and parenting to a master empty."""
        path = self._find_model_file(filename)
        if not path or not os.path.exists(path):
            print(f"[ADAS] Builtin model not found: {filename}")
            return None

        before = set(bpy.data.objects)
        try:
            bpy.ops.import_scene.gltf(filepath=path)
        except Exception as e:
            print(f"[ADAS] Failed to import {filename}: {e}")
            return None

        new_objs = list(set(bpy.data.objects) - before)
        if not new_objs:
            return None

        # Clean stray unwanted objects (like stray Icosphere in person)
        if remove_names:
            for ro in list(new_objs):
                if ro.name in remove_names and ro.name in bpy.data.objects:
                    bpy.data.objects.remove(ro, do_unlink=True)
                    new_objs.remove(ro)

        # Create a master empty container
        master = bpy.data.objects.new(name, None)
        master.empty_display_type = "PLAIN_AXES"
        self.col.objects.link(master)

        top_parents = [o for o in new_objs if o.parent is None and o != master]
        for tp in top_parents:
            tp.parent = master

        # Auto-ground custom imported model so its lowest point sits at Z = 0
        bpy.context.view_layer.update()
        mesh_objs = [o for o in new_objs if o.type == "MESH"]
        if mesh_objs:
            min_zs = [min((m.matrix_world @ v.co).z for v in m.data.vertices) for m in mesh_objs if len(m.data.vertices) > 0]
            if min_zs:
                min_z = min(min_zs)
                if min_z != float("inf") and abs(min_z) > 1e-5:
                    for tp in top_parents:
                        tp.location.z -= min_z
                    bpy.context.view_layer.update()

        # Automatic precision grounding: ensure lowest contact point of model sits exactly at Z = 0.0
        bpy.context.view_layer.update()
        mesh_objs = [o for o in new_objs if o.type == "MESH"]
        if mesh_objs:
            min_zs = [min((m.matrix_world @ v.co).z for v in m.data.vertices) for m in mesh_objs if len(m.data.vertices) > 0]
            if min_zs:
                min_z = min(min_zs)
                if min_z != float("inf") and abs(min_z) > 1e-5:
                    for tp in top_parents:
                        tp.location.z -= min_z
                    bpy.context.view_layer.update()

        if offset_z != 0.0:
            for tp in top_parents:
                tp.location.z += offset_z

        # Ensure all imported objects are in our collection and tagged
        for o in new_objs:
            o["adas_parent_name"] = master.name
            if o.name not in [obj.name for obj in self.col.objects]:
                self.col.objects.link(o)
            for c in list(o.users_collection):
                if c != self.col:
                    try:
                        c.objects.unlink(o)
                    except Exception:
                        pass

        master.rotation_mode = "XYZ"
        master.rotation_euler.z = math.radians(rot_z_deg)
        master.scale = (scale, scale, scale)
        master["adas_base_rot_z"] = math.radians(rot_z_deg)
        master["adas_base_scale"] = float(scale)

        return master

    # ───────────────────── custom import ─────────────────

    def _import_custom(self, path, name):
        abs_path = bpy.path.abspath(path) if path else ""
        if not abs_path or not os.path.exists(abs_path):
            print(f"[ADAS] Custom model not found: {path}")
            return None

        ext = os.path.splitext(abs_path)[1].lower()
        before = set(bpy.data.objects)
        try:
            if ext == ".obj":
                if hasattr(bpy.ops.wm, "obj_import"):
                    bpy.ops.wm.obj_import(filepath=abs_path)
                else:
                    bpy.ops.import_scene.obj(filepath=abs_path)
            elif ext == ".fbx":
                bpy.ops.import_scene.fbx(filepath=abs_path)
            elif ext in (".gltf", ".glb"):
                bpy.ops.import_scene.gltf(filepath=abs_path)
            elif ext == ".blend":
                with bpy.data.libraries.load(abs_path, link=False) as (df, dt):
                    dt.objects = df.objects
            else:
                print(f"[ADAS] Unsupported format: {ext}")
                return None
        except Exception as e:
            print(f"[ADAS] Import failed: {e}")
            return None

        new_objs = list(set(bpy.data.objects) - before)
        if not new_objs:
            return None

        # Link all imported objects to our collection
        for o in new_objs:
            if o.name not in [obj.name for obj in self.col.objects]:
                self.col.objects.link(o)
                for c in list(o.users_collection):
                    if c != self.col:
                        c.objects.unlink(o)

        # Create a master empty container
        master = bpy.data.objects.new(name, None)
        master.empty_display_type = "PLAIN_AXES"
        self.col.objects.link(master)

        top_parents = [o for o in new_objs if o.parent is None and o != master]
        for tp in top_parents:
            tp.parent = master

        # Auto-ground custom imported model so its lowest point sits at Z = 0
        bpy.context.view_layer.update()
        mesh_objs = [o for o in new_objs if o.type == "MESH"]
        if mesh_objs:
            min_zs = [min((m.matrix_world @ v.co).z for v in m.data.vertices) for m in mesh_objs if len(m.data.vertices) > 0]
            if min_zs:
                min_z = min(min_zs)
                if min_z != float("inf") and abs(min_z) > 1e-5:
                    for tp in top_parents:
                        tp.location.z -= min_z
                    bpy.context.view_layer.update()

        master.rotation_mode = "XYZ"
        master["adas_base_rot_z"] = 0.0
        master["adas_base_scale"] = 1.0

        return master

    # ───────────────────── procedural models & fallbacks ─

    def _pbr_mat(self, name, color, metallic=0.0, roughness=0.5):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return mat

    def _make_person(self, name):
        """High-poly photorealistic human figure from человек1.glb / человек.glb, scaled to realistic 1.75m height."""
        # Unscaled height of human mesh without Icosphere is ~1.679m.
        # Scale factor for realistic 1.75m adult pedestrian:
        scale = 1.75 / 1.679  # ~1.042
        model = self._load_glb("человек1.glb", name, scale=scale, remove_names=["Icosphere"])
        if not model:
            model = self._load_glb("человек.glb", name, scale=scale, remove_names=["Icosphere"])
        if model:
            model["adas_base_scale"] = float(scale)
            model["adas_metric_height"] = 1.75
            return model

        # Procedural fallback figure
        color = random.choice(CLOTH_COLORS)
        mat = self._pbr_mat(f"{name}_mat", color, roughness=0.7)
        skin = self._pbr_mat(f"{name}_skin", (0.78, 0.58, 0.45, 1), roughness=0.5)

        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        bm = bmesh.new()

        # Torso
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co.x *= 0.42; v.co.y *= 0.26; v.co.z *= 0.60
            v.co.z += 0.85 + 0.30

        # Head
        bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.13)
        bm.verts.ensure_lookup_table()
        for v in bm.verts[-50:]:
            v.co.z += 1.60

        # Legs
        for sign in (-1, 1):
            bmesh.ops.create_cube(bm, size=1.0)
            bm.verts.ensure_lookup_table()
            for v in bm.verts[-8:]:
                v.co.x *= 0.15; v.co.y *= 0.17; v.co.z *= 0.85
                v.co.x += sign * 0.12; v.co.z += 0.425

        bm.to_mesh(mesh)
        bm.free()
        obj.data.materials.append(mat)
        self.col.objects.link(obj)
        obj["adas_base_rot_z"] = 0.0
        obj["adas_base_scale"] = 1.0
        return obj

    def _make_vehicle(self, name, target_width=None):
        """
        Realistic vehicle from машина.glb, joined into a single unified mesh object,
        centered at ground level (Z=0 at bottom of wheels), hood pointing along +Y,
        scaled precisely to physical metric size (53% of lane or 1.82m).
        """
        base_glb_width = 20.4902
        if target_width is not None and target_width > 0.1:
            scale = target_width / base_glb_width
            target_w = target_width
        else:
            scale = 1.82 / base_glb_width
            target_w = 1.82

        path = self._find_model_file("машина.glb")
        if path and os.path.exists(path):
            before_names = set(bpy.data.objects.keys())
            try:
                bpy.ops.import_scene.gltf(filepath=path)
            except Exception as e:
                print(f"[ADAS] Failed to import car GLB: {e}")
                before_names = None

            if before_names is not None:
                new_names = [n for n in bpy.data.objects.keys() if n not in before_names]
                meshes = [bpy.data.objects[n] for n in new_names if n in bpy.data.objects and bpy.data.objects[n].type == 'MESH']
                if meshes:
                    bpy.ops.object.select_all(action='DESELECT')
                    for m in meshes:
                        m.select_set(True)
                    bpy.context.view_layer.objects.active = meshes[0]
                    bpy.ops.object.parent_clear(type='CLEAR')
                    bpy.ops.object.join()
                    car = bpy.context.view_layer.objects.active

                    for n in new_names:
                        o = bpy.data.objects.get(n)
                        if o and o != car:
                            bpy.data.objects.remove(o, do_unlink=True)

                    # Ground tires at Z = 0.0
                    min_z = min(v.co.z for v in car.data.vertices)
                    for v in car.data.vertices:
                        v.co.z -= min_z
                    car.data.update()

                    car.name = name
                    car.rotation_mode = "XYZ"
                    car.rotation_euler = (0.0, 0.0, 0.0)
                    car.scale = (scale, scale, scale)
                    car["adas_base_rot_z"] = 0.0
                    car["adas_base_scale"] = float(scale)
                    car["adas_metric_width"] = float(target_w)
                    car["adas_class_name"] = "vehicle"

                    if car.name not in [obj.name for obj in self.col.objects]:
                        self.col.objects.link(car)
                    for c in list(car.users_collection):
                        if c != self.col:
                            try:
                                c.objects.unlink(car)
                            except Exception:
                                pass

                    return car

        # Procedural fallback vehicle
        color = random.choice(CAR_COLORS)
        mat = self._pbr_mat(f"{name}_mat", color, metallic=0.8, roughness=0.25)

        dims = {
            "SEDAN": (1.55, 4.10, 1.35, 0.50),
            "SUV":   (1.70, 4.40, 1.55, 0.65),
            "TRUCK": (1.95, 5.50, 2.20, 1.00),
        }
        w, l, h, cab_h = dims.get(vtype, dims["SEDAN"])
        wheel_r = 0.32
        gc = wheel_r
        body_h = h - cab_h - gc

        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        bm = bmesh.new()

        # Lower body
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co.x *= w; v.co.y *= l; v.co.z *= body_h
            v.co.z += gc + body_h / 2.0

        # Cabin
        bmesh.ops.create_cube(bm, size=1.0)
        bm.verts.ensure_lookup_table()
        for v in bm.verts[-8:]:
            v.co.x *= w * 0.85; v.co.y *= l * 0.5; v.co.z *= cab_h
            v.co.y -= l * 0.08
            v.co.z += gc + body_h + cab_h / 2.0

        bm.to_mesh(mesh)
        bm.free()
        obj.data.materials.append(mat)
        self.col.objects.link(obj)
        obj["adas_base_rot_z"] = 0.0
        obj["adas_base_scale"] = 1.0
        return obj

    def _make_obstacle(self, name, subtype=None):
        """Road obstacle from realistic GLB models (бочка, конус, коробка, шина) with fallback."""
        if not subtype or subtype == "DEFAULT":
            subtype = random.choice(["CONE", "TIRE", "BOX", "BARREL"])

        if subtype == "BARREL":
            model = self._load_glb("бочка.glb", name, scale=0.70)
            if model:
                return model
        elif subtype == "CONE":
            model = self._load_glb("конус.glb", name, scale=1.3)
            if model:
                return model
        elif subtype == "BOX":
            model = self._load_glb("коробка.glb", name, scale=0.10)
            if model:
                return model
        elif subtype == "TIRE":
            model = self._load_glb("шина.glb", name, scale=0.25, offset_z=0.0)
            if model:
                return model

        # Procedural fallback
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        bm = bmesh.new()

        if subtype == "CONE":
            mat = self._pbr_mat(f"{name}_mat", (1.0, 0.3, 0.02, 1), roughness=0.4)
            bmesh.ops.create_cube(bm, size=1.0)
            for v in bm.verts:
                v.co.x *= 0.38; v.co.y *= 0.38; v.co.z *= 0.05; v.co.z += 0.025
            bmesh.ops.create_cone(bm, segments=12, radius1=0.18, radius2=0.03, depth=0.7)
            bm.verts.ensure_lookup_table()
            cone_start = 8
            for v in bm.verts[cone_start:]:
                v.co.z += 0.40

        elif subtype == "TIRE":
            mat = self._pbr_mat(f"{name}_mat", (0.06, 0.06, 0.06, 1), roughness=0.9)
            bmesh.ops.create_cone(bm, cap_ends=True, segments=16,
                                  radius1=0.35, radius2=0.35, depth=0.24)
            for v in bm.verts:
                v.co.z += 0.12

        elif subtype == "BARREL":
            mat = self._pbr_mat(f"{name}_mat", (0.9, 0.5, 0.05, 1), roughness=0.5)
            bmesh.ops.create_cone(bm, cap_ends=True, segments=12,
                                  radius1=0.3, radius2=0.3, depth=0.9)
            for v in bm.verts:
                v.co.z += 0.45

        else:  # BOX (Standard shipping crate / euro pallet box: ~1.1m x 1.1m x 0.9m)
            mat = self._pbr_mat(f"{name}_mat", (0.6, 0.45, 0.28, 1), roughness=0.8)
            sx, sy, sz = 1.10, 1.10, 0.85
            bmesh.ops.create_cube(bm, size=1.0)
            for v in bm.verts:
                v.co.x *= sx; v.co.y *= sy; v.co.z *= sz; v.co.z += sz / 2.0

        bm.to_mesh(mesh)
        bm.free()
        obj.data.materials.append(mat)
        self.col.objects.link(obj)
        obj["adas_base_rot_z"] = 0.0
        obj["adas_base_scale"] = 1.0
        return obj


def get_road_bounds(scene=None):
    """
    Computes road boundaries, lane centers, and shoulder positions from reconstructed
    road meshes using CV segmentation and LaneDetector.
    """
    if scene is None:
        scene = bpy.context.scene

    analyzer = RoadAnalyzer(scene)
    road_objs = analyzer.road_objects
    if road_objs:
        all_xs, all_ys = [], []
        for obj in road_objs:
            try:
                mw = obj.matrix_world
                for v in obj.data.vertices:
                    wco = mw @ v.co
                    if wco.z < 3.8:
                        all_xs.append(wco.x)
                        all_ys.append(wco.y)
            except Exception:
                continue
        min_x = min(all_xs) if all_xs else -4.0
        max_x = max(all_xs) if all_xs else 4.0
        min_y = min(all_ys) if all_ys else -6.0
        max_y = max(all_ys) if all_ys else 6.0
    else:
        min_x, max_x = -4.0, 4.0
        min_y, max_y = -6.0, 6.0

    width = max(2.0, max_x - min_x)
    center_x = (min_x + max_x) / 2.0
    lane_offset = max(1.0, min(3.5, width * 0.25))
    y_span = max(1.5, max_y - min_y)
    y_start = min_y + y_span * 0.25
    y_mid = min_y + y_span * 0.50
    y_far = min_y + y_span * 0.75

    rl_x = center_x + lane_offset
    ll_x = center_x - lane_offset
    rl_far_x = center_x + lane_offset * 0.5
    ll_far_x = center_x - lane_offset * 0.5
    rs_x = center_x + lane_offset * 1.5
    ls_x = center_x - lane_offset * 1.5
    cur_lane_w = lane_offset

    # Dynamic precision lane bounds from LaneDetector (CV segmentation)
    det = None
    try:
        from .road_infrastructure import LaneDetector
        det = LaneDetector(analyzer)
        cways = det.analyze_lanes()
        fwd_cws = [cw for cw in cways if cw.direction == 1]
        if fwd_cws and fwd_cws[0].lanes:
            cw = fwd_cws[0]
            r_lane = cw.lanes[0]
            l_lane = cw.lanes[-1]
            if r_lane.centerline_pts and len(r_lane.centerline_pts) >= 2:
                y_start = float(r_lane.centerline_pts[0].y)
                y_far = float(r_lane.centerline_pts[-1].y)
                y_mid = (y_start + y_far) * 0.5
            rl_start_x = float(r_lane.get_center_x_at_y(y_start))
            ll_start_x = float(l_lane.get_center_x_at_y(y_start))
            rl_x = float(r_lane.get_center_x_at_y(y_mid))
            ll_x = float(l_lane.get_center_x_at_y(y_mid))
            rl_far_x = float(r_lane.get_center_x_at_y(y_far))
            ll_far_x = float(l_lane.get_center_x_at_y(y_far))
            cur_lane_w = float(r_lane.get_width_at_y(y_mid))
            rs_x = rl_x + cur_lane_w * 0.75
            ls_x = ll_x - cur_lane_w * 0.75
            center_x = (rl_x + ll_x) * 0.5
    except Exception:
        pass

    route_obj = scene.objects.get("ADAS_Vehicle_Route")
    route_pts = []
    if route_obj and route_obj.type == "CURVE":
        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_route = route_obj.evaluated_get(depsgraph)
            eval_mesh = eval_route.to_mesh()
            route_pts = [eval_route.matrix_world @ v.co for v in eval_mesh.vertices]
            eval_route.to_mesh_clear()
        except Exception:
            route_pts = []

    return {
        "min_x": min_x,
        "max_x": max_x,
        "center_x": center_x,
        "width": width,
        "left_shoulder_x": ls_x,
        "right_shoulder_x": rs_x,
        "left_lane_x": ll_x,
        "left_lane_start_x": locals().get("ll_start_x", ll_x),
        "left_lane_far_x": ll_far_x,
        "right_lane_x": rl_x,
        "right_lane_start_x": locals().get("rl_start_x", rl_x),
        "right_lane_far_x": rl_far_x,
        "lane_width": cur_lane_w,
        "min_y": min_y,
        "max_y": max_y,
        "y_start": y_start,
        "y_mid": y_mid,
        "y_far": y_far,
        "route_pts": route_pts,
        "lane_detector": det,
        "primary_forward_lane": None,
        "primary_oncoming_lane": None,
    }


def get_perspective_scale(y: float, scene=None, x: float = 0.0) -> float:
    """
    Returns perspective scaling factor at world coordinate Y.

    NOTE: Blender's camera projection matrix already handles perspective
    foreshortening (objects further from camera appear smaller via 1/Z).
    Applying an *additional* analytical shrink here caused double-scaling
    (effective ~1/Z²), making distant objects unrealistically tiny.
    This function now always returns 1.0 — objects are placed at constant
    metric scale and the camera does the rest.  The signature is kept
    for API / call-site compatibility.
    """
    return 1.0


def apply_scenario_preset(item, scene=None):
    """
    Applies standard scenario behaviors (preset coordinates, orientation, speed)
    dynamically calculated from the reconstructed road boundaries and route curve.
    Operator can still freely fine-tune any property in the UI after applying.
    """
    bounds = get_road_bounds(scene)
    beh = getattr(item, "scenario_behavior", "CUSTOM")
    t = getattr(item, "obj_type", "OBSTACLE")

    min_x = bounds["min_x"]
    max_x = bounds["max_x"]
    min_y = bounds["min_y"]
    max_y = bounds["max_y"]
    cx = bounds["center_x"]
    rl_x = bounds["right_lane_x"]
    ll_x = bounds["left_lane_x"]
    rl_far_x = bounds.get("right_lane_far_x", rl_x)
    ll_far_x = bounds.get("left_lane_far_x", ll_x)
    rs_x = bounds["right_shoulder_x"]
    ls_x = bounds["left_shoulder_x"]
    y_start = bounds["y_start"]
    y_mid = bounds["y_mid"]
    y_far = bounds["y_far"]
    route_pts = bounds["route_pts"]

    if beh == "CUSTOM":
        return

    # ────── VEHICLE PRESETS ──────
    if t == "VEHICLE":
        item.animated = False
        if beh in ("FOLLOW_ROUTE", "DRIVE_FORWARD"):
            item.pos_x = bounds.get("right_lane_start_x", rl_x)
            item.pos_y = y_start
            item.end_x = rl_far_x
            item.end_y = y_far
            dx = item.end_x - item.pos_x
            dy = item.end_y - item.pos_y
            item.rotation_z = -math.degrees(math.atan2(dx, dy)) if dy > 0 else 0.0

        elif beh == "ONCOMING":
            item.pos_x = ll_far_x
            item.pos_y = y_far
            item.end_x = bounds.get("left_lane_start_x", ll_x)
            item.end_y = y_start
            dx = item.end_x - item.pos_x
            dy = item.end_y - item.pos_y
            item.rotation_z = 180.0 - (math.degrees(math.atan2(dx, -dy)) if dy < 0 else 0.0)

        elif beh == "OVERTAKE":
            item.pos_x = rl_x
            item.pos_y = y_mid
            item.rotation_z = -5.0

        elif beh == "STOPPED_IN_LANE":
            item.pos_x = rl_x
            item.pos_y = y_mid
            item.rotation_z = 0.0

        elif beh == "PARKED_SHOULDER":
            item.pos_x = rs_x
            item.pos_y = y_mid
            item.rotation_z = 0.0

    # ────── PERSON PRESETS ──────
    elif t == "PERSON":
        item.animated = False
        if beh in ("CROSS_ROAD_LR", "STAND_SHOULDER"):
            item.pos_x = ls_x
            item.pos_y = y_mid
            item.rotation_z = 90.0

        elif beh == "CROSS_ROAD_RL":
            item.pos_x = rs_x
            item.pos_y = y_mid
            item.rotation_z = -90.0

        elif beh in ("RUSH_CROSSING", "STAND_ROAD"):
            item.pos_x = rl_x
            item.pos_y = y_mid
            item.rotation_z = 180.0

        elif beh == "WALK_SHOULDER":
            item.pos_x = rs_x
            item.pos_y = y_mid
            item.rotation_z = 0.0

    # ────── OBSTACLE PRESETS ──────
    elif t == "OBSTACLE":
        item.animated = False
        if beh in ("FLY_INTO_ROAD_RIGHT", "STATIC_ON_SHOULDER"):
            item.pos_x = rs_x
            item.pos_y = y_mid
            item.rotation_z = 0.0

        elif beh == "FLY_INTO_ROAD_LEFT":
            item.pos_x = ls_x
            item.pos_y = y_mid
            item.rotation_z = 0.0

        elif beh == "STATIC_IN_LANE":
            item.pos_x = rl_x
            item.pos_y = y_mid
            item.rotation_z = 15.0

        elif beh == "SCATTERED_CONES":
            item.pos_x = rl_x - 0.4
            item.pos_y = y_mid
            item.rotation_z = 25.0

    # ────── CUSTOM PRESETS ──────
    else:
        item.animated = False
        item.pos_x = cx
        item.pos_y = y_mid
        item.rotation_z = 0.0

