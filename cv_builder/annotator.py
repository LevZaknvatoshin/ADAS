"""
YOLO Ground Truth Annotator for ADAS & Obstacle Detection
=========================================================
Projects 3D bounding boxes to 2D normalized camera coordinates:
  - Supports perspective & wide-angle lenses
  - Near-plane edge clipping
  - Multi-ray occlusion checking
  - Exports standard YOLO format: (class_id cx cy w h)
"""

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


class YOLOAnnotator:
    def __init__(self, min_box_size=0.003, check_occlusion=True):
        self.min_box_size = min_box_size
        self.check_occlusion = check_occlusion

    def get_frame_annotations(self, scene=None):
        scene = scene or bpy.context.scene
        cam = scene.camera
        if not cam:
            return []

        depsgraph = bpy.context.evaluated_depsgraph_get()
        cam_eval = cam.evaluated_get(depsgraph)
        cam_pos = cam_eval.matrix_world.translation
        inv_cam_mat = cam_eval.matrix_world.inverted()

        annotations = []

        bbox_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # Scan for objects marked with adas_class_id
        for obj in scene.objects:
            if "adas_class_id" not in obj:
                continue

            eval_obj = obj.evaluated_get(depsgraph)
            bbox_corners = []
            if eval_obj.data and hasattr(eval_obj, "bound_box"):
                bbox_corners = [eval_obj.matrix_world @ Vector(c) for c in eval_obj.bound_box]
            else:
                mesh_children = [c for c in obj.children_recursive if c.type == "MESH"]
                for c in mesh_children:
                    eval_c = c.evaluated_get(depsgraph)
                    if hasattr(eval_c, "bound_box"):
                        bbox_corners.extend([eval_c.matrix_world @ Vector(pt) for pt in eval_c.bound_box])

            if not bbox_corners:
                continue

            corners_cam_space = [inv_cam_mat @ pt for pt in bbox_corners]

            valid_world_points = []
            for i, p_cam in enumerate(corners_cam_space):
                if p_cam.z < -0.05:
                    valid_world_points.append(bbox_corners[i])

            # Clip against near camera plane for 8-corner bounding boxes
            if len(bbox_corners) == 8:
                near_z = -0.05
                for idx1, idx2 in bbox_edges:
                    p1 = corners_cam_space[idx1]
                    p2 = corners_cam_space[idx2]
                    if (p1.z < near_z and p2.z >= near_z) or (p2.z < near_z and p1.z >= near_z):
                        denom = p2.z - p1.z
                        if abs(denom) > 1e-6:
                            t = (near_z - p1.z) / denom
                            clipped_pt_w = bbox_corners[idx1].lerp(bbox_corners[idx2], t)
                            valid_world_points.append(clipped_pt_w)

            if not valid_world_points:
                continue

            xs, ys = [], []
            for pt_w in valid_world_points:
                p = world_to_camera_view(scene, cam, pt_w)
                xs.append(p.x)
                ys.append(1.0 - p.y)  # YOLO standard: 0 = top, 1 = bottom

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            if x_max < 0.0 or x_min > 1.0 or y_max < 0.0 or y_min > 1.0:
                continue

            x_min_c = max(0.0, min(1.0, x_min))
            x_max_c = max(0.0, min(1.0, x_max))
            y_min_c = max(0.0, min(1.0, y_min))
            y_max_c = max(0.0, min(1.0, y_max))

            width = x_max_c - x_min_c
            height = y_max_c - y_min_c

            if width < self.min_box_size or height < self.min_box_size:
                continue

            # Exclude tiny slivers right on the edge of the screen (< 2% size on frame boundary)
            if (x_min_c <= 0.001 or x_max_c >= 0.999 or y_min_c <= 0.001 or y_max_c >= 0.999):
                if height < 0.02 or width < 0.02:
                    continue

            # Occlusion testing
            if self.check_occlusion:
                center_w = sum(bbox_corners, Vector()) / float(len(bbox_corners))
                top_w = center_w + Vector((0.0, 0.0, 0.4))
                is_visible = False
                for pt in [center_w, top_w]:
                    ray_vec = pt - cam_pos
                    dist = ray_vec.length
                    if dist > 0.1:
                        hit, loc, norm, idx, hit_obj, matrix = scene.ray_cast(
                            depsgraph, cam_pos, ray_vec.normalized(), distance=dist - 0.2
                        )
                        if (not hit or hit_obj == eval_obj or hit_obj == obj or
                            hit_obj.parent == obj or hit_obj in getattr(obj, "children_recursive", []) or
                            hit_obj.name.startswith("ADAS_") or hit_obj.name.startswith("Road_") or
                            hit_obj.name.startswith("Median_") or hit_obj.name.startswith("Grass_")):
                            is_visible = True
                            break
                if not is_visible:
                    continue

            cx = (x_min_c + x_max_c) / 2.0
            cy = (y_min_c + y_max_c) / 2.0
            class_id = int(obj["adas_class_id"])
            class_name = obj.get("adas_class_name", "obstacle")

            annotations.append({
                "class_id": class_id,
                "class_name": class_name,
                "cx": cx,
                "cy": cy,
                "w": width,
                "h": height,
                "x_min": x_min_c,
                "y_min": y_min_c,
                "x_max": x_max_c,
                "y_max": y_max_c,
            })

        return annotations

    def format_yolo_lines(self, annotations):
        return [
            f"{a['class_id']} {a['cx']:.6f} {a['cy']:.6f} {a['w']:.6f} {a['h']:.6f}\n"
            for a in annotations
        ]
