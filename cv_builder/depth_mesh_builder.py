"""
DepthMeshBuilder (Redirected to Procedural Clean Digital Twin)
==============================================================
СТРОГО ЗАПРЕЩЕНО: Depth Map, Displace Modifier, UV Camera Projection.
Этот класс перенаправлен на процедурный генератор чистой геометрии (Digital Twin).
"""

import os
import cv2
import numpy as np
import bpy
from .cv_road_twin_builder import build_digital_twin


class DepthMeshBuilder:
    """
    Замена старого генератора: строит ТОЛЬКО чистую сплайновую 3D геометрию
    без каких-либо проекций фото и модификаторов смещения.
    """
    COLLECTION_NAME = "DT_Road"

    def __init__(self, scene=None):
        self.scene = scene or bpy.context.scene

    def clear_previous(self):
        for coll_name in ["ADAS_Environment_v3", "DT_Road", "DT_Environment"]:
            if coll_name in bpy.data.collections:
                coll = bpy.data.collections[coll_name]
                for obj in list(coll.objects):
                    bpy.data.objects.remove(obj, do_unlink=True)

        for name in ["Terrain_Environment_v3", "PerspectiveTerrainMesh", "Cube", "Light"]:
            if name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    def build(self, image_path, depth_map_path=None, masks_dict=None, sky_dome_path=None, calibrator=None, poles_2d=None, is_night=False):
        self.clear_previous()

        ref_path = os.path.abspath(image_path)
        with open(ref_path, "rb") as f:
            img_bgr = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

        fov = calibrator.fov_deg if calibrator else 80.0
        height = calibrator.cam_height if calibrator else 7.0

        props = getattr(self.scene, "adas_props", None)
        road_w = props.road_total_width if props and props.road_total_width > 3.5 else None
        lane_count = (props.lanes_per_direction * 2) if props and props.lanes_per_direction > 1 else None

        meta = {
            "is_night": is_night,
            "poles_2d": poles_2d or []
        }

        metrics = build_digital_twin(
            img_bgr=img_bgr,
            seg_masks=masks_dict or {},
            seg_metadata=meta,
            fov_deg=fov,
            cam_height=height,
            road_width_override=road_w,
            lane_count_override=lane_count,
            scene=self.scene,
        )

        return metrics
