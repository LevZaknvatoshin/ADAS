"""
Camera Rig — High-mounted surveillance / road monitoring camera
===============================================================
Replicates the exact setup from the reference frame:
  - Camera mounted on a pole at 6-8m height
  - Looking down at ~25° depression angle
  - Wide-angle lens (110-130° horizontal FOV)
  - Barrel distortion characteristic of surveillance cameras
"""

import bpy
import math
import os
from mathutils import Vector, Euler


class CameraRig:
    CAM_NAME = "ADAS_Camera"

    def setup(self, height=7.5, pitch_deg=25.0, fov_deg=120.0,
              res_x=1920, res_y=1080, clip_end=500.0):
        """
        Creates a surveillance-style camera looking down at the road.

        In Blender's coordinate system:
          - Camera looks along local -Z
          - Rotation X=90° means looking along +Y (horizontal forward)
          - To look DOWN by pitch_deg, we subtract from 90°
        """
        scene = bpy.context.scene
        scene.render.resolution_x = res_x
        scene.render.resolution_y = res_y
        scene.render.resolution_percentage = 100

        # Get or create camera
        cam_obj = bpy.data.objects.get(self.CAM_NAME)
        if cam_obj:
            cam_data = cam_obj.data
        else:
            cam_data = bpy.data.cameras.new(self.CAM_NAME)
            cam_obj = bpy.data.objects.new(self.CAM_NAME, cam_data)
            scene.collection.objects.link(cam_obj)

        scene.camera = cam_obj

        # Optics — convert horizontal FOV to focal length
        # FOV = 2 * atan(sensor_width / (2 * focal_length))
        # focal_length = sensor_width / (2 * tan(FOV/2))
        sensor_w = 36.0  # 35mm full-frame equivalent
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.sensor_width = sensor_w
        fov_rad = math.radians(fov_deg)
        focal = sensor_w / (2.0 * math.tan(fov_rad / 2.0))
        cam_data.lens = max(2.5, focal)
        cam_data.clip_start = 0.1
        cam_data.clip_end = clip_end

        # Position: camera is at the road center, at pole height, at Y=0
        cam_obj.location = Vector((0.0, 0.0, height))

        # Rotation: look forward and down
        # Euler X: 90° = horizontal, 90-pitch = looking down by pitch degrees
        rot_x = math.radians(90.0 - pitch_deg)
        cam_obj.rotation_euler = Euler((rot_x, 0.0, 0.0), "XYZ")

        return cam_obj

    def load_reference_overlay(self, image_path, opacity=0.5):
        """Load reference frame as camera background image for visual calibration"""
        abs_path = bpy.path.abspath(image_path)
        if not abs_path or not os.path.exists(abs_path):
            print(f"[ADAS] Reference image not found: {image_path}")
            return

        cam_obj = bpy.data.objects.get(self.CAM_NAME)
        if not cam_obj:
            return

        cam_data = cam_obj.data
        cam_data.show_background_images = True
        cam_data.background_images.clear()

        bg = cam_data.background_images.new()
        bg.image = bpy.data.images.load(abs_path)
        bg.alpha = opacity
        bg.display_depth = "FRONT"
        bg.frame_method = "STRETCH"

        print(f"[ADAS] Reference overlay loaded: {os.path.basename(abs_path)}")
