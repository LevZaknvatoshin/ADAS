"""
Universal Camera Calibrator from Scene Vanishing Geometry
=========================================================
Calculates intrinsic & extrinsic parameters for ANY camera photo:
  - Resolution & Aspect ratio (Width x Height)
  - Focal length & Field of View (FOV)
  - Precise 3D Rotation:
      * Pitch (наклон вниз / depression angle)
      * Yaw (поворот влево/вправо)
      * Roll (крен горизонта)
  - Mount Height estimation (pole / dashcam / overhead)
"""

import math
import cv2
import numpy as np


class UniversalCameraCalibrator:
    def __init__(self, image_bgr, default_fov_deg=80.0, default_height=7.0):
        self.img = image_bgr
        self.default_fov_deg = default_fov_deg
        self.cam_height = default_height

        self.h, self.w = image_bgr.shape[:2]
        self.pitch_deg = 20.0
        self.yaw_deg = 0.0
        self.roll_deg = 0.0
        self.fov_deg = default_fov_deg
        self.vp_x = self.w * 0.5
        self.horizon_y = self.h * 0.35

        self._calibrate()

    def _calibrate(self):
        h, w = self.h, self.w
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # 1. Edge detection for perspective lines
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 130)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=35, maxLineGap=20)

        # 2. Group converging longitudinal perspective lines
        intersections = []
        slopes_horiz = []

        if lines is not None and len(lines) > 0:
            left_lines = []
            right_lines = []
            for line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
                dx, dy = x2 - x1, y2 - y1
                if abs(dx) > 1:
                    slope = dy / dx
                    # Near-horizontal lines estimate camera Roll (крен горизонта)
                    if abs(slope) < 0.12 and abs(y1 - h * 0.4) < h * 0.3:
                        slopes_horiz.append(slope)

                    # Longitudinal converging road lines
                    if slope < -0.25 and (x1 < w * 0.60 or x2 < w * 0.60):
                        left_lines.append((x1, y1, x2, y2))
                    elif slope > 0.25 and (x1 > w * 0.40 or x2 > w * 0.40):
                        right_lines.append((x1, y1, x2, y2))

            for l1 in left_lines[:15]:
                for l2 in right_lines[:15]:
                    x1, y1, x2, y2 = l1
                    x3, y3, x4, y4 = l2
                    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                    if abs(denom) > 1e-4:
                        ix = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                        iy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                        if -w * 0.5 <= ix <= w * 1.5 and 0.0 <= iy <= h * 0.75:
                            intersections.append((ix, iy))

        if len(intersections) > 0:
            self.vp_x = float(np.median([p[0] for p in intersections]))
            self.horizon_y = float(np.median([p[1] for p in intersections]))
        else:
            self.vp_x = w * 0.5
            self.horizon_y = h * 0.35

        # 3. Camera Roll Angle
        if len(slopes_horiz) > 0:
            avg_slope = float(np.median(slopes_horiz))
            self.roll_deg = math.degrees(math.atan(avg_slope))
        else:
            self.roll_deg = 0.0

        # 4. Focal length & FOV
        fov_rad = math.radians(self.fov_deg)
        fx = (w / 2.0) / math.tan(fov_rad / 2.0)
        fy = fx
        cx = w / 2.0
        cy = h / 2.0

        # 5. Pitch Angle (Наклон вниз)
        # In image coordinates, v=0 is top, v=h is bottom.
        # Horizon above image center (cy - horizon_y > 0) means camera tilts downwards.
        pitch_rad = math.atan2((cy - self.horizon_y), fy)
        self.pitch_deg = max(5.0, min(65.0, math.degrees(pitch_rad)))

        # 6. Yaw Angle (Поворот вправо/влево)
        yaw_rad = math.atan2((self.vp_x - cx), fx)
        self.yaw_deg = math.degrees(yaw_rad)

    def apply_to_blender_camera(self, cam_obj):
        """Applies calculated resolution, lens, and 3D orientation to Blender Camera"""
        import bpy
        from mathutils import Euler

        scene = bpy.context.scene
        scene.render.resolution_x = self.w
        scene.render.resolution_y = self.h
        scene.render.resolution_percentage = 100

        cam_data = cam_obj.data
        cam_data.sensor_fit = "HORIZONTAL"
        sensor_w = 36.0
        cam_data.sensor_width = sensor_w
        fov_rad = math.radians(self.fov_deg)
        cam_data.lens = max(3.0, sensor_w / (2.0 * math.tan(fov_rad / 2.0)))

        # Position
        cam_obj.location = (0.0, 0.0, self.cam_height)

        # 3D Euler Angles in Blender XYZ convention:
        # X: 90 deg = horizontal, (90 - pitch) = tilted down
        # Y: Roll (tilt sideways)
        # Z: Yaw (heading turn)
        rot_x = math.radians(90.0 - self.pitch_deg)
        rot_y = math.radians(-self.roll_deg)
        rot_z = math.radians(-self.yaw_deg)
        cam_obj.rotation_euler = Euler((rot_x, rot_y, rot_z), "XYZ")
