# -*- coding: utf-8 -*-
"""
road_infrastructure.py — Road Infrastructure Analyzer & Multi-Lane Modeling
==========================================================================
Extracts drivable carriageways, lanes, directions, shoulders, and sidewalks
from reconstructed 3D meshes (RoadCam 2.5D or Segmented Twin) and polygon data.

Preserves the existing scene generator completely.
"""

import os
import sys
import json
import math
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
import bpy
import bmesh
from mathutils import Vector, geometry
from mathutils.bvhtree import BVHTree


class RoadLane:
    """
    Represents a single traffic lane with centerline trajectory, edge boundaries,
    variable 3D lane width along road depth, and physical scale calibration.
    """

    def __init__(self, lane_id: int, carriageway_id: int, direction: int = 1, width: float = 3.5, name: str = ""):
        self.lane_id = lane_id
        self.carriageway_id = carriageway_id
        self.direction = direction  # +1 = forward (along +Y), -1 = oncoming (-Y)
        self.width = width
        self.name = name or f"Lane_{lane_id}"
        self.centerline_pts: List[Vector] = []
        self.left_edge_pts: List[Vector] = []
        self.right_edge_pts: List[Vector] = []
        self.widths_at_pts: List[float] = []

    def get_position_at_t(self, t: float) -> Vector:
        """Evaluates interpolated 3D position along lane centerline (0.0 <= t <= 1.0)."""
        if not self.centerline_pts:
            return Vector((0.0, 20.0, 0.0))
        if len(self.centerline_pts) == 1:
            return self.centerline_pts[0].copy()

        t_clamped = max(0.0, min(1.0, t))
        total_segs = len(self.centerline_pts) - 1
        seg_float = t_clamped * total_segs
        idx = min(int(seg_float), total_segs - 1)
        local_t = seg_float - idx
        return self.centerline_pts[idx].lerp(self.centerline_pts[idx + 1], local_t)

    def get_tangent_at_t(self, t: float) -> Vector:
        """Evaluates forward tangent vector along lane at t."""
        if len(self.centerline_pts) < 2:
            return Vector((0.0, float(self.direction), 0.0))
        t_clamped = max(0.0, min(1.0, t))
        total_segs = len(self.centerline_pts) - 1
        idx = min(int(t_clamped * total_segs), total_segs - 1)
        tangent = self.centerline_pts[idx + 1] - self.centerline_pts[idx]
        if tangent.length < 1e-5:
            return Vector((0.0, float(self.direction), 0.0))
        return tangent.normalized()

    def get_width_at_y(self, y: float) -> float:
        """Returns the interpolated 3D lane width at coordinate Y."""
        if not self.widths_at_pts or not self.centerline_pts:
            return self.width
        if len(self.centerline_pts) == 1:
            return self.widths_at_pts[0]
        pts_with_w = sorted(zip(self.centerline_pts, self.widths_at_pts), key=lambda it: it[0].y)
        if y <= pts_with_w[0][0].y:
            return pts_with_w[0][1]
        if y >= pts_with_w[-1][0].y:
            return pts_with_w[-1][1]
        for i in range(len(pts_with_w) - 1):
            p1, w1 = pts_with_w[i]
            p2, w2 = pts_with_w[i + 1]
            if p1.y <= y <= p2.y:
                dy = max(1e-5, p2.y - p1.y)
                factor = (y - p1.y) / dy
                return w1 + factor * (w2 - w1)
        return self.width

    def get_center_x_at_y(self, y: float) -> float:
        """Returns the interpolated 3D centerline X at coordinate Y."""
        if not self.centerline_pts:
            return 0.0
        pts = sorted(self.centerline_pts, key=lambda p: p.y)
        if y <= pts[0].y:
            return pts[0].x
        if y >= pts[-1].y:
            return pts[-1].x
        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i + 1]
            if p1.y <= y <= p2.y:
                dy = max(1e-5, p2.y - p1.y)
                factor = (y - p1.y) / dy
                return p1.x + factor * (p2.x - p1.x)
        return pts[0].x

    def get_target_vehicle_scale(self, y: float, base_model_width: float = 20.49, target_ratio: float = 0.52) -> float:
        """
        Calculates scale factor so vehicle occupies target_ratio (~50-55%) of the lane width
        at world coordinate Y.
        """
        w = self.get_width_at_y(y)
        return max(0.005, (target_ratio * w) / max(0.1, base_model_width))


class Carriageway:
    """Represents a physically or logically distinct roadway with multiple lanes."""

    def __init__(self, carriageway_id: int, name: str = "Carriageway", direction: int = 1):
        self.id = carriageway_id
        self.name = name
        self.direction = direction  # +1 = forward, -1 = oncoming
        self.lanes: List[RoadLane] = []
        self.center_x: float = 0.0
        self.width: float = 7.0
        self.left_boundary: float = -3.5
        self.right_boundary: float = 3.5
        self.has_median: bool = False
        self.has_sidewalk_left: bool = False
        self.has_sidewalk_right: bool = False
        self.pedestrian_crossings: List[Dict] = []

    def get_primary_lane(self) -> Optional[RoadLane]:
        """Returns primary driving lane (right-hand lane for forward, left-hand for oncoming)."""
        if not self.lanes:
            return None
        return self.lanes[-1] if self.direction == 1 and len(self.lanes) >= 2 else self.lanes[0]


class RoadAnalyzer:
    """
    Analyzes reconstructed 3D road surfaces, photo UV mappings, and semantic masks.
    Builds a high-level representation without altering existing scene objects.
    """

    def __init__(self, scene=None):
        self.scene = scene or bpy.context.scene
        self.bvh_tree: Optional[BVHTree] = None
        self.road_objects: List[bpy.types.Object] = []
        self.primary_road_obj: Optional[bpy.types.Object] = None
        self.reference_image_path: Optional[str] = None
        self.road_mask_path: Optional[str] = None
        self.road_polygon_path: Optional[str] = None

        self._find_road_surfaces()
        self._find_reference_data()
        self._build_bvh()

    def _find_road_surfaces(self):
        """Discovers road mesh objects generated by cv_builder."""
        self.road_objects = [
            o for o in self.scene.objects
            if o.type == "MESH" and o.name in bpy.data.objects and not o.name.startswith("ADAS_Road_StaticCollider") and (
                o.name.startswith("RoadCam_") or
                o.name.startswith("RoadScene_") or
                o.name.startswith("ADAS_Road") or
                o.name.startswith("ADAS_Carriageway") or
                o.name.startswith("Road_Asphalt") or
                o.name.startswith("ADAS_Env_") or
                o.name.startswith("ADAS_Environment") or
                o.get("role") in ("roadcam_2.5d_reconstruction", "road_surface", "road_surface_3d")
            )
        ]
        # Primary road mesh (RoadCam 2.5D or main scene reconstruction)
        for name in ("RoadCam_2.5D_SceneMesh", "RoadScene_2.5D_Reconstruction", "ADAS_Carriageway_Mesh"):
            obj = self.scene.objects.get(name)
            if obj:
                self.primary_road_obj = obj
                break
        if not self.primary_road_obj and self.road_objects:
            self.primary_road_obj = self.road_objects[0]

    def _find_reference_data(self):
        """Locates reference image, cache folder, road mask, and polygon metadata."""
        props = getattr(self.scene, "adas_props", None)
        ref = props.reference_image if props and props.reference_image else ""
        if ref and os.path.exists(bpy.path.abspath(ref)):
            self.reference_image_path = os.path.abspath(bpy.path.abspath(ref))
        elif self.primary_road_obj and self.primary_road_obj.data.materials:
            # Check image texture node on road mesh material
            mat = self.primary_road_obj.data.materials[0]
            if mat and mat.use_nodes:
                for n in mat.node_tree.nodes:
                    if n.type == "TEX_IMAGE" and n.image and n.image.filepath:
                        cand = bpy.path.abspath(n.image.filepath)
                        if os.path.exists(cand):
                            self.reference_image_path = os.path.abspath(cand)
                            break

        if self.reference_image_path:
            base_dir = os.path.dirname(self.reference_image_path)
            stem = os.path.splitext(os.path.basename(self.reference_image_path))[0]
            cache_dir = os.path.join(base_dir, f".cache_ai_{stem}")
            mask_cand = os.path.join(cache_dir, "road_mask.png")
            poly_cand = os.path.join(cache_dir, "road_polygon.json")
            if os.path.exists(mask_cand):
                self.road_mask_path = mask_cand
            if os.path.exists(poly_cand):
                self.road_polygon_path = poly_cand

    def _build_bvh(self):
        """Constructs an accurate BVH Tree from evaluated road mesh geometry for ground raycasting."""
        if not self.road_objects:
            return
        bm = bmesh.new()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for ro in self.road_objects:
            try:
                if not ro or ro.name not in bpy.data.objects:
                    continue
                eval_obj = ro.evaluated_get(depsgraph)
                me = eval_obj.to_mesh()
                me.transform(eval_obj.matrix_world)
                bm.from_mesh(me)
                eval_obj.to_mesh_clear()
            except (ReferenceError, KeyError):
                continue
        bm.verts.ensure_lookup_table()
        if len(bm.faces) > 0:
            self.bvh_tree = BVHTree.FromBMesh(bm)
        bm.free()

    def raycast_ground_z(self, x: float, y: float, default_z: float = 0.0) -> float:
        """
        Raycasts downwards to locate exact road surface Z coordinate at (x, y).
        Guarantees objects never sink under or hover above the road.
        """
        if self.bvh_tree:
            ray_origin = Vector((x, y, 50.0))
            ray_dir = Vector((0.0, 0.0, -1.0))
            hit_loc, hit_norm, face_idx, hit_dist = self.bvh_tree.ray_cast(ray_origin, ray_dir)
            if hit_loc is not None and hit_loc.z < 4.8:
                return hit_loc.z

        min_dist = float("inf")
        closest_z = default_z
        for ro in self.road_objects:
            try:
                if not ro or ro.name not in bpy.data.objects:
                    continue
                mw = ro.matrix_world
                for v in ro.data.vertices:
                    wco = mw @ v.co
                    if wco.z > 4.8:
                        continue
                    d2 = (wco.x - x) ** 2 + (wco.y - y) ** 2
                    if d2 < min_dist:
                        min_dist = d2
                        closest_z = wco.z
                        if d2 < 0.2:
                            return closest_z
            except (ReferenceError, KeyError):
                continue
        return closest_z

    def uv_to_world_co(self, u: float, v: float) -> Vector:
        """
        Converts normalized image coordinate (u, v) (where v=0 is bottom, v=1 is top)
        to world 3D position on the reconstructed road mesh.
        """
        if not self.primary_road_obj:
            return Vector((0.0, 18.0, 0.0))

        me = self.primary_road_obj.data
        mw = self.primary_road_obj.matrix_world
        verts = me.vertices
        n_verts = len(verts)

        # Fast direct grid resolution check for RoadCam 2.5D SceneMesh
        if self.primary_road_obj.name.startswith("RoadCam_") and n_verts > 100:
            scene = self.scene
            w = scene.render.resolution_x
            h = scene.render.resolution_y
            aspect = (w / float(h)) if h > 0 else 1.5
            grid_w = max(32, int(round(math.sqrt(n_verts * aspect))))
            grid_h = max(16, int(round(n_verts / float(grid_w))))
            if grid_w * grid_h == n_verts:
                i = int(round(max(0.0, min(1.0, u)) * (grid_w - 1)))
                v_img = 1.0 - max(0.0, min(1.0, v))
                j = int(round(max(0.0, min(1.0, v_img)) * (grid_h - 1)))
                idx = j * grid_w + i
                if 0 <= idx < n_verts:
                    return mw @ verts[idx].co

        # Fallback using bounding box & raycast
        x = (u - 0.5) * 12.0
        y = (v - 0.5) * 12.0
        z = self.raycast_ground_z(x, y)
        return Vector((x, y, z))


_GLOBAL_LANE_CACHE: Dict[Tuple, List['Carriageway']] = {}


def clear_lane_cache():
    """Clears the cached lane detections so that next call re-evaluates fresh."""
    global _GLOBAL_LANE_CACHE
    _GLOBAL_LANE_CACHE.clear()


class LaneDetector:
    """
    Detects lanes, carriageways, and traffic directions based on CV segmentation,
    lane markings, road mesh geometry, and scene context.
    """

    def __init__(self, analyzer: RoadAnalyzer):
        self.analyzer = analyzer
        self.carriageways: List[Carriageway] = []

    def analyze_lanes(self, polygon_data: Optional[Dict] = None, force_refresh: bool = False) -> List[Carriageway]:
        """
        Calculates lanes and carriageways using semantic segmentation, lane markings,
        and 3D road mesh projection. Uses memory caching for instant UI evaluation.
        """
        cache_key = (
            getattr(self.analyzer.scene, "name", "Scene"),
            self.analyzer.reference_image_path,
            getattr(self.analyzer.primary_road_obj, "name", "")
        )
        if not force_refresh and cache_key in _GLOBAL_LANE_CACHE:
            self.carriageways = _GLOBAL_LANE_CACHE[cache_key]
            return self.carriageways

        self.carriageways.clear()

        # 1. Try automated CV segmentation if reference image and OpenCV are available
        cv_success = self._analyze_via_cv_segmentation()
        if cv_success and self.carriageways:
            _GLOBAL_LANE_CACHE[cache_key] = self.carriageways
            return self.carriageways

        # 2. Geometric fallback if CV segmentation is unavailable
        self.carriageways = self._analyze_via_mesh_geometry(polygon_data)
        _GLOBAL_LANE_CACHE[cache_key] = self.carriageways
        return self.carriageways

    def _analyze_via_cv_segmentation(self) -> bool:
        """Analyzes road using semantic road mask and lane markings in image space."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return False

        img_path = self.analyzer.reference_image_path
        if not img_path or not os.path.exists(img_path):
            return False

        mask_path = self.analyzer.road_mask_path
        road_mask = None
        if mask_path and os.path.exists(mask_path):
            road_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return False
        h, w = img_bgr.shape[:2]

        # If mask is missing, generate fast adaptive asphalt segmentation
        if road_mask is None or road_mask.shape[:2] != (h, w):
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            road_cand = (sat < 65) & (val > 40) & (val < 220)
            road_cand[:int(h * 0.35), :] = False
            road_mask = (road_cand.astype(np.uint8)) * 255
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k)
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, k)

        # Detect lane markings (white in HLS + yellow in HSV)
        road_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=road_mask)
        hls = cv2.cvtColor(road_bgr, cv2.COLOR_BGR2HLS)
        l_chan = hls[:, :, 1]
        white_mask = cv2.inRange(l_chan, 175, 255)
        hsv = cv2.cvtColor(road_bgr, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))
        marking_mask = cv2.bitwise_and(cv2.bitwise_or(white_mask, yellow_mask), road_mask)

        # Determine ground road mesh longitudinal boundaries (Y axis on flat asphalt)
        y_min, y_max = 17.65, 19.35
        if self.analyzer.primary_road_obj:
            try:
                mw = self.analyzer.primary_road_obj.matrix_world
                ground_pts = [mw @ v.co for v in self.analyzer.primary_road_obj.data.vertices if (mw @ v.co).z < 1.85]
                if ground_pts:
                    ys = [p.y for p in ground_pts]
                    y_min = float(np.percentile(ys, 1))
                    y_max = float(np.percentile(ys, 98))
            except Exception:
                pass

        # Collect 3D road surface points from CV segmentation mask
        pts_fwd_left, pts_fwd_right = [], []
        pts_onc_left, pts_onc_right = [], []
        pts_all_left, pts_all_right = [], []

        # Scan rows with road mask from near camera towards horizon
        road_rows = np.where(np.sum(road_mask > 0, axis=1) > 20)[0]
        if len(road_rows) < 10:
            return False

        min_py, max_py = road_rows[0], road_rows[-1]
        sample_rows = np.linspace(max_py, min_py, 28).astype(int)

        for py in sample_rows:
            v = 1.0 - (py / float(h))
            xs = np.where(road_mask[py, :] > 0)[0]
            if len(xs) < 15:
                continue

            # Sample 3D road points along row
            step = max(4, len(xs) // 25)
            row_3d = []
            for px in xs[::step]:
                u = px / float(w - 1)
                p = self.analyzer.uv_to_world_co(u, v)
                if p.z < 3.0 and (y_min - 0.2) <= p.y <= (y_max + 0.2):
                    row_3d.append(p)

            if not row_3d:
                continue

            y_val = float(np.median([p.y for p in row_3d]))
            xs_3d = [p.x for p in row_3d]
            pts_all_left.append((min(xs_3d), y_val))
            pts_all_right.append((max(xs_3d), y_val))

            # Detect split carriageways (divided highway)
            xs_fwd = [p.x for p in row_3d if p.x > 0.25]
            xs_onc = [p.x for p in row_3d if p.x < -0.25]
            if xs_fwd and len(xs_fwd) >= 2:
                pts_fwd_left.append((min(xs_fwd), y_val))
                pts_fwd_right.append((max(xs_fwd), y_val))
            if xs_onc and len(xs_onc) >= 2:
                pts_onc_left.append((min(xs_onc), y_val))
                pts_onc_right.append((max(xs_onc), y_val))

        # Check if divided highway: both sides have sufficient samples & total width >= 8.5m
        is_divided = (len(pts_fwd_left) >= 5 and len(pts_onc_left) >= 5)
        if is_divided:
            span_f = max(p[0] for p in pts_fwd_right) - min(p[0] for p in pts_fwd_left)
            span_o = max(p[0] for p in pts_onc_right) - min(p[0] for p in pts_onc_left)
            if (span_f + span_o) < 6.0:
                is_divided = False

        if is_divided:
            # 1. Forward Carriageway (Right roadway, traffic away from camera, direction +1)
            cw_fwd = Carriageway(carriageway_id=1, name="Forward_Roadway", direction=1)
            # 2. Oncoming Carriageway (Left roadway, traffic toward camera, direction -1)
            cw_onc = Carriageway(carriageway_id=2, name="Oncoming_Roadway", direction=-1)

            next_id = self._build_lanes_from_boundaries(
                cw_fwd, pts_fwd_left, pts_fwd_right, y_min, y_max, start_lane_idx=1, is_divided=True, is_oncoming=False
            )
            self._build_lanes_from_boundaries(
                cw_onc, pts_onc_left, pts_onc_right, y_min, y_max, start_lane_idx=next_id, is_divided=True, is_oncoming=True,
                ref_forward_cw=cw_fwd
            )
            self.carriageways.extend([cw_fwd, cw_onc])
        else:
            # Single continuous roadway
            cw = Carriageway(carriageway_id=1, name="Main_Roadway", direction=1)
            self._build_lanes_from_boundaries(
                cw, pts_all_left, pts_all_right, y_min, y_max, start_lane_idx=1, is_divided=False, is_oncoming=False
            )
            self.carriageways.append(cw)

        return bool(self.carriageways and any(cw.lanes for cw in self.carriageways))

    def _build_lanes_from_boundaries(
        self,
        cw: Carriageway,
        pts_left: List[Tuple[float, float]],
        pts_right: List[Tuple[float, float]],
        y_min: float,
        y_max: float,
        start_lane_idx: int = 1,
        is_divided: bool = False,
        is_oncoming: bool = False,
        ref_forward_cw: Optional[Carriageway] = None,
    ) -> int:
        """
        Fits smooth boundary curves, calculates dynamic lane count and widths,
        and constructs monotonic, 100% smooth RoadLane instances.
        """
        if not pts_left or not pts_right:
            return start_lane_idx

        # 1. Fit smooth boundary polynomials
        ys_l = [p[1] for p in pts_left]
        xs_l = [p[0] for p in pts_left]
        ys_r = [p[1] for p in pts_right]
        xs_r = [p[0] for p in pts_right]

        deg_l = 1
        deg_r = 1

        poly_l = np.poly1d(np.polyfit(ys_l, xs_l, deg=deg_l))
        poly_r = np.poly1d(np.polyfit(ys_r, xs_r, deg=deg_r))

        # Sample longitudinal points along road
        num_pts = 24
        y_samples = np.linspace(y_min, y_max, num_pts)

        # Average carriageway width evaluation
        raw_widths = [poly_r(y) - poly_l(y) for y in y_samples]
        avg_w = float(np.median([w for w in raw_widths if w > 1.0] or [5.5]))

        # For oncoming roadway, ensure width matches forward roadway if far distance was masked
        if is_oncoming and is_divided and ref_forward_cw and ref_forward_cw.lanes:
            ref_w = sum(l.width for l in ref_forward_cw.lanes)
            avg_w = max(avg_w, ref_w)
            # Symmetric linear convergence mirroring forward carriageway
            dy_span = max(0.5, y_max - y_min)
            m_onc = (-0.80 - (-0.40)) / dy_span
            b_onc = -0.40 - m_onc * y_min
            poly_r = np.poly1d([m_onc, b_onc])
            poly_l = np.poly1d([m_onc, b_onc - avg_w])

        # 2. Dynamic lane count determination (based on physical standard ~3.5m lane width)
        if avg_w < 5.0:
            num_lanes = 1
        elif avg_w < 8.5:
            num_lanes = 2
        elif avg_w < 12.0:
            num_lanes = 3
        else:
            num_lanes = max(3, int(round(avg_w / 3.5)))

        # 3. Build each lane
        next_id = start_lane_idx
        created_lanes = []

        # Direction of travel: forward is +1, oncoming is -1
        lane_dir = -1 if is_oncoming else 1

        # Sample points ordered in travel direction
        travel_ys = list(reversed(y_samples)) if lane_dir == -1 else list(y_samples)

        for j in range(num_lanes):
            lane_num = next_id
            next_id += 1

            # Lateral fraction alpha across carriageway width
            # For forward: j=0 is rightmost lane (alpha close to 1.0)
            # For oncoming: j=0 is rightmost for the oncoming driver
            if is_oncoming:
                alpha = (j + 0.5) / float(num_lanes)
            else:
                alpha = 1.0 - (j + 0.5) / float(num_lanes)

            # Descriptive Russian naming
            dir_str = "(встречная)" if lane_dir == -1 else "(попутная)"
            if num_lanes == 1:
                lname = f"Полоса {lane_num}: Единственная {dir_str}"
            elif num_lanes == 2:
                pos = "Правая" if j == 0 else "Левая"
                lname = f"Полоса {lane_num}: {pos} {dir_str}"
            elif num_lanes == 3:
                pos = "Правая" if j == 0 else ("Средняя" if j == 1 else "Левая")
                lname = f"Полоса {lane_num}: {pos} {dir_str}"
            else:
                lname = f"Полоса {lane_num}: {j+1}-я {dir_str}"

            lane = RoadLane(
                lane_id=lane_num,
                carriageway_id=cw.id,
                direction=lane_dir,
                width=avg_w / float(num_lanes),
                name=lname,
            )

            for y in travel_ys:
                xl = float(poly_l(y))
                xr = float(poly_r(y))
                cw_w = max(1.5, xr - xl)
                x = xl + alpha * cw_w
                z = self.analyzer.raycast_ground_z(x, y)
                if z > 1.95:
                    continue
                lane_w = cw_w / float(num_lanes)

                lane.centerline_pts.append(Vector((x, y, z)))
                lane.widths_at_pts.append(float(lane_w))

            if lane.widths_at_pts:
                lane.width = sum(lane.widths_at_pts) / len(lane.widths_at_pts)

            created_lanes.append(lane)

        cw.lanes.extend(created_lanes)
        return next_id

    def _analyze_via_mesh_geometry(self, polygon_data: Optional[Dict] = None) -> List[Carriageway]:
        """Geometric fallback when CV segmentation is not available."""
        all_xs, all_ys = [], []
        for ro in self.analyzer.road_objects:
            try:
                if not ro or ro.name not in bpy.data.objects:
                    continue
                mw = ro.matrix_world
                for v in ro.data.vertices:
                    wco = mw @ v.co
                    if wco.z < 1.85:
                        all_xs.append(wco.x)
                        all_ys.append(wco.y)
            except (ReferenceError, KeyError):
                continue

        if not all_xs:
            min_x, max_x = -4.0, 4.0
            min_y, max_y = 17.65, 19.35
        else:
            min_x, max_x = min(all_xs), max(all_xs)
            min_y = float(np.percentile(all_ys, 1))
            max_y = float(np.percentile(all_ys, 98))

        pts_l = [(min_x, y) for y in np.linspace(min_y, max_y, 16)]
        pts_r = [(max_x, y) for y in np.linspace(min_y, max_y, 16)]

        cw = Carriageway(carriageway_id=1, name="Main_Roadway", direction=1)
        self._build_lanes_from_boundaries(
            cw, pts_l, pts_r, min_y, max_y, start_lane_idx=1, is_divided=False, is_oncoming=False
        )
        self.carriageways.append(cw)
        return self.carriageways

    def get_all_lanes(self) -> List[RoadLane]:
        """Returns flat list of all detected lanes across all carriageways."""
        if not self.carriageways:
            self.analyze_lanes()
        all_l = []
        for cw in self.carriageways:
            all_l.extend(cw.lanes)
        return all_l

    def get_lane_by_id(self, lane_id_str: str) -> Optional[RoadLane]:
        """Finds lane by identifier string e.g. 'LANE_1', 'lane_1', or matching name."""
        all_l = self.get_all_lanes()
        if not all_l:
            return None
        # Direct index lookup e.g. 'LANE_1' -> index 0
        if lane_id_str.startswith("LANE_"):
            try:
                idx = int(lane_id_str.split("_")[1]) - 1
                if 0 <= idx < len(all_l):
                    return all_l[idx]
            except Exception:
                pass
        for idx, l in enumerate(all_l):
            if lane_id_str in (f"lane_{idx+1}", f"LANE_{idx+1}", str(l.lane_id), l.name):
                return l
        for l in all_l:
            if lane_id_str.lower() in l.name.lower():
                return l
        return all_l[0]

    def get_ui_lane_items(self) -> List[Tuple[str, str, str]]:
        """Returns list of (id, label, description) for Blender EnumProperty dropdown."""
        all_l = self.get_all_lanes()
        if not all_l:
            return [("LANE_1", "Полоса 1: Правая (попутная)", "Стандартная полоса движения")]
        items = []
        for idx, l in enumerate(all_l):
            dir_str = "Попутно" if l.direction == 1 else "Встречно"
            ident = f"LANE_{idx+1}"
            label = f"{l.name} [{dir_str}, ~{l.width:.1f}м]"
            desc = f"Полоса движения: {l.name}, средняя ширина {l.width:.2f}м, направление: {dir_str}"
            items.append((ident, label, desc))
        return items

    def get_primary_forward_lane(self) -> Optional[RoadLane]:
        """Returns the primary forward driving lane."""
        if not self.carriageways:
            self.analyze_lanes()
        for cw in self.carriageways:
            if cw.direction == 1 and cw.lanes:
                return cw.lanes[0]
        return self.carriageways[0].lanes[0] if (self.carriageways and self.carriageways[0].lanes) else None

    def get_primary_oncoming_lane(self) -> Optional[RoadLane]:
        """Returns the primary oncoming lane."""
        if not self.carriageways:
            self.analyze_lanes()
        for cw in self.carriageways:
            for l in cw.lanes:
                if l.direction == -1:
                    return l
        return None

    def get_lane_at(self, x: float, y: float) -> Optional[RoadLane]:
        """Finds closest lane to coordinates (x, y)."""
        if not self.carriageways:
            self.analyze_lanes()
        best_lane = None
        min_dist = float("inf")
        for cw in self.carriageways:
            for lane in cw.lanes:
                lx = lane.get_center_x_at_y(y)
                d = abs(x - lx)
                if d < min_dist:
                    min_dist = d
                    best_lane = lane
        return best_lane

    def get_lane_width_at(self, x: float, y: float) -> float:
        """Evaluates lane width at position (x, y)."""
        lane = self.get_lane_at(x, y)
        if lane:
            return lane.get_width_at_y(y)
        return 3.5

    def get_orthogonal_width_at(self, x: float, y: float) -> float:
        """
        Calculates true orthogonal lane width perpendicular to trajectory tangent.
        Eliminates perspective shear and diagonal road angle exaggeration.
        """
        lane = self.get_lane_at(x, y)
        if not lane or not lane.centerline_pts or len(lane.centerline_pts) < 2:
            return 3.5

        pts = lane.centerline_pts
        best_i = 0
        min_dy = float("inf")
        for i in range(len(pts) - 1):
            mid_y = (pts[i].y + pts[i + 1].y) * 0.5
            dy = abs(y - mid_y)
            if dy < min_dy:
                min_dy = dy
                best_i = i

        t_vec = pts[best_i + 1] - pts[best_i]
        t_len = math.hypot(t_vec.x, t_vec.y)
        if t_len < 1e-4:
            return lane.get_width_at_y(y)

        nx = -t_vec.y / t_len
        cos_theta = abs(nx)
        raw_w = lane.get_width_at_y(y)
        if cos_theta > 0.15:
            return max(2.5, min(6.0, raw_w * cos_theta))
        return raw_w

    def create_3d_lane_curves(self, scene=None, coll_name: str = "ADAS_Lane_Lines") -> List[bpy.types.Object]:
        """
        Rollback: lane curves are disabled per user request.
        Cleans up any existing lane guide curves and returns empty list.
        """
        if scene is None:
            scene = self.analyzer.scene or bpy.context.scene

        coll = bpy.data.collections.get(coll_name)
        if coll:
            for obj in list(coll.objects):
                try:
                    c_data = obj.data if obj.type == "CURVE" else None
                    bpy.data.objects.remove(obj, do_unlink=True)
                    if c_data and c_data.users == 0:
                        bpy.data.curves.remove(c_data)
                except Exception:
                    pass
            try:
                bpy.data.collections.remove(coll)
            except Exception:
                pass
        return []

    def get_target_scale_for_actor(
        self,
        obj_type: str,
        subtype: str = "DEFAULT",
        x: float = 0.0,
        y: float = 0.0,
        base_model_scale: float = 1.0,
    ) -> float:
        """
        Calculates exact physical 3D scale so vehicle occupies ~50-55% of the detected
        orthogonal lane width at position (x, y).
        """
        lane_w = self.get_orthogonal_width_at(x, y)

        if obj_type == "VEHICLE":
            # Real car width is ~1.82m, real lane ~3.5m -> ratio ~0.53
            target_car_w = 0.53 * lane_w
            base_mesh_w = 20.4902
            return max(0.005, target_car_w / base_mesh_w)

        elif obj_type == "PERSON":
            # Person occupies ~15% of lane width
            ref_lane_w = 3.5
            ratio = max(0.4, min(1.6, lane_w / ref_lane_w))
            return 1.0 * ratio

        elif obj_type == "OBSTACLE":
            # Cone/barrel occupies ~10-12% of lane width
            ref_lane_w = 3.5
            ratio = max(0.4, min(1.6, lane_w / ref_lane_w))
            st = subtype.upper()
            defaults = {"CONE": 1.3, "BARREL": 0.70, "BOX": 0.10, "TIRE": 0.25}
            base = defaults.get(st, 0.10 if st == "DEFAULT" else 0.4)
            return base * ratio

        return 1.0


class CollisionBuilder:
    """
    Creates lightweight, accurate static physical collision meshes for road surfaces,
    trottoirs, and guardrails without degrading visual meshes.
    """

    COLLISION_COLLECTION_NAME = "ADAS_Physics_Colliders"

    def __init__(self, scene=None):
        self.scene = scene or bpy.context.scene

    def get_or_create_collider_collection(self) -> bpy.types.Collection:
        coll = bpy.data.collections.get(self.COLLISION_COLLECTION_NAME)
        if not coll:
            coll = bpy.data.collections.new(self.COLLISION_COLLECTION_NAME)
            self.scene.collection.children.link(coll)
        return coll

    def build_road_collision(self, road_objects: List[bpy.types.Object]) -> Optional[bpy.types.Object]:
        """
        Creates or updates a dedicated static collision mesh representing the road surface.
        Configures Blender Rigid Body (Type=PASSIVE, Shape=MESH, Friction=0.85).
        """
        if not road_objects:
            return None

        coll = self.get_or_create_collider_collection()
        collider_name = "ADAS_Road_StaticCollider"

        # Remove previous collider if present
        if collider_name in bpy.data.objects:
            old_obj = bpy.data.objects[collider_name]
            bpy.data.objects.remove(old_obj, do_unlink=True)

        # Merge evaluated road meshes into clean collision geometry
        bm = bmesh.new()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for ro in road_objects:
            try:
                if not ro or ro.name not in bpy.data.objects:
                    continue
                eval_obj = ro.evaluated_get(depsgraph)
                me = eval_obj.to_mesh()
                me.transform(eval_obj.matrix_world)
                bm.from_mesh(me)
                eval_obj.to_mesh_clear()
            except (ReferenceError, KeyError):
                continue

        # Simplify collision geometry: remove loose vertices and double points
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.04)
        bm.verts.ensure_lookup_table()

        # Keep faces whose normal points generally upwards (ground surface)
        ground_faces = [f for f in bm.faces if f.normal.z > -0.2]
        if ground_faces and len(ground_faces) < len(bm.faces):
            del_faces = [f for f in bm.faces if f not in ground_faces]
            bmesh.ops.delete(bm, geom=del_faces, context="FACES")

        col_mesh = bpy.data.meshes.new(collider_name + "_Mesh")
        bm.to_mesh(col_mesh)
        bm.free()

        col_obj = bpy.data.objects.new(collider_name, col_mesh)
        coll.objects.link(col_obj)

        # Set invisible in render and wire display
        col_obj.display_type = "WIRE"
        col_obj.hide_render = True
        col_obj["role"] = "physics_road_surface"

        # Setup Blender Rigid Body Physics
        self._ensure_rigidbody_world()
        bpy.context.view_layer.objects.active = col_obj
        col_obj.select_set(True)

        if not col_obj.rigid_body:
            bpy.ops.rigidbody.object_add(type="PASSIVE")

        rb = col_obj.rigid_body
        rb.collision_shape = "MESH"
        rb.mesh_source = "BASE"
        rb.friction = 0.85
        rb.restitution = 0.08  # asphalt damping

        col_obj.select_set(False)
        return col_obj

    def _ensure_rigidbody_world(self):
        """Ensures active scene has a configured RigidBodyWorld."""
        if not self.scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()
        rbw = self.scene.rigidbody_world
        rbw.substeps_per_frame = 10
        rbw.solver_iterations = 20
        if rbw.point_cache:
            rbw.point_cache.frame_start = self.scene.frame_start
            rbw.point_cache.frame_end = self.scene.frame_end


def np_linspace(start: float, stop: float, num: int) -> List[float]:
    """Lightweight pure-python linspace."""
    if num <= 1:
        return [start]
    step = (stop - start) / float(num - 1)
    return [start + i * step for i in range(num)]
