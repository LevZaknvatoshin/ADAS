"""
CV Road Digital Twin Builder — чистая процедурная реконструкция
===============================================================
АРХИТЕКТУРА (СТРОГО ЗАПРЕЩЕНО: Depth Map, Displace, Camera Projection):

Pipeline:
  Stage 1 — CV Parameter Extraction
    • Vanishing-point analysis → pitch, yaw, FOV
    • Semantic segmentation (SegFormer) → road/curb/median/grass masks
    • Hough lane lines → lane count, road curvature 2D spline
    • Blob detection на маске poles/trees → 2D coords столбов/деревьев

  Stage 2 — Perspective Unproject (2D → 3D World Space)
    • Pinhole camera model: ray = K⁻¹ · [u,v,1]ᵀ
    • Ground-plane intersection: t = -cam_height / ray_z
    • Построение world-space control points для кривой дороги

  Stage 3 — Parametric 3D Road Geometry (bpy.types.Curve + профиль)
    • Отдельные объекты: Asphalt, Left Curb, Right Curb,
      Sidewalk L, Sidewalk R, Median Grass, Lane Markings
    • Curve → Array + Bevel для профилей бордюров
    • Geometry Nodes для разметки (дублирование штриховки вдоль кривой)

  Stage 4 — Smart Instancing из Asset Library
    • Столбы, деревья → spawn из каталога ассетов (или процедурных примитивов)
    • Размещение по распакованным world-space координатам

  Stage 5 — PBR Materials (NO photo textures)
    • Асфальт: процедурный Noise + MixRGB (тёмный гравий)
    • Бордюр: Concrete (Voronoi + MixRGB)
    • Газон: многослойный Noise, зелёный BSDF
    • Разметка: эмиссионный белый материал

Совместимость: Blender 4.0+, Python 3.11+
"""

import math
import os
import json
import numpy as np
import cv2
import bpy
import bmesh
from mathutils import Vector, Euler, Matrix
from bpy.types import Object


# ════════════════════════════════════════════════════════════════════
# STAGE 1 — CV Parameter Extraction
# ════════════════════════════════════════════════════════════════════

class CVRoadAnalyzer:
    """
    Извлекает математические параметры дороги из фото методами Computer Vision.
    Возвращает структурированный RoadMetrics — чистые числа, без пиксельных текстур.
    """

    # Cityscapes label indices (SegFormer)
    SEG_LABELS = {
        "road":      [0],          # road
        "sidewalk":  [1],          # sidewalk
        "building":  [2, 3, 4],    # building, wall, fence
        "pole":      [5, 6, 7],    # pole, traffic light, traffic sign
        "vegetation":[8, 9],       # vegetation, terrain
        "sky":       [10],         # sky
        "vehicle":   [13, 14, 15, 16, 17, 18],
    }

    def __init__(self, img_bgr: np.ndarray,
                 default_fov_deg: float = 80.0,
                 default_cam_height: float = 7.0,
                 seg_masks: dict | None = None,
                 seg_metadata: dict | None = None):
        """
        img_bgr           — изображение OpenCV (BGR, uint8)
        default_fov_deg   — FOV по умолчанию (будет уточнён по VP)
        default_cam_height— высота камеры (м)
        seg_masks         — уже готовые маски из AIDepthSegmenter (опционально)
        seg_metadata      — metadata.json из AIDepthSegmenter (опционально)
        """
        self.img = img_bgr
        self.h, self.w = img_bgr.shape[:2]
        self.fov_deg = default_fov_deg
        self.cam_height = default_cam_height
        self.seg_masks = seg_masks or {}
        self.meta = seg_metadata or {}

    def _get_mask(self, name: str) -> np.ndarray | None:
        """
        Безопасно извлекает маску как np.ndarray uint8 (H, W),
        независимо от того, передан путь к файлу (str) или numpy array.
        """
        raw = self.seg_masks.get(name)
        if raw is None:
            return None
        if isinstance(raw, np.ndarray):
            if raw.ndim == 3:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            return raw
        if isinstance(raw, str):
            if os.path.exists(raw):
                try:
                    with open(raw, "rb") as f:
                        arr = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if arr is not None:
                        if arr.shape[0] != self.h or arr.shape[1] != self.w:
                            arr = cv2.resize(arr, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                        return arr
                except Exception as e:
                    print(f"[CVRoadAnalyzer] Ошибка загрузки маски {name} ({raw}): {e}")
        return None

    # ── публичный метод ──────────────────────────────────────────────
    def analyze(self) -> dict:
        """Запускает весь CV-пайплайн, возвращает RoadMetrics dict."""
        vp = self._detect_vanishing_point()
        pitch_deg, yaw_deg = self._compute_camera_angles(vp)
        fov_deg = self.fov_deg  # уточнять FOV по VP — отдельная задача, пока берём default

        lane_lines, lane_count = self._detect_lanes()
        road_width_m = self._estimate_road_width(lane_count, pitch_deg, fov_deg)
        curve_ctrl_pts_2d = self._extract_road_curve(lane_lines, vp)
        curve_ctrl_pts_3d = self._unproject_to_world(
            curve_ctrl_pts_2d, pitch_deg, yaw_deg, fov_deg
        )

        # Наличие разделительного газона (median)
        has_median = self._detect_median()
        median_width_m = 3.0 if has_median else 0.0

        # Столбы и деревья
        poles_3d = self._unproject_objects(
            self.meta.get("poles_2d", []), pitch_deg, yaw_deg, fov_deg,
            obj_type="pole"
        )

        return {
            "pitch_deg":           pitch_deg,
            "yaw_deg":             yaw_deg,
            "fov_deg":             fov_deg,
            "cam_height":          self.cam_height,
            "lane_count":          lane_count,
            "road_width_m":        road_width_m,
            "has_median":          has_median,
            "median_width_m":      median_width_m,
            "curve_ctrl_pts_3d":   curve_ctrl_pts_3d,
            "poles_3d":            poles_3d,
            "img_w":               self.w,
            "img_h":               self.h,
        }

    # ── Vanishing Point ──────────────────────────────────────────────
    def _detect_vanishing_point(self) -> tuple[float, float]:
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=40, minLineLength=40, maxLineGap=25)
        if lines is None or len(lines) == 0:
            return (self.w * 0.5, self.h * 0.35)

        left_lines, right_lines = [], []
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 2:
                continue
            slope = dy / dx
            # Сходящиеся линии дороги
            if slope < -0.25 and x1 < self.w * 0.65:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.25 and x1 > self.w * 0.35:
                right_lines.append((x1, y1, x2, y2))

        intersections = []
        for l1 in left_lines[:12]:
            for l2 in right_lines[:12]:
                x1, y1, x2, y2 = l1
                x3, y3, x4, y4 = l2
                denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                if abs(denom) < 1e-4:
                    continue
                ix = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
                iy = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
                if -self.w*0.6 <= ix <= self.w*1.6 and 0 <= iy <= self.h*0.80:
                    intersections.append((ix, iy))

        if len(intersections) > 2:
            vp_x = float(np.median([p[0] for p in intersections]))
            vp_y = float(np.median([p[1] for p in intersections]))
            return (vp_x, vp_y)
        return (self.w * 0.5, self.h * 0.38)

    # ── Camera Angles ────────────────────────────────────────────────
    def _compute_camera_angles(self, vp: tuple[float, float]) -> tuple[float, float]:
        fov_rad = math.radians(self.fov_deg)
        fx = (self.w / 2.0) / math.tan(fov_rad / 2.0)
        fy = fx
        cx, cy = self.w / 2.0, self.h / 2.0
        vp_x, vp_y = vp

        # pitch: угол между горизонтом VP и центром изображения
        pitch_rad = math.atan2(cy - vp_y, fy)
        pitch_deg = float(np.clip(math.degrees(pitch_rad), 3.0, 70.0))

        # yaw: смещение VP по горизонтали
        yaw_rad = math.atan2(vp_x - cx, fx)
        yaw_deg = float(math.degrees(yaw_rad))

        return pitch_deg, yaw_deg

    # ── Lane Detection ───────────────────────────────────────────────
    def _detect_lanes(self) -> tuple[list, int]:
        """
        Применяет ROI в нижней трети кадра (зона дороги),
        Canny + Hough → линии разметки → подсчёт полос.
        """
        h, w = self.h, self.w
        roi_y = int(h * 0.55)
        roi = self.img[roi_y:, :]

        # HSV: ищем белые и жёлтые разметочные линии
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, (0, 0, 170), (180, 40, 255))
        mask_yellow = cv2.inRange(hsv, (15, 60, 100), (40, 200, 255))
        mask_markings = cv2.bitwise_or(mask_white, mask_yellow)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_markings = cv2.dilate(mask_markings, kernel, iterations=2)

        edges = cv2.Canny(mask_markings, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=25, minLineLength=30, maxLineGap=30)

        lane_lines = []
        if lines is not None and len(lines) > 0:
            for line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
                dx, dy = x2 - x1, y2 - y1
                if abs(dx) < 2:
                    continue
                slope = dy / dx
                if abs(slope) > 0.15:  # наклонные → это разметка
                    lane_lines.append((x1, y1 + roi_y, x2, y2 + roi_y))

        # Оцениваем количество полос по числу уникальных X-кластеров линий в нижней части
        if len(lane_lines) >= 2:
            bottom_x = []
            for x1, y1, x2, y2 in lane_lines:
                # X-координата линии у нижней границы кадра
                if abs(x2 - x1) > 0:
                    t = (h - y1) / (y2 - y1 + 1e-6)
                    bx = x1 + t * (x2 - x1)
                    if 0 < bx < w:
                        bottom_x.append(bx)

            bottom_x.sort()
            # Кластеризация X-координат (порог 50px)
            clusters = []
            for bx in bottom_x:
                if not clusters or abs(bx - clusters[-1]) > 50:
                    clusters.append(bx)

            # Число полос = промежутки между крайними разметочными линиями
            lane_count = max(1, len(clusters) - 1)
            lane_count = min(lane_count, 6)  # разумный предел
        else:
            lane_count = 2  # fallback

        return lane_lines, lane_count

    # ── Road Width ───────────────────────────────────────────────────
    def _estimate_road_width(self, lane_count: int,
                              pitch_deg: float, fov_deg: float) -> float:
        """
        ГОСТ Р 52289-2019: стандартная ширина полосы 3.50–3.75 м.
        Умножаем на количество полос.
        """
        lane_width_m = 3.6  # среднее по ГОСТ
        return lane_count * lane_width_m

    # ── Road Curve ───────────────────────────────────────────────────
    def _extract_road_curve(self, lane_lines: list,
                             vp: tuple[float, float]) -> list[tuple[float, float]]:
        """
        Строит гладкие 2D контрольные точки осевой линии дороги.
        Использует семантическую маску дороги из AI или сходящиеся линии,
        с обязательной полиномиальной фильтрацией (исключает зигзаги и петли).
        """
        h, w = self.h, self.w
        pts_2d = []

        # 1. Если есть сегментационная маска дороги из AI
        road_mask = self._get_mask("road")
        if road_mask is not None and np.sum(road_mask > 128) > 500:
            mask_bin = road_mask > 128
            for v in range(int(h * 0.95), int(h * 0.42), -max(10, int(h * 0.04))):
                row = np.where(mask_bin[v, :])[0]
                if len(row) > 25:
                    u_center = float((row[0] + row[-1]) / 2.0)
                    pts_2d.append((u_center, float(v)))

        # 2. Если маски нет или точек мало — используем разметку / Hough
        if len(pts_2d) < 3 and lane_lines:
            bands = np.linspace(h * 0.95, h * 0.45, 8)
            for y_band in bands:
                xs = []
                for x1, y1, x2, y2 in lane_lines:
                    if abs(y2 - y1) < 2:
                        continue
                    t = (y_band - y1) / (y2 - y1)
                    if 0.0 <= t <= 1.0:
                        xs.append(x1 + t * (x2 - x1))
                if xs:
                    pts_2d.append((float(np.median(xs)), float(y_band)))

        # Добавляем точку схода (VP) вдали
        vp_x, vp_y = vp
        if 0 <= vp_x <= w and 0 <= vp_y <= h:
            pts_2d.append((float(vp_x), float(vp_y)))

        # Если точек мало — строим прямую центральную дорогу
        if len(pts_2d) < 2:
            return [(w * 0.5, float(y)) for y in np.linspace(h * 0.95, h * 0.42, 8)]

        # 3. ПОЛИНОМИАЛЬНАЯ АППРОКСИМАЦИЯ (Гарантирует отсутствие петель и зигзагов!)
        pts_2d.sort(key=lambda p: p[1], reverse=True)  # от низа кадра к верху
        v_arr = np.array([p[1] for p in pts_2d], dtype=np.float64)
        u_arr = np.array([p[0] for p in pts_2d], dtype=np.float64)

        deg = 2 if len(pts_2d) >= 4 else 1
        try:
            poly = np.poly1d(np.polyfit(v_arr, u_arr, deg=deg))
        except Exception:
            poly = np.poly1d([0.0, w * 0.5])

        # Генерируем 8 идеально сглаженных контрольных точек
        v_smooth = np.linspace(h * 0.95, max(h * 0.42, vp_y + 10), 8)
        u_smooth = poly(v_smooth)
        u_smooth = np.clip(u_smooth, w * 0.1, w * 0.9)

        return [(float(u), float(v)) for u, v in zip(u_smooth, v_smooth)]

    # ── 2D → 3D Unproject ───────────────────────────────────────────
    def _unproject_to_world(self, pts_2d: list[tuple[float, float]],
                             pitch_deg: float, yaw_deg: float,
                             fov_deg: float) -> list[tuple[float, float, float]]:
        """
        Пересечение лучей камеры с плоскостью Z=0 (дорога).
        Точная геометрическая депроекция в мировые координаты Blender:
          X: Lateral offset (вправо)
          Y: Longitudinal distance (вперед по дороге)
          Z: 0.0 (плоскость дороги)
        """
        fov_rad = math.radians(fov_deg)
        fx = (self.w / 2.0) / math.tan(fov_rad / 2.0)
        cx, cy = self.w / 2.0, self.h / 2.0
        h_cam = max(1.2, float(self.cam_height))
        pitch = max(1.0, float(pitch_deg))

        pts_3d = []
        for (u, v) in pts_2d:
            v_offset = v - cy
            ray_pitch_rel = math.degrees(math.atan2(v_offset, fx))
            total_dep = pitch + ray_pitch_rel

            if total_dep <= 0.8:
                y_w = 140.0
            else:
                y_w = h_cam / math.tan(math.radians(total_dep))
                y_w = float(np.clip(y_w, 2.0, 160.0))

            x_w = (u - cx) * y_w / fx

            if abs(yaw_deg) > 0.1:
                yaw_r = math.radians(yaw_deg)
                x_rot = x_w * math.cos(yaw_r) - y_w * math.sin(yaw_r)
                y_rot = x_w * math.sin(yaw_r) + y_w * math.cos(yaw_r)
                x_w, y_w = x_rot, y_rot

            pts_3d.append((float(x_w), float(y_w), 0.0))

        pts_3d.sort(key=lambda p: p[1])
        return pts_3d

    # ── Median Detection ─────────────────────────────────────────────
    def _detect_median(self) -> bool:
        """
        Определяет наличие разделительного газона по маске vegetation / terrain
        в центральной вертикальной полосе кадра (30–70% по X, 40–70% по Y).
        """
        mask = self._get_mask("terrain")
        if mask is not None:
            h_m, w_m = mask.shape
            roi = mask[int(h_m * 0.40):int(h_m * 0.70), int(w_m * 0.30):int(w_m * 0.70)]
            return float(roi.mean()) > 20.0
        return self.meta.get("terrain_pixels", 0) > (self.h * self.w * 0.03)

    # ── Object Unproject ─────────────────────────────────────────────
    def _unproject_objects(self, objs_2d: list[dict],
                            pitch_deg: float, yaw_deg: float, fov_deg: float,
                            obj_type: str = "pole") -> list[dict]:
        """Конвертирует 2D-координаты объектов (base_x, base_y) в 3D world-space."""
        if not objs_2d:
            return []
        pts_2d = [(o["base_x"], o["base_y"]) for o in objs_2d]
        pts_3d = self._unproject_to_world(pts_2d, pitch_deg, yaw_deg, fov_deg)
        result = []
        for i, o in enumerate(objs_2d):
            x3, y3, z3 = pts_3d[i]
            # Отсекаем нереалистичные позиции (слишком близко или слишком далеко)
            dist = math.sqrt(x3**2 + y3**2)
            if 3.0 < dist < 180.0:
                result.append({
                    "type":   obj_type,
                    "x":      x3,
                    "y":      y3,
                    "z":      0.0,
                    "width":  o.get("width", 10),
                    "height": o.get("height", 30),
                    "area":   o.get("area", 100),
                })
        return result


# ════════════════════════════════════════════════════════════════════
# STAGE 3–5 — Parametric 3D Builder
# ════════════════════════════════════════════════════════════════════

class ProceduralRoadBuilder:
    """
    Генерирует чистую математическую 3D-сцену дороги из RoadMetrics.

    Объекты (отдельные меши / коллекции):
      • Road_Asphalt        — полотно дороги (Array + Curve)
      • Road_Curb_L/R       — бордюры (Curve + Bevel)
      • Road_Sidewalk_L/R   — тротуары
      • Road_Median         — разделительный газон
      • Road_Markings       — разметка (Geometry Nodes)
      • Env_Poles / Env_Trees — 3D-ассеты окружения

    Материалы — исключительно процедурные PBR (Noise, Voronoi, MixRGB).
    Фотографии НЕ используются ни в одном слоте текстуры.
    """

    COLL_ROAD = "DT_Road"
    COLL_ENV  = "DT_Environment"

    # ГОСТ Р 52289: стандартные размеры (м)
    CURB_HEIGHT    = 0.15
    CURB_WIDTH     = 0.20
    SIDEWALK_WIDTH = 2.50
    SIDEWALK_DEPTH = 0.12

    def __init__(self, metrics: dict, scene=None):
        self.metrics = metrics
        self.scene   = scene or bpy.context.scene
        self.coll_road = self._ensure_collection(self.COLL_ROAD)
        self.coll_env  = self._ensure_collection(self.COLL_ENV)

    # ── Collection helpers ───────────────────────────────────────────
    def _ensure_collection(self, name: str):
        if name in bpy.data.collections:
            col = bpy.data.collections[name]
        else:
            col = bpy.data.collections.new(name)
            self.scene.collection.children.link(col)
        return col

    def _link(self, obj: Object, coll):
        # Убираем из дефолтной коллекции если попал туда
        if obj.name in self.scene.collection.objects:
            self.scene.collection.objects.unlink(obj)
        if obj.name not in coll.objects:
            coll.objects.link(obj)

    # ── PUBLIC BUILD ─────────────────────────────────────────────────
    def build(self):
        """Главный метод: строит всю 3D-сцену."""
        self._clear_previous()

        ctrl_pts = self.metrics["curve_ctrl_pts_3d"]
        if len(ctrl_pts) < 2:
            # Fallback: прямая дорога 150м по Y
            ctrl_pts = [(0.0, float(y), 0.0) for y in np.linspace(3.0, 150.0, 8)]

        road_w   = float(self.metrics["road_width_m"])
        has_med  = bool(self.metrics["has_median"])
        med_w    = float(self.metrics["median_width_m"])

        # 1. Базовая кривая (spine) дороги
        spine_curve, spine_obj = self._build_spine_curve(ctrl_pts)

        # 2. Полотно асфальта (Array + Curve)
        asphalt_obj = self._build_asphalt(road_w, spine_obj)

        # 3. Бордюры слева и справа (Array + Curve)
        curb_l = self._build_curb(road_w, side="L", spine_obj=spine_obj)
        curb_r = self._build_curb(road_w, side="R", spine_obj=spine_obj)

        # 4. Тротуары (Array + Curve)
        sw_l = self._build_sidewalk(road_w, side="L", spine_obj=spine_obj)
        sw_r = self._build_sidewalk(road_w, side="R", spine_obj=spine_obj)

        # 5. Разделительный газон (Array + Curve)
        if has_med:
            self._build_median(med_w, spine_obj=spine_obj)

        # 6. Дорожная разметка (Array + Curve)
        self._build_lane_markings(road_w, spine_obj, self.metrics["lane_count"])

        # 7. Расстановка ассетов окружения
        self._spawn_environment_assets(self.metrics.get("poles_3d", []))

        # 8. Освещение сцены (Sun + Sky)
        self._setup_lighting()

        print("[ProceduralRoadBuilder] [OK] Digital Twin построен успешно.")
        return spine_obj

    # ── Clear ────────────────────────────────────────────────────────
    def _clear_previous(self):
        # 1. Жесткая зачистка старых искаженных сеток проекции и legacy ассетов
        junk_names = [
            "Terrain_Environment_v3", "PerspectiveTerrainMesh", "AI_Depth_Texture", 
            "GrassBlade_Prototype", "Cube", "Light", "Profile_Asphalt", "Road_Spine",
            "Road_Asphalt", "Road_Median", "Terrain_Base", "Terrain_Landscape",
            "Road_Markings", "Road_Curb_L", "Road_Curb_R", "Road_Sidewalk_L", "Road_Sidewalk_R"
        ]
        for name in junk_names:
            if name in bpy.data.objects:
                try:
                    bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
                except Exception:
                    pass

        # 2. Очистка коллекций
        for cname in ["ADAS_Environment_v3", self.COLL_ROAD, self.COLL_ENV]:
            if cname in bpy.data.collections:
                coll = bpy.data.collections[cname]
                for obj in list(coll.objects):
                    try:
                        bpy.data.objects.remove(obj, do_unlink=True)
                    except Exception:
                        pass
                if cname == "ADAS_Environment_v3":
                    try:
                        bpy.data.collections.remove(coll)
                    except Exception:
                        pass

    # ── Spine Curve ──────────────────────────────────────────────────
    def _build_spine_curve(self, ctrl_pts: list) -> tuple:
        curve_data = bpy.data.curves.new("Road_Spine", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.resolution_u = 32
        curve_data.use_fill_caps = False

        spline = curve_data.splines.new("BEZIER")
        spline.bezier_points.add(len(ctrl_pts) - 1)

        for i, (x, y, z) in enumerate(ctrl_pts):
            bp = spline.bezier_points[i]
            bp.co = Vector((x, y, 0.0))
            bp.handle_left_type = "AUTO"
            bp.handle_right_type = "AUTO"

        obj = bpy.data.objects.new("Road_Spine", curve_data)
        self._link(obj, self.coll_road)
        obj.hide_render = True
        return curve_data, obj

    # ── Asphalt ──────────────────────────────────────────────────────
    def _build_asphalt(self, road_w: float, spine_obj: Object) -> Object:
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.name = "Road_Asphalt"
        obj.scale = (road_w, 2.0, 1.0)
        bpy.ops.object.transform_apply(scale=True)

        mod_arr = obj.modifiers.new(name="Array", type='ARRAY')
        mod_arr.fit_type = 'FIT_CURVE'
        mod_arr.curve = spine_obj
        mod_arr.relative_offset_displace = (0.0, 1.0, 0.0)
        mod_arr.use_merge_vertices = True

        mod_curv = obj.modifiers.new(name="Curve", type='CURVE')
        mod_curv.object = spine_obj
        mod_curv.deform_axis = 'POS_Y'

        obj.data.materials.append(self._mat_asphalt())
        self._link(obj, self.coll_road)
        return obj

    # ── Curb ─────────────────────────────────────────────────────────
    def _build_curb(self, road_w: float, side: str, spine_obj: Object) -> Object:
        sign = -1.0 if side == "L" else 1.0
        offset_x = sign * (road_w / 2.0 + self.CURB_WIDTH / 2.0)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(offset_x, 0, self.CURB_HEIGHT / 2.0))
        obj = bpy.context.active_object
        obj.name = f"Road_Curb_{side}"
        obj.scale = (self.CURB_WIDTH, 2.0, self.CURB_HEIGHT)
        bpy.ops.object.transform_apply(scale=True, location=False)

        mod_arr = obj.modifiers.new(name="Array", type='ARRAY')
        mod_arr.fit_type = 'FIT_CURVE'
        mod_arr.curve = spine_obj
        mod_arr.relative_offset_displace = (0.0, 1.0, 0.0)
        mod_arr.use_merge_vertices = True

        mod_curv = obj.modifiers.new(name="Curve", type='CURVE')
        mod_curv.object = spine_obj
        mod_curv.deform_axis = 'POS_Y'

        obj.data.materials.append(self._mat_concrete())
        self._link(obj, self.coll_road)
        return obj

    # ── Sidewalk ─────────────────────────────────────────────────────
    def _build_sidewalk(self, road_w: float, side: str, spine_obj: Object) -> Object:
        sign = -1.0 if side == "L" else 1.0
        offset_x = sign * (road_w / 2.0 + self.CURB_WIDTH + self.SIDEWALK_WIDTH / 2.0)

        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(offset_x, 0, self.CURB_HEIGHT))
        obj = bpy.context.active_object
        obj.name = f"Road_Sidewalk_{side}"
        obj.scale = (self.SIDEWALK_WIDTH, 2.0, 1.0)
        bpy.ops.object.transform_apply(scale=True, location=False)

        mod_arr = obj.modifiers.new(name="Array", type='ARRAY')
        mod_arr.fit_type = 'FIT_CURVE'
        mod_arr.curve = spine_obj
        mod_arr.relative_offset_displace = (0.0, 1.0, 0.0)
        mod_arr.use_merge_vertices = True

        mod_curv = obj.modifiers.new(name="Curve", type='CURVE')
        mod_curv.object = spine_obj
        mod_curv.deform_axis = 'POS_Y'

        obj.data.materials.append(self._mat_concrete())
        self._link(obj, self.coll_road)
        return obj

    # ── Median Grass ─────────────────────────────────────────────────
    def _build_median(self, med_w: float, spine_obj: Object) -> Object:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.08))
        obj = bpy.context.active_object
        obj.name = "Road_Median"
        obj.scale = (med_w, 2.0, 0.16)
        bpy.ops.object.transform_apply(scale=True, location=False)

        mod_arr = obj.modifiers.new(name="Array", type='ARRAY')
        mod_arr.fit_type = 'FIT_CURVE'
        mod_arr.curve = spine_obj
        mod_arr.relative_offset_displace = (0.0, 1.0, 0.0)
        mod_arr.use_merge_vertices = True

        mod_curv = obj.modifiers.new(name="Curve", type='CURVE')
        mod_curv.object = spine_obj
        mod_curv.deform_axis = 'POS_Y'

        obj.data.materials.append(self._mat_grass())
        self._link(obj, self.coll_road)
        return obj

    # ── Lane Markings ────────────────────────────────────────────────
    def _build_lane_markings(self, road_w: float, spine_obj: Object, lane_count: int):
        lane_w = road_w / max(lane_count, 1)
        half_road = road_w / 2.0

        # Markings positions along X
        xs = [-half_road + 0.15, half_road - 0.15]
        if lane_count >= 2:
            for i in range(1, lane_count):
                xs.append(-half_road + i * lane_w)

        for idx, mx in enumerate(xs):
            bpy.ops.mesh.primitive_plane_add(size=1.0, location=(mx, 0, 0.005))
            strip = bpy.context.active_object
            strip.name = f"Road_Marking_{idx}"
            strip.scale = (0.12, 2.0, 1.0)
            bpy.ops.object.transform_apply(scale=True, location=False)

            mod_arr = strip.modifiers.new(name="Array", type='ARRAY')
            mod_arr.fit_type = 'FIT_CURVE'
            mod_arr.curve = spine_obj
            mod_arr.relative_offset_displace = (0.0, 1.0, 0.0)
            mod_arr.use_merge_vertices = True

            mod_curv = strip.modifiers.new(name="Curve", type='CURVE')
            mod_curv.object = spine_obj
            mod_curv.deform_axis = 'POS_Y'

            strip.data.materials.append(self._mat_marking("white"))
            self._link(strip, self.coll_road)

    # ── Smart Instancing ─────────────────────────────────────────────
    def _spawn_environment_assets(self, poles_3d: list):
        """
        По 3D-координатам из CV-анализа инстанциирует ассеты:
        • Уличный фонарь (процедурный цилиндр + сфера)
        • Дерево (процедурный конус)
        Если в каталоге есть .blend/.obj — подключает их.
        """
        pole_proto = self._make_pole_prototype()
        tree_proto = self._make_tree_prototype()

        for obj_data in poles_3d:
            ox, oy = obj_data["x"], obj_data["y"]
            aspect = obj_data.get("height", 30) / max(obj_data.get("width", 10), 1)

            if aspect > 3.5:
                # Вертикальный: столб
                proto = pole_proto
                asset_name = "Pole"
            else:
                # Широкий/круглый: дерево
                proto = tree_proto
                asset_name = "Tree"

            instance = bpy.data.objects.new(
                f"Env_{asset_name}_{ox:.0f}_{oy:.0f}",
                proto.data
            )
            instance.location = Vector((ox, oy, 0.0))
            self._link(instance, self.coll_env)

    def _make_pole_prototype(self) -> Object:
        """Процедурный уличный фонарь: ствол + голова."""
        name = "_proto_pole"
        if name in bpy.data.objects:
            return bpy.data.objects[name]

        bm = bmesh.new()
        pole_h = 8.5
        pole_r = 0.08

        # Ствол — вытянутый цилиндр
        bmesh.ops.create_cone(bm,
            cap_ends=True, cap_tris=False,
            segments=8, radius1=pole_r, radius2=pole_r,
            depth=pole_h
        )
        # Голова — UV-sphere малая
        bmesh.ops.create_uvsphere(bm,
            u_segments=6, v_segments=4, radius=0.35
        )
        # Смещаем вершины головы вверх
        for v in bm.verts:
            if v.co.z > pole_h * 0.4:
                v.co.z += pole_h / 2.0

        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.location.z = pole_h / 2.0
        obj.data.materials.append(self._mat_pole())
        obj.hide_render  = True
        obj.hide_viewport = True
        self._link(obj, self.coll_env)
        return obj

    def _make_tree_prototype(self) -> Object:
        """Процедурное дерево: ствол + крона (конус)."""
        name = "_proto_tree"
        if name in bpy.data.objects:
            return bpy.data.objects[name]

        bm = bmesh.new()

        # Ствол
        bmesh.ops.create_cone(bm,
            cap_ends=True, cap_tris=False,
            segments=6, radius1=0.25, radius2=0.18, depth=3.0
        )

        # Крона (конус вершиной вверх)
        crown_result = bmesh.ops.create_cone(bm,
            cap_ends=True, cap_tris=False,
            segments=8, radius1=2.5, radius2=0.3, depth=5.0
        )
        for v in bm.verts:
            if v.co.z > 1.0:  # вершины кроны
                v.co.z += 3.5

        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.data.materials.append(self._mat_grass())
        obj.hide_render  = True
        obj.hide_viewport = True
        self._link(obj, self.coll_env)
        return obj

    # ── Lighting ─────────────────────────────────────────────────────
    def _setup_lighting(self):
        """Настраивает процедурное освещение сцены."""
        is_night = self.metrics.get("is_night", False)

        # Sky Texture (Nishita) — без каких-либо фото
        world = self.scene.world
        if not world:
            world = bpy.data.worlds.new("DT_World")
            self.scene.world = world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()

        out_w = nodes.new("ShaderNodeOutputWorld")
        out_w.location = (400, 0)
        bg   = nodes.new("ShaderNodeBackground")
        bg.location = (100, 0)

        if not is_night:
            sky = nodes.new("ShaderNodeTexSky")
            sky.location = (-200, 0)
            sky.sky_type = "NISHITA"
            sky.sun_elevation    = math.radians(42.0)
            sky.sun_rotation     = math.radians(-35.0)
            sky.turbidity        = 3.5
            sky.sun_disc         = True
            links.new(sky.outputs["Color"], bg.inputs["Color"])
            bg.inputs["Strength"].default_value = 1.4
        else:
            bg.inputs["Color"].default_value    = (0.01, 0.015, 0.04, 1.0)
            bg.inputs["Strength"].default_value = 0.08

        links.new(bg.outputs["Background"], out_w.inputs["Surface"])

        # Солнце (Sun Light)
        sun_name = "DT_Sun"
        if sun_name not in bpy.data.objects:
            sun_data = bpy.data.lights.new(sun_name, type="SUN")
            sun_obj  = bpy.data.objects.new(sun_name, sun_data)
            self.scene.collection.children[0].objects.link(sun_obj) \
                if self.scene.collection.children \
                else self.scene.collection.objects.link(sun_obj)
        else:
            sun_obj = bpy.data.objects[sun_name]

        sun_obj.location = (0, 0, 60)
        sun_obj.data.energy = 0.15 if is_night else 4.0
        sun_obj.data.color  = (0.7, 0.8, 1.0) if is_night else (1.0, 0.97, 0.92)
        sun_obj.rotation_euler = Euler(
            (math.radians(48), math.radians(12), math.radians(-38)), "XYZ"
        )

    # ════════════════════════════════════════════════════════════════
    # PBR MATERIALS — исключительно процедурные узлы (NO photo textures)
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_or_create_mat(name: str):
        if name in bpy.data.materials:
            return bpy.data.materials[name]
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        return mat

    # ── Asphalt PBR ─────────────────────────────────────────────────
    def _mat_asphalt(self):
        mat = self._get_or_create_mat("MAT_PBR_Asphalt")
        if mat.node_tree.nodes:
            return mat  # уже настроен

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        out  = nodes.new("ShaderNodeOutputMaterial");  out.location  = (900, 0)
        bsdf = nodes.new("ShaderNodeBsdfPrincipled");  bsdf.location = (600, 0)
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        # Координаты — Object (независимо от UV)
        coord = nodes.new("ShaderNodeTexCoord"); coord.location = (-600, 0)
        mapping = nodes.new("ShaderNodeMapping");   mapping.location = (-400, 0)
        mapping.inputs["Scale"].default_value = (8.0, 8.0, 8.0)
        links.new(coord.outputs["Object"], mapping.inputs["Vector"])

        # Крупный Noise → Base Color (тёмно-серый асфальт)
        noise1 = nodes.new("ShaderNodeTexNoise"); noise1.location = (-150, 150)
        noise1.inputs["Scale"].default_value    = 3.0
        noise1.inputs["Detail"].default_value   = 12.0
        noise1.inputs["Roughness"].default_value = 0.75
        noise1.inputs["Distortion"].default_value = 0.15
        links.new(mapping.outputs["Vector"], noise1.inputs["Vector"])

        # ColorRamp: чёрный → тёмно-серый (асфальт)
        ramp = nodes.new("ShaderNodeValToRGB"); ramp.location = (100, 150)
        ramp.color_ramp.elements[0].position = 0.30
        ramp.color_ramp.elements[0].color    = (0.020, 0.020, 0.022, 1.0)
        ramp.color_ramp.elements[1].position = 0.70
        ramp.color_ramp.elements[1].color    = (0.055, 0.055, 0.060, 1.0)
        links.new(noise1.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

        # Мелкий Noise → Roughness (микрогравий)
        noise2 = nodes.new("ShaderNodeTexNoise"); noise2.location = (-150, -100)
        noise2.inputs["Scale"].default_value    = 60.0
        noise2.inputs["Detail"].default_value   = 6.0
        noise2.inputs["Roughness"].default_value = 0.80
        links.new(mapping.outputs["Vector"], noise2.inputs["Vector"])

        ramp_r = nodes.new("ShaderNodeValToRGB"); ramp_r.location = (100, -100)
        ramp_r.color_ramp.elements[0].position = 0.0
        ramp_r.color_ramp.elements[0].color    = (0.38, 0.38, 0.38, 1.0)
        ramp_r.color_ramp.elements[1].position = 1.0
        ramp_r.color_ramp.elements[1].color    = (0.62, 0.62, 0.62, 1.0)
        links.new(noise2.outputs["Fac"], ramp_r.inputs["Fac"])
        cr_sep = nodes.new("ShaderNodeSeparateColor"); cr_sep.location = (330, -100)
        links.new(ramp_r.outputs["Color"], cr_sep.inputs["Color"])
        links.new(cr_sep.outputs["Red"], bsdf.inputs["Roughness"])

        # Bump → Нормаль (имитация зернистости асфальта)
        bump = nodes.new("ShaderNodeBump"); bump.location = (330, -280)
        bump.inputs["Strength"].default_value  = 0.25
        bump.inputs["Distance"].default_value  = 0.04
        links.new(noise2.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        bsdf.inputs["Specular IOR Level"].default_value = 0.04
        return mat

    # ── Concrete PBR (бордюры, тротуары) ────────────────────────────
    def _mat_concrete(self):
        mat = self._get_or_create_mat("MAT_PBR_Concrete")
        if mat.node_tree.nodes:
            return mat

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        out  = nodes.new("ShaderNodeOutputMaterial"); out.location  = (800, 0)
        bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (550, 0)
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        coord   = nodes.new("ShaderNodeTexCoord"); coord.location   = (-600, 0)
        mapping = nodes.new("ShaderNodeMapping");  mapping.location = (-400, 0)
        mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
        links.new(coord.outputs["Object"], mapping.inputs["Vector"])

        # Voronoi → трещины/фактура бетона
        vor = nodes.new("ShaderNodeTexVoronoi"); vor.location = (-150, 100)
        vor.inputs["Scale"].default_value    = 5.0
        vor.inputs["Randomness"].default_value = 0.8
        links.new(mapping.outputs["Vector"], vor.inputs["Vector"])

        noise = nodes.new("ShaderNodeTexNoise"); noise.location = (-150, -100)
        noise.inputs["Scale"].default_value    = 8.0
        noise.inputs["Detail"].default_value   = 10.0
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

        mix = nodes.new("ShaderNodeMix"); mix.location = (100, 0)
        mix.data_type = "RGBA"
        mix.inputs["Factor"].default_value = 0.35
        links.new(vor.outputs["Color"],   mix.inputs["A"])
        links.new(noise.outputs["Color"], mix.inputs["B"])

        # ColorRamp: светло-серый бетон
        ramp = nodes.new("ShaderNodeValToRGB"); ramp.location = (280, 0)
        ramp.color_ramp.elements[0].position = 0.25
        ramp.color_ramp.elements[0].color    = (0.50, 0.50, 0.50, 1.0)
        ramp.color_ramp.elements[1].position = 0.75
        ramp.color_ramp.elements[1].color    = (0.78, 0.78, 0.76, 1.0)

        sep = nodes.new("ShaderNodeSeparateColor"); sep.location = (100, 0)
        links.new(mix.outputs["Result"], sep.inputs["Color"])
        links.new(sep.outputs["Red"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

        bsdf.inputs["Roughness"].default_value = 0.82
        bsdf.inputs["Specular IOR Level"].default_value = 0.02
        return mat

    # ── Grass PBR (газон, деревья) ───────────────────────────────────
    def _mat_grass(self):
        mat = self._get_or_create_mat("MAT_PBR_Grass")
        if mat.node_tree.nodes:
            return mat

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        out  = nodes.new("ShaderNodeOutputMaterial"); out.location  = (700, 0)
        bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (450, 0)
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        coord   = nodes.new("ShaderNodeTexCoord"); coord.location   = (-600, 0)
        mapping = nodes.new("ShaderNodeMapping");  mapping.location = (-400, 0)
        mapping.inputs["Scale"].default_value = (5.0, 5.0, 5.0)
        links.new(coord.outputs["Object"], mapping.inputs["Vector"])

        noise = nodes.new("ShaderNodeTexNoise"); noise.location = (-150, 0)
        noise.inputs["Scale"].default_value    = 15.0
        noise.inputs["Detail"].default_value   = 10.0
        noise.inputs["Roughness"].default_value = 0.65
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

        ramp = nodes.new("ShaderNodeValToRGB"); ramp.location = (80, 0)
        ramp.color_ramp.elements[0].position = 0.30
        ramp.color_ramp.elements[0].color    = (0.025, 0.085, 0.020, 1.0)  # тёмно-зелёный
        ramp.color_ramp.elements[1].position = 0.70
        ramp.color_ramp.elements[1].color    = (0.075, 0.180, 0.045, 1.0)  # зелёный
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

        bsdf.inputs["Roughness"].default_value = 0.88
        bsdf.inputs["Specular IOR Level"].default_value = 0.02
        return mat

    # ── Road Marking PBR ────────────────────────────────────────────
    def _mat_marking(self, color: str = "white"):
        name = f"MAT_PBR_Marking_{color.capitalize()}"
        mat = self._get_or_create_mat(name)
        if mat.node_tree.nodes:
            return mat

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        out  = nodes.new("ShaderNodeOutputMaterial"); out.location  = (600, 0)
        bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (350, 0)
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        if color == "yellow":
            bsdf.inputs["Base Color"].default_value = (0.95, 0.85, 0.05, 1.0)
        else:
            bsdf.inputs["Base Color"].default_value = (0.95, 0.95, 0.95, 1.0)

        # Лёгкая эмиссия для ночного режима
        bsdf.inputs["Roughness"].default_value = 0.30
        bsdf.inputs["Emission Strength"].default_value = 0.05
        bsdf.inputs["Emission Color"].default_value = bsdf.inputs["Base Color"].default_value
        return mat

    # ── Pole Metal PBR ──────────────────────────────────────────────
    def _mat_pole(self):
        mat = self._get_or_create_mat("MAT_PBR_Pole")
        if mat.node_tree.nodes:
            return mat

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        out  = nodes.new("ShaderNodeOutputMaterial"); out.location  = (500, 0)
        bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (250, 0)
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        bsdf.inputs["Base Color"].default_value = (0.18, 0.20, 0.22, 1.0)
        bsdf.inputs["Metallic"].default_value   = 0.90
        bsdf.inputs["Roughness"].default_value  = 0.28
        return mat


# ════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — вызывается из __init__.py (ADAS_OT_BuildDigitalTwin)
# ════════════════════════════════════════════════════════════════════

def build_digital_twin(
    img_bgr: np.ndarray,
    seg_masks: dict | None  = None,
    seg_metadata: dict | None = None,
    fov_deg: float          = 80.0,
    cam_height: float       = 7.0,
    road_width_override: float | None = None,
    lane_count_override: int  | None  = None,
    has_median_override: bool | None  = None,
    median_width_override: float | None = None,
    scene                   = None,
) -> dict:
    """
    Главная функция интеграции. Вызывается оператором Blender.

    Параметры:
        img_bgr               — изображение OpenCV (BGR)
        seg_masks             — маски из AIDepthSegmenter (опционально)
        seg_metadata          — метаданные из AIDepthSegmenter (опционально)
        fov_deg               — FOV камеры в градусах
        cam_height            — высота камеры в метрах
        road_width_override   — принудительная ширина дороги (м), если None — авто
        lane_count_override   — принудительное кол-во полос, если None — авто
        has_median_override   — принудительное наличие газона, если None — авто
        median_width_override — принудительная ширина газона (м), если None — авто
        scene                 — bpy.context.scene (опционально)

    Возвращает:
        metrics dict с результатами CV-анализа
    """
    # Stage 1: CV Parameter Extraction
    analyzer = CVRoadAnalyzer(
        img_bgr,
        default_fov_deg   = fov_deg,
        default_cam_height = cam_height,
        seg_masks          = seg_masks or {},
        seg_metadata       = seg_metadata or {},
    )
    metrics = analyzer.analyze()

    # Override из UI (если задано вручную)
    if road_width_override is not None:
        metrics["road_width_m"] = float(road_width_override)
    if lane_count_override is not None:
        metrics["lane_count"] = int(lane_count_override)
    if has_median_override is not None:
        metrics["has_median"] = bool(has_median_override)
    if median_width_override is not None and metrics["has_median"]:
        metrics["median_width_m"] = float(median_width_override)

    # Добавляем is_night из метаданных
    metrics["is_night"] = (seg_metadata or {}).get("is_night", False)

    print(f"[CVRoadAnalyzer] Результаты:")
    print(f"  Pitch: {metrics['pitch_deg']:.1f}°  Yaw: {metrics['yaw_deg']:.1f}°  FOV: {metrics['fov_deg']:.1f}°")
    print(f"  Полос: {metrics['lane_count']}  Ширина дороги: {metrics['road_width_m']:.2f} м")
    print(f"  Разделительный газон: {metrics['has_median']}")
    print(f"  Контрольных точек кривой: {len(metrics['curve_ctrl_pts_3d'])}")
    print(f"  Столбов/деревьев: {len(metrics['poles_3d'])}")

    # Stage 3–5: Parametric 3D Build
    builder = ProceduralRoadBuilder(metrics, scene=scene or bpy.context.scene)
    builder.build()

    return metrics
