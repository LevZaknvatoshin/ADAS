"""
ADAS CV Builder — Obstacle Detection Synthetic Data Generator for Blender
=========================================================================
Blender Add-on for procedural road environment generation from reference camera
frames, interactive operator-driven scenario testing, and automatic YOLO
Ground Truth export for road obstacle detection model training.

Workflow:
  Step 1: Operator loads reference frame → system builds digital twin
  Step 2: Operator configures test scenarios (obstacles, people, cars)
  Step 3: System renders dataset with synchronized YOLO annotations
"""

bl_info = {
    "name": "ADAS CV Builder — Road Obstacle Simulator",
    "author": "Lev Zaknvatoshin",
    "version": (2, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > ADAS Simulator",
    "description": "Procedural road scene builder + interactive obstacle scenario tester + YOLO dataset exporter",
    "category": "3D View",
}

import bpy
import os
import sys
import math
import shutil
import json
import subprocess
import numpy as np
from bpy.props import (
    FloatProperty, IntProperty, BoolProperty,
    StringProperty, EnumProperty, PointerProperty,
    CollectionProperty, FloatVectorProperty,
)
from bpy.types import PropertyGroup
from mathutils import Vector

# Module imports
from .camera_rig import CameraRig
from .ai_depth_segmenter import AIDepthSegmenter
from .depth_mesh_builder import DepthMeshBuilder  # legacy (camera-projection pipeline)
from .cv_road_twin_builder import build_digital_twin   # ← NEW: clean procedural pipeline
from .environment import (
    EnvironmentManager,
    EnvironmentModelLoader,
    get_available_environment_models,
    get_environments_dir,
)
from .obstacles import ObstacleManager, get_road_bounds, apply_scenario_preset
from .annotator import YOLOAnnotator
from .batch_render import BatchRenderer
from .road_infrastructure import RoadAnalyzer, LaneDetector, CollisionBuilder
from .physics_controller import RigidBodyController, SceneScaleEstimator
from .actor_behavior import VehicleController, PedestrianController, PedestrianState, SceneValidator


# ─────────────────────────────────────────────────────────────────────
# Scene Item (one obstacle / actor placed by operator)
# ─────────────────────────────────────────────────────────────────────
def _update_scene_item_prop(self, context):
    col = bpy.data.collections.get("ADAS_Objects")
    if not col:
        return
    props = getattr(context.scene, "adas_props", None)
    if not props:
        return
    idx = -1
    for i, it in enumerate(props.scene_items):
        if it == self:
            idx = i
            break
    if idx == -1:
        return
    for obj in col.objects:
        if obj.get("adas_item_index") == idx and obj.parent is None:
            from .obstacles import ObstacleManager
            mgr = ObstacleManager()
            z = mgr.get_road_z(self.pos_x, self.pos_y)
            obj.location = (self.pos_x, self.pos_y, z)
            base_rot_z = obj.get("adas_base_rot_z", 0.0)
            obj.rotation_euler.z = base_rot_z + math.radians(self.rotation_z)
            # Dynamic 53% lane scaling for vehicles
            if getattr(self, "obj_type", "") == "VEHICLE":
                try:
                    from .road_infrastructure import RoadAnalyzer, LaneDetector
                    det = LaneDetector(RoadAnalyzer())
                    lane_w = det.get_orthogonal_width_at(self.pos_x, self.pos_y)
                    if 1.0 <= lane_w <= 12.0:
                        target_w = 0.53 * lane_w
                        obj["adas_base_scale"] = float(target_w / 20.4902)
                except Exception:
                    pass

            base_scale = obj.get("adas_base_scale", 1.0)
            s = base_scale * self.scale_factor
            obj.scale = (s, s, s)
            mgr.align_to_road_surface(obj, self)
            break


def _update_scene_item_type(self, context):
    col = bpy.data.collections.get("ADAS_Objects")
    props = getattr(context.scene, "adas_props", None)
    if not props:
        return
    idx = -1
    for i, it in enumerate(props.scene_items):
        if it == self:
            idx = i
            break
    if idx == -1:
        return

    # Select smart default behavior for this object category
    if self.obj_type == "VEHICLE":
        self.scenario_behavior = "DRIVE_FORWARD"
    elif self.obj_type == "PERSON":
        self.scenario_behavior = "CROSS_ROAD_RL"
    elif self.obj_type == "OBSTACLE":
        self.scenario_behavior = "STATIC_IN_LANE"
    else:
        self.scenario_behavior = "STATIC"

    apply_scenario_preset(self, context.scene)

    from .obstacles import ObstacleManager, remove_item_objects
    remove_item_objects(idx)
    mgr = ObstacleManager()
    mgr.spawn_item(self, index=idx, total_frames=props.total_frames, fps=props.fps)


def _update_scenario_behavior(self, context):
    if getattr(self, "scenario_behavior", "CUSTOM") != "CUSTOM":
        apply_scenario_preset(self, context.scene)
    _update_scene_item_prop(self, context)


def get_behavior_items(self, context):
    t = getattr(self, "obj_type", "OBSTACLE")
    if t == "VEHICLE":
        return [
            ("DRIVE_FORWARD", "Правая полоса (попутная)", "Разместить на правой полосе лицом по ходу движения"),
            ("ONCOMING", "Левая полоса (встречная)", "Разместить на левой встречной полосе лицом к камере"),
            ("STOPPED_IN_LANE", "Центр правой полосы", "Разместить по центру полосы движения"),
            ("OVERTAKE", "Левая полоса (попутная)", "Разместить на левой полосе попутного направления"),
            ("PARKED_SHOULDER", "Правая обочина", "Разместить на правой обочине дороги"),
            ("CUSTOM", "Пользовательские координаты", "Ручная настройка положения и угла"),
        ]
    elif t == "PERSON":
        return [
            ("CROSS_ROAD_RL", "Правая сторона дороги", "Пешеход на правой части полотна"),
            ("CROSS_ROAD_LR", "Левая сторона дороги", "Пешеход на левой части полотна"),
            ("STAND_ROAD", "Центр полосы", "Пешеход на полосе движения"),
            ("STAND_SHOULDER", "Обочина", "Пешеход на краю дороги / обочине"),
            ("CUSTOM", "Пользовательские координаты", "Ручная настройка положения и угла"),
        ]
    elif t == "OBSTACLE":
        return [
            ("STATIC_IN_LANE", "На полосе движения", "Препятствие прямо на полосе движения"),
            ("STATIC_ON_SHOULDER", "На обочине дороги", "Препятствие на обочине (дорожные работы)"),
            ("SCATTERED_CONES", "Зона ограждения", "Препятствие со смещением"),
            ("CUSTOM", "Пользовательские координаты", "Ручная настройка положения и угла"),
        ]
    else:
        return [
            ("STATIC", "Статичный объект", "Неподвижный объект"),
            ("CUSTOM", "Пользовательский", "Ручная настройка параметров"),
        ]


class ADAS_SceneItem(PropertyGroup):
    """One interactive obstacle entry in the operator's scenario list"""
    obj_type: EnumProperty(
        name="Тип объекта",
        items=[
            ("PERSON", "Человек / Person", "Пешеход на дороге или обочине"),
            ("VEHICLE", "Автомобиль / Car", "Легковой автомобиль (машина.glb)"),
            ("OBSTACLE", "Препятствие / Obstacle", "Конус, шина, бочка или коробка"),
            ("CUSTOM", "Свой файл / Custom Model", "Пользовательский файл .glb/.gltf/.obj/.fbx"),
        ],
        default="OBSTACLE",
        update=_update_scene_item_type,
    )
    subtype: EnumProperty(
        name="Модель",
        items=[
            ("DEFAULT", "По умолчанию", "Автоматический выбор модели"),
            ("CONE", "Конус", "Дорожный сигнальный конус"),
            ("TIRE", "Шина", "Упавшая автомобильная покрышка"),
            ("BOX", "Коробка", "Картонная коробка"),
            ("BARREL", "Бочка", "Металлическая дорожная бочка"),
        ],
        default="DEFAULT",
        update=_update_scene_item_type,
    )
    scenario_behavior: EnumProperty(
        name="Сценарий поведения",
        items=get_behavior_items,
        update=_update_scenario_behavior,
    )
    custom_model_path: StringProperty(
        name="Файл модели",
        description="Path to .obj / .fbx / .blend / .gltf / .glb model",
        subtype="FILE_PATH",
        update=_update_scene_item_type,
    )
    pos_x: FloatProperty(name="X старт (м)", default=0.0, min=-30.0, max=30.0,
                         description="Lateral position on road (0 = center)",
                         update=_update_scene_item_prop)
    pos_y: FloatProperty(name="Y старт (м)", default=0.0, min=-50.0, max=100.0,
                         description="Distance along road from center",
                         update=_update_scene_item_prop)
    rotation_z: FloatProperty(name="Поворот (°)", default=0.0, min=-180.0, max=180.0,
                              description="Yaw rotation in degrees",
                              update=_update_scene_item_prop)
    scale_factor: FloatProperty(name="Масштаб", default=1.0, min=0.1, max=5.0,
                                update=_update_scene_item_prop)

    # Animation
    animated: BoolProperty(name="Анимация движения", default=False,
                           description="Animate this object along a trajectory",
                           update=_update_scene_item_prop)
    end_x: FloatProperty(name="X конец (м)", default=0.0, min=-30.0, max=30.0,
                         update=_update_scene_item_prop)
    end_y: FloatProperty(name="Y конец (м)", default=5.0, min=-50.0, max=100.0,
                         update=_update_scene_item_prop)
    speed_kmh: FloatProperty(name="Скорость (км/ч)", default=5.0, min=0.1, max=120.0,
                             update=_update_scene_item_prop)


def get_env_models_enum(self, context):
    models = get_available_environment_models()
    items = []
    for filename, filepath, size_mb in models:
        items.append((filename, f"{filename} ({size_mb:.1f} MB)", f"3D-модель окружения: {filepath}"))
    if not items:
        items.append(("NONE", "Нет моделей в папке Окружения", "Поместите файлы .glb, .obj, .fbx в папку Окружения"))
    return items


def focus_object_in_viewport(context, item_index):
    """Align 3D viewport and pivot center around the selected scenario object"""
    col = bpy.data.collections.get("ADAS_Objects")
    if not col:
        return
    target_obj = None
    for obj in col.objects:
        if obj.get("adas_item_index") == item_index and obj.parent is None:
            target_obj = obj
            break

    if not target_obj:
        return

    try:
        bpy.ops.object.select_all(action='DESELECT')
        target_obj.select_set(True)
        context.view_layer.objects.active = target_obj
    except Exception:
        pass

    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.region_3d.view_perspective = 'PERSP'
                    space.region_3d.view_location = target_obj.matrix_world.translation
                    dim = getattr(target_obj, "dimensions", None)
                    d = dim.length if dim and dim.length > 0.5 else 3.5
                    space.region_3d.view_distance = max(4.0, d * 1.8)

            for region in area.regions:
                if region.type == 'WINDOW':
                    try:
                        with context.temp_override(area=area, region=region):
                            bpy.ops.view3d.view_selected(use_all_regions=False)
                    except Exception:
                        pass
                    break


def _on_scene_items_index_changed(self, context):
    try:
        idx = getattr(self, "scene_items_index", -1)
        if idx >= 0:
            focus_object_in_viewport(context, idx)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# Main Properties
# ─────────────────────────────────────────────────────────────────────
class ADAS_Properties(PropertyGroup):
    # ── Step 1: Environment Mode Selection ──
    env_source: EnumProperty(
        name="Источник окружения",
        items=[
            ("PHOTO", "По фото (Нейросеть / 2.5D)", "Реконструировать 3D-дорогу по фотографии с камеры", "IMAGE_DATA", 0),
            ("MODEL", "Готовая 3D-модель (Окружения)", "Загрузить готовую 3D-модель дороги из папки Окружения", "MOD_BUILD", 1),
        ],
        default="PHOTO",
    )
    selected_env_model: EnumProperty(
        name="Модель дороги",
        description="Выберите готовую 3D-модель окружения из папки Окружения",
        items=get_env_models_enum,
    )
    env_model_scale: FloatProperty(
        name="Масштаб модели",
        description="Коэффициент масштабирования (0.0254 для SketchUp дюймов, 1.0 для метров)",
        default=0.0254, min=0.00001, max=1000.0,
    )
    env_model_rot_z: FloatProperty(
        name="Поворот Z (°)",
        description="Поворот модели вокруг оси Z для выравнивания вдоль направления движения (+Y)",
        default=90.0, min=-180.0, max=180.0,
    )
    env_model_offset_x: FloatProperty(
        name="Смещение X (м)",
        description="Боковое смещение дороги (влево / вправо)",
        default=0.0, min=-500.0, max=500.0,
    )
    env_model_offset_y: FloatProperty(
        name="Смещение Y (м)",
        description="Продольное смещение дороги (вперед / назад)",
        default=0.0, min=-500.0, max=500.0,
    )
    env_model_offset_z: FloatProperty(
        name="Смещение Z (м)",
        description="Вертикальное смещение дороги (по высоте)",
        default=0.0, min=-50.0, max=50.0,
    )
    env_add_collision: BoolProperty(
        name="Физический коллайдер дороги",
        description="Создать Static Mesh Collider для физики автомобилей, пешеходов и препятствий",
        default=True,
    )
    env_setup_camera: BoolProperty(
        name="Установить дорожную камеру",
        description="Автоматически расположить камеру в начале дороги с видом вперед",
        default=True,
    )

    # ── Step 1: Reference & Camera ──
    reference_image: StringProperty(
        name="Референсный кадр",
        description="Path to the reference camera frame (.jpg / .png)",
        subtype="FILE_PATH",
    )
    cam_mount_height: FloatProperty(
        name="Высота камеры (м)",
        description="Camera mount height above ground (pole height)",
        default=7.5, min=1.0, max=20.0,
    )
    cam_pitch: FloatProperty(
        name="Наклон вниз (°)",
        description="Camera depression angle from horizontal (positive = looking down)",
        default=25.0, min=5.0, max=85.0,
    )
    cam_fov: FloatProperty(
        name="Угол обзора (°)",
        description="Horizontal field of view in degrees",
        default=120.0, min=30.0, max=180.0,
    )
    cam_res_x: IntProperty(name="Ширина (px)", default=1920, min=320, max=3840)
    cam_res_y: IntProperty(name="Высота (px)", default=1080, min=240, max=2160)

    # ── Step 1: Road geometry ──
    road_total_width: FloatProperty(
        name="Ширина проезжей части (м)",
        description="Total paved width of one direction",
        default=7.0, min=3.0, max=20.0,
    )
    lanes_per_direction: IntProperty(
        name="Полос в направлении",
        default=2, min=1, max=4,
    )
    median_width: FloatProperty(
        name="Разделительная полоса (м)",
        description="Width of grass median between directions",
        default=3.0, min=0.0, max=15.0,
    )
    road_length: FloatProperty(
        name="Длина дороги (м)",
        default=200.0, min=50.0, max=500.0,
    )
    auto_match_reference: BoolProperty(
        name="Авто-калибровка по фото",
        description="Auto-detect resolution, FOV, lighting, asphalt & shoulder textures from reference",
        default=True,
    )
    reconstruction_mode: EnumProperty(
        name="Режим реконструкции",
        items=[
            ("ROADCAM_25D", "RoadCam 2.5D (Новый метод)", "Непрерывный 3D-рельеф всей сцены с оптическим сопоставлением"),
            ("SEGMENTED_ROAD", "Сегментированный Twin (Разметка + Отбойники)", "Отдельный меш полотна дороги, разметка полос и отбойники"),
        ],
        default="ROADCAM_25D",
    )
    depth_strength: FloatProperty(
        name="Сила рельефа (Depth Strength)",
        description="Коэффициент выраженности глубины (по умолчанию 1.0)",
        default=1.0, min=0.01, max=3.0,
    )
    grid_resolution: IntProperty(
        name="Плотность сетки (Grid)",
        description="Разрешение сетки по ширине кадра (240 по умолчанию)",
        default=240, min=32, max=600,
    )
    use_pbr_shading: BoolProperty(
        name="PBR-шейдер асфальта",
        description="Использовать рельеф и шероховатость асфальта вместо чистой эмиссии фото",
        default=False,
    )
    roadcam_pitch: FloatProperty(
        name="Наклон камеры (°)",
        description="Угол наклона камеры (82° = горизонтальная ориентация дороги от камеры)",
        default=82.0, min=10.0, max=90.0,
    )
    base_distance: FloatProperty(
        name="Дистанция сцены (м)",
        description="Базовая глубина 2.5D сцены (20м = четкая неразмазанная текстура как раньше)",
        default=20.0, min=5.0, max=100.0,
    )
    max_road_distance: FloatProperty(
        name="Дальность дороги (м)",
        description="Физическая длина реконструируемой дороги (30м по умолчанию для исключения размытия)",
        default=30.0, min=10.0, max=250.0,
    )
    near_road_distance: FloatProperty(
        name="Ближний край (м)",
        description="Расстояние от камеры до ближайшего видимого края дороги под капотом",
        default=6.0, min=1.0, max=30.0,
    )
    clip_sky_roof: BoolProperty(
        name="Убрать крышу неба",
        description="Удаляет нависающий свод / черную крышу неба над дорогой, создавая открытую сцену",
        default=True,
    )
    sky_mode: EnumProperty(
        name="Режим неба",
        items=[
            ("REMOVE", "Открытая сцена (Без крыши)", "Полностью убирает крышу над дорогой, оставляя чистое открытое 3D-пространство"),
            ("BACKDROP", "Вертикальный задник", "Проецирует небо вертикально на горизонте как задний план"),
            ("DOME", "Полный свод (Купол)", "Исходный 2.5D свод со сплошным куполом над сценой"),
        ],
        default="REMOVE",
        description="Способ отображения неба и верхнего свода",
    )
    sky_cutoff: FloatProperty(
        name="Линия горизонта (0 = Авто)",
        description="Положение линии горизонта (0.0 для автоматического определения по карте глубин)",
        default=0.0, min=0.0, max=0.9,
    )
    depth_scale: FloatProperty(
        name="Шкала 3D-рельефа",
        description="Выраженность уклона и 3D-рельефа полотна дороги по карте глубин",
        default=2.0, min=0.1, max=10.0,
    )
    has_divided_road: BoolProperty(
        name="Разделённая дорога",
        description="Road has a grass median divider",
        default=False,
    )
    has_junction: BoolProperty(
        name="Перекресток / съезды (Т-развязка)",
        description="Include side turn-offs and junction branches",
        default=True,
    )
    has_guardrail: BoolProperty(
        name="Отбойники (барьеры)",
        description="Generate roadside metal guardrails",
        default=True,
    )
    has_buildings: BoolProperty(
        name="Фоновые строения (здания)",
        description="Generate distant low-poly horizon structures/facilities",
        default=True,
    )

    # ── Step 1: Digital Twin mode ──
    use_procedural_twin: BoolProperty(
        name="Процедурный Digital Twin (CV)",
        description=(
            "РЕКОМЕНДУЕТСЯ: Строит чистую математическую геометрию дороги "
            "(сплайны, раздельные меши, PBR-материалы) без Depth Map / Displace / "
            "Camera Projection. Пригодно для физических симуляций и CARLA Simulator."
        ),
        default=True,
    )

    # ── Step 2: Weather ──
    weather: EnumProperty(
        name="Погода",
        items=[
            ("OVERCAST", "Пасмурно (как на фото)", "Overcast sky matching reference"),
            ("SUNNY", "Ясно / Солнце", "Clear sunny day"),
            ("SUNSET", "Закат", "Low sun with warm tones"),
            ("NIGHT", "Ночь + фонари", "Night with streetlights"),
            ("RAIN", "Дождь / мокрый асфальт", "Wet road with reflections"),
        ],
        default="OVERCAST",
    )

    # ── Step 2: Scenario items (operator list) ──
    scene_items: CollectionProperty(type=ADAS_SceneItem)
    scene_items_index: IntProperty(
        default=0,
        update=_on_scene_items_index_changed,
    )

    # ── Step 3: Export ──
    output_dir: StringProperty(
        name="Папка экспорта",
        description="Output directory for rendered frames + YOLO labels",
        default="//adas_dataset",
        subtype="DIR_PATH",
    )
    render_engine: EnumProperty(
        name="Движок рендера",
        items=[
            ("BLENDER_EEVEE", "EEVEE (быстрый)", "Fast GPU rendering"),
            ("CYCLES", "Cycles (реализм)", "Path-traced photorealism"),
        ],
        default="BLENDER_EEVEE",
    )
    render_resolution_scale: EnumProperty(
        name="Качество рендера",
        description="Resolution multiplier for sharper render and anti-aliasing",
        items=[
            ("100", "1x (Оригинал)", "Исходное разрешение фото"),
            ("200", "2x (HD Super-Sampling)", "Увеличение в 2 раза для четкости"),
            ("300", "3x (Высокая четкость)", "Увеличение в 3 раза"),
            ("400", "4x (Ultra HD)", "Увеличение в 4 раза"),
        ],
        default="200",
    )
    total_frames: IntProperty(
        name="Кадров в сцене",
        default=60, min=1, max=600,
    )
    fps: IntProperty(name="FPS", default=30, min=10, max=60)
    frame_step: IntProperty(
        name="Шаг кадров",
        description="Save every N-th frame (1 = all frames)",
        default=1, min=1, max=10,
    )
    batch_variations: IntProperty(
        name="Кол-во вариаций",
        description="Auto-randomize N additional scenarios based on operator setup",
        default=1, min=1, max=500,
    )
    randomize_positions: BoolProperty(
        name="Рандомизация позиций",
        description="Randomly shift obstacle positions between batch variations",
        default=True,
    )
    randomize_weather: BoolProperty(
        name="Рандомизация погоды",
        description="Randomly change weather between variations",
        default=False,
    )
    export_qa_images: BoolProperty(
        name="QA-визуализация",
        description="Save images with drawn bounding boxes for quality check",
        default=True,
    )

    # ── UI Enhancements & Experiment Management ──
    ui_tab: EnumProperty(
        name="Вкладка",
        items=[
            ("ENV", "1. Окружение", "Выбор фото и постройка 3D-двойника", "WORLD", 0),
            ("SCENARIO", "2. Объекты", "Расстановка объектов на дороге", "AUTO", 1),
            ("PHYSICS", "3. Анимация", "Ручная анимация, таймлайн и запекание", "ACTION", 2),
            ("EXPERIMENT", "4. Опыты & Экспорт", "Изолированные опыты и экспорт YOLO", "OUTPUT", 3),
        ],
        default="ENV",
    )
    scenario_template: EnumProperty(
        name="Шаблон сцены",
        items=[
            ("CONVOY", "Попутный поток", "2-3 автомобиля на дистанции с заданной скоростью"),
            ("CROSSING", "Пешеход-нарушитель", "Пешеход переходит дорогу перед автомобилем"),
            ("ACCIDENT", "Аварийная зона", "Конусы ограждения и препятствие на полосе"),
            ("SHOULDER", "Стоянка на обочине", "Остановившийся автомобиль на правой обочине"),
        ],
        default="CONVOY",
    )
    experiment_name: StringProperty(
        name="Имя опыта",
        description="Название теста для изолированного сохранения в папке experiments/",
        default="test_001",
    )
    use_experiments_archive: BoolProperty(
        name="Сохранять в архив опытов",
        description="Автоматически изолировать каждый тест в папку experiments/exp_NNN_*",
        default=True,
    )
    last_experiment_dir: StringProperty(
        name="Текущая папка опыта",
        default="",
    )


# ─────────────────────────────────────────────────────────────────────
# Operators
# ─────────────────────────────────────────────────────────────────────

def find_system_python() -> str:
    """Finds system Python executable where torch and transformers are installed."""
    candidates = [
        r"C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe",
        r"C:\Python312\python.exe",
        r"C:\Program Files\Python312\python.exe",
        shutil.which("python"),
        shutil.which("python3"),
        sys.executable,
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return sys.executable


def find_helper_script(name: str) -> str:
    """Locates helper scripts in addon folder or workspace."""
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(addon_dir)
    for d in [addon_dir, workspace_dir]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(addon_dir, name)


class ADAS_OT_LoadEnvironmentModel(bpy.types.Operator):
    """Загрузить выбранную 3D-модель дороги из папки Окружения в сцену"""
    bl_idname = "adas.load_env_model"
    bl_label = "Загрузить 3D-окружение дороги"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.adas_props
        model_name = props.selected_env_model

        if not model_name or model_name == "NONE":
            self.report({"ERROR"}, "Выберите корректную 3D-модель из папки Окружения!")
            return {"CANCELLED"}

        env_dir = get_environments_dir()
        model_path = os.path.join(env_dir, model_name)
        if not os.path.exists(model_path):
            self.report({"ERROR"}, f"Файл модели не найден: {model_path}")
            return {"CANCELLED"}

        try:
            res = EnvironmentModelLoader.load_model(
                filepath=model_path,
                scale=props.env_model_scale,
                rot_z_deg=props.env_model_rot_z,
                offset=(props.env_model_offset_x, props.env_model_offset_y, props.env_model_offset_z),
                add_collision=props.env_add_collision,
                setup_camera=props.env_setup_camera,
            )

            # Apply lighting/weather preset
            try:
                env_mgr = EnvironmentManager()
                env_mgr.apply_weather(props.weather)
            except Exception:
                pass

            # Set viewport shading to MATERIAL so textures are immediately visible
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    for space in area.spaces:
                        if space.type == "VIEW_3D":
                            space.shading.type = "MATERIAL"

            b = res["bounds"]
            self.report(
                {"INFO"},
                f"3D-дорога '{model_name}' успешно загружена: {res['mesh_objects']} мешей, длина ~{b['length']:.1f}м, ширина ~{b['width']:.1f}м"
            )
            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Ошибка при загрузке 3D-окружения: {e}")
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}


class ADAS_OT_OpenEnvironmentsFolder(bpy.types.Operator):
    """Открыть папку Окружения в проводнике Windows"""
    bl_idname = "adas.open_environments_folder"
    bl_label = "Открыть папку Окружения"

    def execute(self, context):
        env_dir = get_environments_dir()
        if not os.path.exists(env_dir):
            os.makedirs(env_dir, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(env_dir)
            else:
                subprocess.Popen(["xdg-open", env_dir])
            self.report({"INFO"}, f"Папка открыта: {env_dir}")
        except Exception as e:
            self.report({"ERROR"}, f"Не удалось открыть папку: {e}")
        return {"FINISHED"}


class ADAS_OT_SetEnvScalePreset(bpy.types.Operator):
    """Быстро задать масштаб модели (дюймы, метры, миллиметры)"""
    bl_idname = "adas.set_env_scale_preset"
    bl_label = "Масштаб"
    bl_options = {"REGISTER", "UNDO"}

    scale_val: FloatProperty(default=1.0)

    def execute(self, context):
        context.scene.adas_props.env_model_scale = self.scale_val
        return {"FINISHED"}


class ADAS_OT_BuildScene(bpy.types.Operator):
    """Step 1: Build photorealistic 3D road digital twin from photo with AI Segmentation & Depth Relief"""
    bl_idname = "adas.build_scene"
    bl_label = "Построить 3D-окружение"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.adas_props

        if not props.reference_image:
            self.report({"ERROR"}, "Укажите путь к референсному кадру дороги!")
            return {"CANCELLED"}

        ref_path = bpy.path.abspath(props.reference_image)
        if not os.path.exists(ref_path):
            self.report({"ERROR"}, f"Файл не найден: {ref_path}")
            return {"CANCELLED"}

        base_dir = os.path.dirname(ref_path)
        stem = os.path.splitext(os.path.basename(ref_path))[0]
        cache_dir = os.path.join(base_dir, f".cache_ai_{stem}")
        os.makedirs(cache_dir, exist_ok=True)

        depth_npy = os.path.join(cache_dir, "depth.npy")
        depth_png = os.path.join(cache_dir, "depth_preview.png")
        poly_json = os.path.join(cache_dir, "road_polygon.json")

        python_bin = find_system_python()
        depth_estimator_path = find_helper_script("depth_estimator.py")
        auto_segmenter_path = find_helper_script("auto_segmenter.py")

        # 1. Monocular Depth Estimation (Depth Anything V2 Small)
        if not os.path.exists(depth_npy):
            print(f"[ADAS] Running DepthEstimator via {python_bin}...")
            ret = subprocess.run([
                python_bin, depth_estimator_path,
                "--input", ref_path,
                "--output", depth_npy,
                "--preview", depth_png,
            ], capture_output=True, text=True)
            if ret.returncode != 0:
                print(f"[ADAS Error] {ret.stderr}")
                self.report({"ERROR"}, f"Ошибка оценки глубины: {ret.stderr[:160]}")
                return {"CANCELLED"}

        depth_data = np.load(depth_npy).astype(np.float32)

        # ── Mode A: RoadCam 2.5D (New MVP Method) ────────────────────────
        if props.reconstruction_mode == "ROADCAM_25D":
            poly_data_opt = None
            if os.path.exists(poly_json):
                try:
                    with open(poly_json, "r", encoding="utf-8") as f:
                        poly_data_opt = json.load(f)
                except Exception:
                    pass

            from .road_twin_reconstructor import build_roadcam_25d_scene
            res = build_roadcam_25d_scene(
                image_path=ref_path,
                depth_map=depth_data,
                depth_strength=props.depth_strength,
                grid_w=props.grid_resolution,
                base_distance=props.base_distance,
                cam_pitch=props.roadcam_pitch,
                scene=context.scene,
                use_pbr=props.use_pbr_shading,
                clip_sky_roof=props.clip_sky_roof,
                sky_mode=props.sky_mode,
                sky_cutoff=props.sky_cutoff,
                poly_data=poly_data_opt,
                max_road_distance=props.max_road_distance,
                near_road_distance=props.near_road_distance,
            )

            # Switch 3D Viewport to Reference Camera and configure orbit pivot at road center
            for screen in bpy.data.screens:
                for area in screen.areas:
                    if area.type == "VIEW_3D":
                        for space in area.spaces:
                            if space.type == "VIEW_3D":
                                space.region_3d.view_perspective = "CAMERA"
                                space.region_3d.view_location = (0.0, 0.0, 0.5)
                                space.region_3d.view_distance = 15.0

            # Trigger background road & lane segmentation if not already cached
            mask_png = os.path.join(cache_dir, "road_mask.png")
            if not os.path.exists(mask_png) or not os.path.exists(poly_json):
                try:
                    subprocess.Popen([
                        python_bin, auto_segmenter_path,
                        "--image", ref_path,
                        "--output_json", poly_json,
                        "--output_mask", mask_png,
                    ])
                except Exception as e:
                    print(f"[ADAS] AutoSegmenter trigger warning: {e}")

            # Clean legacy lane guide collections if present
            if "ADAS_Lane_Lines" in bpy.data.collections:
                coll = bpy.data.collections["ADAS_Lane_Lines"]
                for obj in list(coll.objects):
                    bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.collections.remove(coll)

            self.report(
                {"INFO"},
                f"✓ RoadCam 2.5D сцена создана! Разрешение: {res['width']}x{res['height']}"
            )
            return {"FINISHED"}

        # ── Mode B: Segmented Twin (Road + Markings + Guardrails) ────────
        else:
            if not os.path.exists(poly_json):
                print(f"[ADAS] Running AutoSegmenter via {python_bin}...")
                ret = subprocess.run([
                    python_bin, auto_segmenter_path,
                    "--image", ref_path,
                    "--output_json", poly_json,
                ], capture_output=True, text=True)
                if ret.returncode != 0:
                    print(f"[ADAS Error] {ret.stderr}")
                    self.report({"ERROR"}, f"Ошибка сегментации: {ret.stderr[:160]}")
                    return {"CANCELLED"}

            with open(poly_json, "r", encoding="utf-8") as f:
                poly_data = json.load(f)

            from .road_twin_reconstructor import build_road_digital_twin
            res = build_road_digital_twin(
                image_path=ref_path,
                polygon_data=poly_data,
                depth_map=depth_data,
                depth_scale=props.depth_scale,
                scene=context.scene,
            )

            for screen in bpy.data.screens:
                for area in screen.areas:
                    if area.type == "VIEW_3D":
                        for space in area.spaces:
                            if space.type == "VIEW_3D":
                                space.region_3d.view_perspective = "CAMERA"

            if "ADAS_Lane_Lines" in bpy.data.collections:
                coll = bpy.data.collections["ADAS_Lane_Lines"]
                for obj in list(coll.objects):
                    bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.collections.remove(coll)

            self.report(
                {"INFO"},
                f"✓ 3D Digital Twin построен! Разрешение: {res['width']}x{res['height']}, объектов дороги: {res['road_objects']}"
            )
            return {"FINISHED"}


class ADAS_OT_UpscaleReference(bpy.types.Operator):
    """Upscales low-resolution reference image using Lanczos-4 and unsharp masking for sharp textures"""
    bl_idname = "adas.upscale_reference"
    bl_label = "Увеличить четкость фото (4x Upscale)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.adas_props
        ref = bpy.path.abspath(props.reference_image)
        if not ref or not os.path.exists(ref):
            self.report({"WARNING"}, "Выберите существующий файл референсного кадра!")
            return {"CANCELLED"}

        try:
            from PIL import Image, ImageFilter
            img = Image.open(ref)
            w, h = img.size
            if w >= 1920:
                self.report({"INFO"}, f"Фото уже имеет высокое разрешение ({w}x{h})")
                return {"FINISHED"}

            new_w, new_h = w * 4, h * 4
            upscaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            sharp = upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=3))

            base, ext = os.path.splitext(ref)
            out_path = f"{base}_upscaled.jpg"
            sharp.convert("RGB").save(out_path, quality=98)

            props.reference_image = out_path
            self.report({"INFO"}, f"✓ Разрешение увеличено: {w}x{h} → {new_w}x{new_h}! Перестройте 3D-сцену")
            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Ошибка апскейлинга: {e}")
            return {"CANCELLED"}


class ADAS_OT_ReviewPolygon(bpy.types.Operator):
    """Open interactive OpenCV window to inspect or adjust AI road polygon"""
    bl_idname = "adas.review_polygon"
    bl_label = "Ручная правка контура (OpenCV)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = context.scene.adas_props
        if not props.reference_image:
            self.report({"ERROR"}, "Сначала укажите путь к референсному кадру!")
            return {"CANCELLED"}

        ref_path = bpy.path.abspath(props.reference_image)
        if not os.path.exists(ref_path):
            self.report({"ERROR"}, f"Файл не найден: {ref_path}")
            return {"CANCELLED"}

        base_dir = os.path.dirname(ref_path)
        stem = os.path.splitext(os.path.basename(ref_path))[0]
        cache_dir = os.path.join(base_dir, f".cache_ai_{stem}")
        poly_json = os.path.join(cache_dir, "road_polygon.json")

        python_bin = find_system_python()
        annotator_script = find_helper_script("annotate_road.py")

        self.report({"INFO"}, "Открыто окно ручной разметки... (нажмите S в окне OpenCV для сохранения)")
        subprocess.Popen([
            python_bin, annotator_script,
            "--image", ref_path,
            "--output", poly_json,
            "--initial", poly_json if os.path.exists(poly_json) else "",
        ])
        return {"FINISHED"}


class ADAS_OT_SelectRoute(bpy.types.Operator):
    """Select and edit the 3D vehicle driving route curve in the viewport"""
    bl_idname = "adas.select_route"
    bl_label = "Выбрать маршрут в 3D"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        route_obj = bpy.data.objects.get("ADAS_Vehicle_Route")
        if not route_obj:
            self.report({"WARNING"}, "Маршрут ADAS_Vehicle_Route еще не создан. Постройте 3D-сцену!")
            return {"CANCELLED"}
        bpy.ops.object.select_all(action='DESELECT')
        route_obj.select_set(True)
        context.view_layer.objects.active = route_obj
        self.report({"INFO"}, "Маршрут выбран! Нажмите Tab в 3D View для редактирования контрольных точек.")
        return {"FINISHED"}


class ADAS_OT_RebuildRoute(bpy.types.Operator):
    """Rebuild the 3D vehicle driving route curve from the reconstructed road mesh"""
    bl_idname = "adas.rebuild_route"
    bl_label = "Перестроить маршрут дороги"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from .road_twin_reconstructor import build_vehicle_route_curve
        route_obj = build_vehicle_route_curve(context.scene)
        if route_obj:
            self.report({"INFO"}, "Маршрут ADAS_Vehicle_Route успешно обновлен вдоль оси дороги!")
            return {"FINISHED"}
        else:
            self.report({"WARNING"}, "Не удалось найти полисетку дороги для построения маршрута.")
            return {"CANCELLED"}


class ADAS_OT_AddSceneItem(bpy.types.Operator):
    """Add a new obstacle / actor to the scenario"""
    bl_idname = "adas.add_scene_item"
    bl_label = "Добавить объект"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.adas_props
        item = props.scene_items.add()
        idx = len(props.scene_items) - 1
        item.name = f"Объект {len(props.scene_items)}"
        props.scene_items_index = idx

        # Default behavior preset according to type
        if item.obj_type == "VEHICLE":
            item.scenario_behavior = "DRIVE_FORWARD"
        elif item.obj_type == "PERSON":
            item.scenario_behavior = "CROSS_ROAD_RL"
        elif item.obj_type == "OBSTACLE":
            item.scenario_behavior = "STATIC_IN_LANE"
        else:
            item.scenario_behavior = "STATIC"

        # Calculate coordinates dynamically from active road twin
        apply_scenario_preset(item, context.scene)

        # Auto-spawn object on the road
        from .obstacles import ObstacleManager, remove_item_objects
        remove_item_objects(idx)
        mgr = ObstacleManager()
        mgr.spawn_item(item, index=idx, total_frames=props.total_frames, fps=props.fps)
        return {"FINISHED"}


class ADAS_OT_RemoveSceneItem(bpy.types.Operator):
    """Remove selected obstacle from scenario"""
    bl_idname = "adas.remove_scene_item"
    bl_label = "Удалить объект"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.adas_props
        idx = props.scene_items_index
        if 0 <= idx < len(props.scene_items):
            from .obstacles import ObstacleManager, remove_item_objects
            remove_item_objects(idx)
            props.scene_items.remove(idx)
            props.scene_items_index = max(0, idx - 1)
            mgr = ObstacleManager()
            mgr.clear()
            for i, it in enumerate(props.scene_items):
                mgr.spawn_item(it, index=i, total_frames=props.total_frames, fps=props.fps)
        return {"FINISHED"}


class ADAS_OT_SpawnScenario(bpy.types.Operator):
    """Step 2: Instantiate all scene items into the 3D scene with animation"""
    bl_idname = "adas.spawn_scenario"
    bl_label = "Создать сценарий в 3D"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.adas_props
        mgr = ObstacleManager()
        mgr.clear()

        scene = context.scene
        scene.frame_start = 1
        scene.frame_end = props.total_frames
        scene.render.fps = props.fps

        for i, item in enumerate(props.scene_items):
            if getattr(item, "scenario_behavior", "CUSTOM") != "CUSTOM":
                apply_scenario_preset(item, scene)
            mgr.spawn_item(item, index=i, total_frames=props.total_frames, fps=props.fps)

        scene.frame_set(1)
        self.report({"INFO"}, f"Сценарий создан: {len(props.scene_items)} объектов!")
        return {"FINISHED"}


class ADAS_OT_AnalyzeInfrastructure(bpy.types.Operator):
    """Build lightweight static collision surface for road physical interactions"""
    bl_idname = "adas.analyze_infrastructure"
    bl_label = "Построить коллизии дороги"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        analyzer = RoadAnalyzer(scene)
        col_builder = CollisionBuilder(scene)

        # Build road static collision mesh
        col_obj = col_builder.build_road_collision(analyzer.road_objects)

        # Update existing spawned scenario items scale based on perspective
        for i, item in enumerate(context.scene.adas_props.scene_items):
            obj = bpy.data.objects.get(f"Vehicle_{i}") or bpy.data.objects.get(f"Person_{i}") or bpy.data.objects.get(f"Obstacle_{i}")
            if obj:
                base_scale = obj.get("adas_base_scale", 1.0)
                s = base_scale * item.scale_factor
                obj.scale = (s, s, s)

        msg = "✓ Статический коллайдер дороги построен!"
        if col_obj:
            msg += f" [{col_obj.name}]"
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class ADAS_OT_BakePhysics(bpy.types.Operator):
    """Simulate and bake falling physics for traffic cones and dynamic obstacles onto road surface"""
    bl_idname = "adas.bake_physics"
    bl_label = "Запечь физику конусов и объектов"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        props = scene.adas_props
        analyzer = RoadAnalyzer(scene)
        col_builder = CollisionBuilder(scene)
        rb_ctrl = RigidBodyController(scene)

        # Ensure road has static collision
        col_builder.build_road_collision(analyzer.road_objects)

        # Find dynamic obstacles (cones, barrels, tires, boxes)
        dynamic_objs = []
        for i, item in enumerate(props.scene_items):
            if item.obj_type == "OBSTACLE":
                obj = bpy.data.objects.get(f"Obstacle_{i}")
                if obj:
                    road_z = analyzer.raycast_ground_z(item.pos_x, item.pos_y)
                    rb_ctrl.setup_falling_cone(
                        obj,
                        target_x=item.pos_x,
                        target_y=item.pos_y,
                        road_surface_z=road_z,
                        drop_height=1.0,
                    )
                    dynamic_objs.append(obj)

        if not dynamic_objs:
            self.report({"WARNING"}, "В сценарии нет объектов-препятствий (OBSTACLE) для симуляции физики.")
            return {"CANCELLED"}

        rb_ctrl.bake_simulation_to_keyframes(
            dynamic_objs,
            frame_start=scene.frame_start,
            frame_end=scene.frame_end,
        )

        self.report({"INFO"}, f"✓ Физика падения успешно запечена для {len(dynamic_objs)} объект(ов)!")
        return {"FINISHED"}


class ADAS_OT_ApplyRagdoll(bpy.types.Operator):
    """Применить физику и рэгдолл (падение под гравитацией, столкновения) и запечь в таймлайн без ручной анимации"""
    bl_idname = "adas.apply_ragdoll"
    bl_label = "Рэгдолл / Запечь физику"
    bl_description = "Автоматическая симуляция падения и столкновений с запеканием в таймлайн"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        props = scene.adas_props
        from .physics_controller import RigidBodyController
        rb_ctrl = RigidBodyController(scene)
        rb_ctrl.ensure_road_collision()

        col = bpy.data.collections.get("ADAS_Objects")
        if not col or not props.scene_items:
            self.report({"WARNING"}, "Нет объектов в сценарии. Сначала добавьте объекты.")
            return {"CANCELLED"}

        phys_objs = []
        for i, item in enumerate(props.scene_items):
            target_obj = None
            for obj in col.objects:
                if obj.get("adas_item_index") == i and obj.parent is None:
                    target_obj = obj
                    break
            if target_obj:
                rb_ctrl.setup_object_ragdoll(
                    target_obj,
                    obj_type=item.obj_type,
                    subtype=getattr(item, "subtype", "DEFAULT"),
                )
                phys_objs.append(target_obj)

        if not phys_objs:
            self.report({"WARNING"}, "Сначала нажмите 'Обновить сценарий на дороге', чтобы заспавнить 3D объекты.")
            return {"CANCELLED"}

        f_start = scene.frame_start
        f_end = max(30, props.total_frames or scene.frame_end)
        rb_ctrl.bake_simulation_to_keyframes(phys_objs, frame_start=f_start, frame_end=f_end)
        scene.frame_set(f_start)
        self.report({"INFO"}, f"✓ Рэгдолл/физика запечена для {len(phys_objs)} объект(ов) на {f_end} кадров!")
        return {"FINISHED"}


class ADAS_OT_ToggleLivePhysics(bpy.types.Operator):
    """Включить интерактивную физику (Rigid Body). Нажмите Spacebar в Blender для живой симуляции"""
    bl_idname = "adas.toggle_live_physics"
    bl_label = "Живая физика (Spacebar)"
    bl_description = "Включает живую физику падения. Запустите анимацию пробелом (Spacebar) для симуляции"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        props = scene.adas_props
        from .physics_controller import RigidBodyController
        rb_ctrl = RigidBodyController(scene)
        rb_ctrl.ensure_road_collision()

        col = bpy.data.collections.get("ADAS_Objects")
        if not col or not props.scene_items:
            self.report({"WARNING"}, "Нет объектов в сценарии.")
            return {"CANCELLED"}

        count = 0
        for i, item in enumerate(props.scene_items):
            for obj in col.objects:
                if obj.get("adas_item_index") == i and obj.parent is None:
                    rb_ctrl.setup_object_ragdoll(
                        obj,
                        obj_type=item.obj_type,
                        subtype=getattr(item, "subtype", "DEFAULT"),
                    )
                    count += 1
                    break

        scene.frame_set(1)
        self.report({"INFO"}, f"⚡ Физика активна для {count} объектов! Нажмите Пробел (Spacebar) для просмотра.")
        return {"FINISHED"}


class ADAS_OT_ClearRagdoll(bpy.types.Operator):
    """Очистить физику и вернуть объекты в статичное состояние"""
    bl_idname = "adas.clear_ragdoll"
    bl_label = "Сбросить симуляцию"
    bl_description = "Удаляет симуляцию и возвращает объекты на дорогу"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        from .physics_controller import RigidBodyController
        rb_ctrl = RigidBodyController(scene)
        col = bpy.data.collections.get("ADAS_Objects")
        if col:
            objs = [o for o in col.objects if o.parent is None]
            rb_ctrl.clear_ragdoll(objs)
            try:
                bpy.ops.adas.drop_all_to_road()
            except Exception:
                pass
        self.report({"INFO"}, "Физика и запекание очищены.")
        return {"FINISHED"}


class ADAS_OT_PreviewAnnotations(bpy.types.Operator):
    """Preview YOLO bounding boxes for the current frame"""
    bl_idname = "adas.preview_annotations"
    bl_label = "Просмотр разметки (текущий кадр)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        annotator = YOLOAnnotator()
        boxes = annotator.get_frame_annotations(context.scene)
        self.report({"INFO"}, f"Обнаружено {len(boxes)} объектов в кадре {context.scene.frame_current}")
        for b in boxes:
            name = b["class_name"]
            print(f"  [{name}] center=({b['cx']:.3f}, {b['cy']:.3f}) size=({b['w']:.3f}, {b['h']:.3f})")
        return {"FINISHED"}


class ADAS_OT_RenderDataset(bpy.types.Operator):
    """Step 3: Render dataset with YOLO annotations"""
    bl_idname = "adas.render_dataset"
    bl_label = "Рендер датасета"
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = context.scene.adas_props
        exp_dir = None
        if props.use_experiments_archive:
            try:
                from .project_paths import get_project_paths
                from .experiment_manager import ExperimentManager
                paths = get_project_paths()
                mgr = ExperimentManager(paths)
                exp_dir = mgr.create_experiment(props.experiment_name)
                props.last_experiment_dir = exp_dir
                mgr.save_scenario(context.scene, props)
                props.output_dir = mgr.dataset_dir
                print(f"[ADAS] Experiment isolated folder: {exp_dir}")
            except Exception as e:
                print(f"[ADAS Warning] Experiment manager fallback: {e}")

        try:
            renderer = BatchRenderer(props)
            renderer.run()
            msg = "✓ Датасет успешно сгенерирован!"
            if exp_dir:
                msg += f" в папку: {os.path.basename(exp_dir)}"
            self.report({"INFO"}, msg)
            return {"FINISHED"}
        except Exception as e:
            err_msg = f"Ошибка генерации датасета: {e}"
            print(f"[ADAS ERROR] {err_msg}")
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, err_msg)
            return {"CANCELLED"}


class ADAS_OT_KeyframeCurrent(bpy.types.Operator):
    """Поставить ключевой кадр (Loc/Rot/Scale) на текущем кадре"""
    bl_idname = "adas.keyframe_current"
    bl_label = "Поставить ключ (I)"
    bl_options = {"REGISTER", "UNDO"}

    target: bpy.props.EnumProperty(
        items=[
            ("ACTIVE", "Выбранный объект", "Записать ключ для активного объекта"),
            ("ALL", "Все объекты сценария", "Записать ключи для всех объектов ADAS"),
        ],
        default="ACTIVE",
    )

    def execute(self, context):
        scene = context.scene
        f = scene.frame_current

        objs = []
        if self.target == "ACTIVE":
            target = context.active_object
            if target:
                while target.parent:
                    target = target.parent
                objs = [target]
        else:
            col = bpy.data.collections.get("ADAS_Objects")
            if col:
                objs = [o for o in col.objects if o.parent is None]

        if not objs:
            self.report({"WARNING"}, "Выберите объект сценария на дороге.")
            return {"CANCELLED"}

        for obj in objs:
            obj.keyframe_insert(data_path="location", frame=f)
            obj.keyframe_insert(data_path="rotation_euler", frame=f)
            obj.keyframe_insert(data_path="scale", frame=f)

        self.report({"INFO"}, f"✓ Ключ кадра {f} сохранен для: {', '.join(o.name for o in objs)}")
        return {"FINISHED"}


class ADAS_OT_KeyframeDelete(bpy.types.Operator):
    """Удалить ключевой кадр на текущем кадре для активного объекта"""
    bl_idname = "adas.keyframe_delete"
    bl_label = "Удалить ключ"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        f = scene.frame_current
        obj = context.active_object
        if obj:
            while obj.parent:
                obj = obj.parent

        if not obj or not obj.animation_data or not obj.animation_data.action:
            self.report({"WARNING"}, "У выбранного объекта нет ключей анимации.")
            return {"CANCELLED"}

        try:
            obj.keyframe_delete(data_path="location", frame=f)
            obj.keyframe_delete(data_path="rotation_euler", frame=f)
            obj.keyframe_delete(data_path="scale", frame=f)
            self.report({"INFO"}, f"✓ Ключ на кадре {f} удален у {obj.name}.")
        except Exception as e:
            self.report({"WARNING"}, f"Ключ не найден на кадре {f}: {e}")
        return {"FINISHED"}


class ADAS_OT_ClearAnimation(bpy.types.Operator):
    """Очистить всю анимацию с выбранного объекта или со всех объектов ADAS"""
    bl_idname = "adas.clear_animation"
    bl_label = "Очистить анимацию"
    bl_options = {"REGISTER", "UNDO"}

    all_objects: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        if self.all_objects:
            col = bpy.data.collections.get("ADAS_Objects")
            count = 0
            if col:
                for o in col.objects:
                    if o.animation_data:
                        o.animation_data_clear()
                        count += 1
            self.report({"INFO"}, f"Анимация очищена для {count} объектов ADAS.")
        else:
            obj = context.active_object
            if obj:
                while obj.parent:
                    obj = obj.parent
            if obj and obj.animation_data:
                obj.animation_data_clear()
                self.report({"INFO"}, f"Анимация объекта {obj.name} очищена.")
            elif obj:
                self.report({"INFO"}, f"У объекта {obj.name} нет анимации.")
            else:
                self.report({"WARNING"}, "Выберите объект.")
        return {"FINISHED"}


class ADAS_OT_BakeAction(bpy.types.Operator):
    """Запечь ручную анимацию всех объектов сценария в чистые покадровые ключи (Bake Action)"""
    bl_idname = "adas.bake_action"
    bl_label = "Запечь анимацию в ключи"
    bl_description = "Запекает покадровые перемещения объектов сценария в постоянные ключи для безошибочного рендера"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        props = scene.adas_props
        col = bpy.data.collections.get("ADAS_Objects")
        if not col:
            self.report({"WARNING"}, "Коллекция ADAS_Objects не найдена. Добавьте объекты на вкладке 2.")
            return {"CANCELLED"}

        objs = [o for o in col.objects if o.parent is None and (o.type == 'MESH' or len(o.children) > 0)]
        if not objs:
            self.report({"WARNING"}, "В коллекции нет объектов для запекания.")
            return {"CANCELLED"}

        bpy.ops.object.select_all(action='DESELECT')
        for o in objs:
            o.select_set(True)
        context.view_layer.objects.active = objs[0]

        start_f = scene.frame_start
        end_f = min(scene.frame_end, props.total_frames) if props.total_frames > 0 else scene.frame_end

        try:
            bpy.ops.nla.bake(
                frame_start=start_f,
                frame_end=end_f,
                step=1,
                only_selected=True,
                visual_keying=True,
                clear_constraints=False,
                clear_parents=False,
                use_current_action=False,
                clean_curves=True,
                bake_types={'OBJECT'}
            )
            scene.frame_set(start_f)
            self.report({"INFO"}, f"✓ Успешно запечено {len(objs)} объектов (кадры {start_f}–{end_f})!")
            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Ошибка запекания: {e}")
            return {"CANCELLED"}


class ADAS_OT_ToggleAutoKeying(bpy.types.Operator):
    """Включить/выключить авто-запись ключевых кадров (Auto-Keying) при перемещении во вьюпорте"""
    bl_idname = "adas.toggle_auto_keying"
    bl_label = "Авто-ключевание"
    bl_options = {"REGISTER"}

    def execute(self, context):
        ts = context.scene.tool_settings
        ts.use_keyframe_insert_auto = not ts.use_keyframe_insert_auto
        state = "ВКЛЮЧЕНО" if ts.use_keyframe_insert_auto else "ВЫКЛЮЧЕНО"
        self.report({"INFO"}, f"Авто-ключевание: {state}")
        return {"FINISHED"}


class ADAS_OT_JumpFrame(bpy.types.Operator):
    """Быстрый переход по кадрам таймлайна"""
    bl_idname = "adas.jump_frame"
    bl_label = "Переход по кадрам"
    bl_options = {"REGISTER"}

    target: bpy.props.EnumProperty(
        items=[
            ("START", "К началу", "Перейти на первый кадр"),
            ("END", "К концу", "Перейти на последний кадр"),
        ],
        default="START",
    )

    def execute(self, context):
        scene = context.scene
        if self.target == "START":
            scene.frame_set(scene.frame_start)
        else:
            props = scene.adas_props
            end_f = min(scene.frame_end, props.total_frames) if props.total_frames > 0 else scene.frame_end
            scene.frame_set(end_f)
        return {"FINISHED"}


class ADAS_OT_TogglePlayback(bpy.types.Operator):
    """Воспроизведение / Пауза анимации (Spacebar)"""
    bl_idname = "adas.toggle_playback"
    bl_label = "Плей / Пауза"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            bpy.ops.screen.animation_play()
        except Exception:
            pass
        return {"FINISHED"}


class ADAS_OT_FocusSceneItem(bpy.types.Operator):
    """Сфокусировать камеру вьюпорта на объекте (вращение вокруг него)"""
    bl_idname = "adas.focus_scene_item"
    bl_label = "Камера вокруг объекта"
    bl_description = "Центрирует 3D-вид на объекте. Теперь вращение вьюпорта (СКМ / колесо мыши) происходит вокруг выбранного объекта"
    bl_options = {"REGISTER"}

    index: bpy.props.IntProperty(default=-1)

    def execute(self, context):
        props = context.scene.adas_props
        idx = self.index if self.index >= 0 else props.scene_items_index
        if not props.scene_items or idx < 0 or idx >= len(props.scene_items):
            self.report({"WARNING"}, "Выберите объект в списке.")
            return {"CANCELLED"}

        focus_object_in_viewport(context, idx)
        item = props.scene_items[idx]
        self.report({"INFO"}, f"🔍 Вьюпорт сфокусирован на: {item.name}")
        return {"FINISHED"}


class ADAS_OT_SwitchView(bpy.types.Operator):
    """Быстрое переключение ракурса во вьюпорте"""
    bl_idname = "adas.switch_view"
    bl_label = "Переключить ракурс"
    bl_options = {"REGISTER"}

    mode: bpy.props.EnumProperty(
        items=[
            ("REFERENCE", "Фото-камера", "Камера референсного снимка (Numpad 0)"),
            ("DRIVER", "Вид водителя", "Камера на уровне дороги / водителя"),
            ("TOP", "Вид сверху", "Ортогональный план сверху для точной расстановки"),
            ("CENTER", "Центр дороги", "Орбитальный фокус прямо на середину дороги"),
        ],
        default="REFERENCE",
    )

    def execute(self, context):
        scene = context.scene
        mode = self.mode

        if mode == "REFERENCE":
            cam = scene.objects.get("RoadCam_ReferenceCamera") or scene.objects.get("ADAS_SourceCamera") or scene.objects.get("ReferenceCamera")
            if cam:
                scene.camera = cam
                for area in context.screen.areas:
                    if area.type == "VIEW_3D":
                        for space in area.spaces:
                            if space.type == "VIEW_3D":
                                space.region_3d.view_perspective = "CAMERA"
                self.report({"INFO"}, f"Вид из камеры фиксации: {cam.name}")
            else:
                self.report({"WARNING"}, "Камера референса не найдена. Постройте 3D-сцену.")

        elif mode == "DRIVER":
            cam = scene.objects.get("DriverViewCamera") or scene.objects.get("RoadCam_DriverViewCamera")
            if not cam:
                cam_data = bpy.data.cameras.new("DriverViewCamera")
                cam = bpy.data.objects.new("DriverViewCamera", cam_data)
                scene.collection.objects.link(cam)
                bounds = get_road_bounds(scene)
                cam.location = (bounds.get("center_x", 0.0), bounds.get("y_start", -5.0) - 1.5, 1.4)
                cam.rotation_euler = (math.radians(85.0), 0, 0)
                cam_data.lens = 35.0

            scene.camera = cam
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    for space in area.spaces:
                        if space.type == "VIEW_3D":
                            space.region_3d.view_perspective = "CAMERA"
            self.report({"INFO"}, "Вид с уровня водителя активирован")

        elif mode == "TOP":
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    for space in area.spaces:
                        if space.type == "VIEW_3D":
                            r3d = space.region_3d
                            r3d.view_perspective = "ORTHO"
                            r3d.view_rotation = (1.0, 0.0, 0.0, 0.0)
                            r3d.view_location = (0.0, 0.0, 0.0)
                            r3d.view_distance = 35.0
            self.report({"INFO"}, "Ортогональный вид сверху (Top View)")

        elif mode == "CENTER":
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    for space in area.spaces:
                        if space.type == "VIEW_3D":
                            r3d = space.region_3d
                            r3d.view_perspective = "PERSP"
                            r3d.view_location = (0.0, 0.0, 0.5)
                            r3d.view_distance = 15.0
            self.report({"INFO"}, "Фокус вьюпорта отцентрирован на середине дороги")

        return {"FINISHED"}


class ADAS_OT_SnapToCursor(bpy.types.Operator):
    """Поставить активный объект в точку 3D-курсора на дороге"""
    bl_idname = "adas.snap_to_cursor"
    bl_label = "Поставить по 3D-курсору"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.adas_props
        if not props.scene_items or props.scene_items_index >= len(props.scene_items):
            self.report({"WARNING"}, "Выберите объект в списке сценария.")
            return {"CANCELLED"}

        item = props.scene_items[props.scene_items_index]
        cur = context.scene.cursor.location
        item.pos_x = cur.x
        item.pos_y = cur.y
        _update_scene_item_prop(item, context)
        self.report({"INFO"}, f"Объект перемещен: X={cur.x:.2f}m, Y={cur.y:.2f}m")
        return {"FINISHED"}


class ADAS_OT_SnapToLane(bpy.types.Operator):
    """Быстрая привязка объекта к выбранной полосе или обочине"""
    bl_idname = "adas.snap_to_lane"
    bl_label = "Привязать к полосе"
    bl_options = {"REGISTER", "UNDO"}

    lane: bpy.props.EnumProperty(
        items=[
            ("LEFT_SHOULDER", "Обочина левая", ""),
            ("LEFT_LANE", "Левая полоса", ""),
            ("CENTER", "По центру", ""),
            ("RIGHT_LANE", "Правая полоса", ""),
            ("RIGHT_SHOULDER", "Обочина правая", ""),
        ],
        default="RIGHT_LANE",
    )

    def execute(self, context):
        props = context.scene.adas_props
        if not props.scene_items or props.scene_items_index >= len(props.scene_items):
            self.report({"WARNING"}, "Выберите объект в списке.")
            return {"CANCELLED"}

        item = props.scene_items[props.scene_items_index]
        bounds = get_road_bounds(context.scene)

        lane_map = {
            "LEFT_SHOULDER": bounds["left_shoulder_x"],
            "LEFT_LANE": bounds["left_lane_x"],
            "CENTER": bounds["center_x"],
            "RIGHT_LANE": bounds["right_lane_x"],
            "RIGHT_SHOULDER": bounds["right_shoulder_x"],
        }
        item.pos_x = lane_map.get(self.lane, bounds["center_x"])
        _update_scene_item_prop(item, context)
        self.report({"INFO"}, f"Объект привязан к: {self.lane}")
        return {"FINISHED"}


class ADAS_OT_DropToRoad(bpy.types.Operator):
    """Сбросить выбранный объект точно на асфальт под действием гравитации и коллизии"""
    bl_idname = "adas.drop_to_road"
    bl_label = "Приземлить на дорогу (Drop)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.adas_props
        if not props.scene_items or props.scene_items_index >= len(props.scene_items):
            self.report({"WARNING"}, "Выберите объект в списке.")
            return {"CANCELLED"}

        item = props.scene_items[props.scene_items_index]
        col = bpy.data.collections.get("ADAS_Objects")
        if not col:
            self.report({"WARNING"}, "Коллекция ADAS_Objects не найдена.")
            return {"CANCELLED"}

        idx = props.scene_items_index
        target_obj = None
        for obj in col.objects:
            if obj.get("adas_item_index") == idx and obj.parent is None:
                target_obj = obj
                break

        if not target_obj:
            self.report({"WARNING"}, "Объект на сцене не найден. Нажмите 'Обновить сценарий'.")
            return {"CANCELLED"}

        from .obstacles import ObstacleManager
        mgr = ObstacleManager()
        gz = mgr.get_road_z(item.pos_x, item.pos_y)

        all_zs = []
        for c in [target_obj] + target_obj.children_recursive:
            if c.type == "MESH":
                for b in c.bound_box:
                    wco = c.matrix_world @ Vector(b)
                    all_zs.append(wco.z - target_obj.location.z)

        min_local_z = min(all_zs) if all_zs else 0.0
        final_z = gz - min_local_z
        target_obj.location.z = final_z
        mgr.align_to_road_surface(target_obj, item)
        self.report({"INFO"}, f"Объект приземлен на асфальт: Z = {final_z:.2f}м")
        return {"FINISHED"}


class ADAS_OT_DropAllToRoad(bpy.types.Operator):
    """Приземлить все объекты сценария на дорогу с проверкой коллизии"""
    bl_idname = "adas.drop_all_to_road"
    bl_label = "Приземлить все объекты на дорогу"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.adas_props
        from .obstacles import ObstacleManager
        mgr = ObstacleManager()
        col = bpy.data.collections.get("ADAS_Objects")
        if not col:
            return {"CANCELLED"}

        count = 0
        for i, item in enumerate(props.scene_items):
            for obj in col.objects:
                if obj.get("adas_item_index") == i and obj.parent is None:
                    gz = mgr.get_road_z(item.pos_x, item.pos_y)
                    all_zs = []
                    for c in [obj] + obj.children_recursive:
                        if c.type == "MESH":
                            for b in c.bound_box:
                                wco = c.matrix_world @ Vector(b)
                                all_zs.append(wco.z - obj.location.z)
                    min_local_z = min(all_zs) if all_zs else 0.0
                    obj.location.z = gz - min_local_z
                    mgr.align_to_road_surface(obj, item)
                    count += 1
                    break
        self.report({"INFO"}, f"Приземлено объектов на асфальт: {count}")
        return {"FINISHED"}


class ADAS_OT_ApplyTemplateScenario(bpy.types.Operator):
    """Применить готовый шаблон сценария в один клик"""
    bl_idname = "adas.apply_template_scenario"
    bl_label = "Применить шаблон сценария"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.adas_props
        from .obstacles import remove_item_objects, ObstacleManager
        for i in range(len(props.scene_items)):
            remove_item_objects(i)
        props.scene_items.clear()

        bounds = get_road_bounds(context.scene)
        tmpl = props.scenario_template

        if tmpl == "CONVOY":
            i1 = props.scene_items.add()
            i1.name = "Лидер потока"
            i1.obj_type = "VEHICLE"
            i1.subtype = "DEFAULT"
            i1.pos_x = bounds["right_lane_x"]
            i1.pos_y = bounds["y_mid"] + 10.0
            i1.animated = False

            i2 = props.scene_items.add()
            i2.name = "Ведомый автомобиль"
            i2.obj_type = "VEHICLE"
            i2.subtype = "DEFAULT"
            i2.pos_x = bounds["right_lane_x"]
            i2.pos_y = bounds["y_start"] + 5.0
            i2.animated = False

        elif tmpl == "CROSSING":
            i1 = props.scene_items.add()
            i1.name = "Автомобиль перед переходом"
            i1.obj_type = "VEHICLE"
            i1.subtype = "DEFAULT"
            i1.pos_x = bounds["right_lane_x"]
            i1.pos_y = bounds["y_start"] + 8.0
            i1.animated = False

            i2 = props.scene_items.add()
            i2.name = "Пешеход на обочине"
            i2.obj_type = "PERSON"
            i2.pos_x = bounds["right_shoulder_x"]
            i2.pos_y = bounds["y_mid"]
            i2.animated = False

        elif tmpl == "ACCIDENT":
            i1 = props.scene_items.add()
            i1.name = "Препятствие на полосе"
            i1.obj_type = "OBSTACLE"
            i1.subtype = "BOX"
            i1.pos_x = bounds["right_lane_x"]
            i1.pos_y = bounds["y_mid"]

            i2 = props.scene_items.add()
            i2.name = "Конус 1"
            i2.obj_type = "OBSTACLE"
            i2.subtype = "CONE"
            i2.pos_x = bounds["right_lane_x"] - 0.6
            i2.pos_y = bounds["y_mid"] - 2.5

            i3 = props.scene_items.add()
            i3.name = "Конус 2"
            i3.obj_type = "OBSTACLE"
            i3.subtype = "CONE"
            i3.pos_x = bounds["right_lane_x"] + 0.6
            i3.pos_y = bounds["y_mid"] - 2.5

        elif tmpl == "SHOULDER":
            i1 = props.scene_items.add()
            i1.name = "Стоянка на обочине"
            i1.obj_type = "VEHICLE"
            i1.subtype = "DEFAULT"
            i1.pos_x = bounds["right_shoulder_x"]
            i1.pos_y = bounds["y_mid"]
            i1.animated = False

            i2 = props.scene_items.add()
            i2.name = "Знак аварийной остановки"
            i2.obj_type = "OBSTACLE"
            i2.subtype = "CONE"
            i2.pos_x = bounds["right_shoulder_x"]
            i2.pos_y = bounds["y_mid"] - 3.0

        mgr = ObstacleManager()
        for i, item in enumerate(props.scene_items):
            mgr.spawn_item(item, index=i, total_frames=props.total_frames, fps=props.fps)

        props.scene_items_index = 0
        self.report({"INFO"}, f"✓ Шаблон '{tmpl}' применен! Создано {len(props.scene_items)} объекта(ов).")
        return {"FINISHED"}


class ADAS_OT_OpenExperimentFolder(bpy.types.Operator):
    """Открыть папку текущего опыта в проводнике Windows"""
    bl_idname = "adas.open_experiment_folder"
    bl_label = "Открыть папку опыта"
    bl_options = {"REGISTER"}

    def execute(self, context):
        from .project_paths import get_project_paths
        paths = get_project_paths()
        props = context.scene.adas_props
        exp_dir = getattr(props, "last_experiment_dir", "")
        if not exp_dir or not os.path.exists(exp_dir):
            exp_dir = paths.experiments_dir
        if os.path.exists(exp_dir):
            try:
                os.startfile(exp_dir)
            except Exception:
                import subprocess
                subprocess.Popen(["explorer", exp_dir])
            self.report({"INFO"}, f"Открыта папка: {exp_dir}")
        else:
            self.report({"WARNING"}, "Папка опытов не найдена.")
        return {"FINISHED"}


class ADAS_OT_SaveScenarioSnapshot(bpy.types.Operator):
    """Сохранить снимок сценария в текущий опыт (scenario.json)"""
    bl_idname = "adas.save_scenario_snapshot"
    bl_label = "Сохранить сценарий"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            from .project_paths import get_project_paths
            from .experiment_manager import ExperimentManager
            paths = get_project_paths()
            mgr = ExperimentManager(paths)
            props = context.scene.adas_props
            if not mgr.current_dir:
                exp_dir = mgr.create_experiment(props.experiment_name)
                props.last_experiment_dir = exp_dir
            saved_json = mgr.save_scenario(context.scene, props)
            self.report({"INFO"}, f"✓ Сценарий сохранен: {os.path.basename(saved_json)}")
            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Ошибка сохранения: {e}")
            return {"CANCELLED"}


# ─────────────────────────────────────────────────────────────────────
# UI List for scenario items
# ─────────────────────────────────────────────────────────────────────
class ADAS_UL_SceneItems(bpy.types.UIList):
    bl_idname = "ADAS_UL_scene_items"

    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        type_icons = {"PERSON": "OUTLINER_OB_ARMATURE", "VEHICLE": "AUTO",
                      "OBSTACLE": "MESH_CONE", "CUSTOM": "FILE_3D"}
        ic = type_icons.get(item.obj_type, "OBJECT_DATA")
        row = layout.row(align=True)
        row.prop(item, "obj_type", text="", icon=ic)
        row.label(text=f"{item.name} ({item.pos_x:+.1f}m, {item.pos_y:.0f}m)")
        op = row.operator("adas.focus_scene_item", text="", icon="VIEWZOOM", emboss=False)
        op.index = index


# ─────────────────────────────────────────────────────────────────────
# Panel (Modern Tabbed UI & Navigation Toolbar)
# ─────────────────────────────────────────────────────────────────────
class ADAS_PT_MainPanel(bpy.types.Panel):
    bl_label = "ADAS — Симулятор препятствий"
    bl_idname = "ADAS_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ADAS Simulator"

    def draw(self, context):
        layout = self.layout
        props = context.scene.adas_props

        # ── Quick Viewport Toolbar (Always Visible at Top) ──
        bar = layout.row(align=True)
        bar.scale_y = 1.25
        op = bar.operator("adas.switch_view", text="Фото (0)", icon="CAMERA_DATA")
        op.mode = "REFERENCE"
        op = bar.operator("adas.switch_view", text="Водитель", icon="AUTO")
        op.mode = "DRIVER"
        op = bar.operator("adas.switch_view", text="Сверху", icon="AXIS_TOP")
        op.mode = "TOP"
        op = bar.operator("adas.switch_view", text="Центр", icon="PIVOT_CURSOR")
        op.mode = "CENTER"

        layout.separator(factor=0.6)

        # ── Step-by-Step Tabs ──
        tab_row = layout.row(align=True)
        tab_row.scale_y = 1.15
        tab_row.prop(props, "ui_tab", expand=True)

        layout.separator(factor=0.5)

        # ═════════════════════════════════════════════════════════════
        # TAB 1: ENVIRONMENT & 3D ROAD TWIN
        # ═════════════════════════════════════════════════════════════
        if props.ui_tab == "ENV":
            # Mode toggle: По фото vs Готовая 3D-модель
            mode_box = layout.box()
            mode_box.label(text="Способ построения окружения:", icon="WORLD")
            row_mode = mode_box.row(align=True)
            row_mode.scale_y = 1.35
            row_mode.prop(props, "env_source", expand=True)

            if props.env_source == "MODEL":
                box = layout.box()
                box.label(text="Выбор готовой 3D-модели (Окружения):", icon="FILE_3D")
                r_pick = box.row(align=True)
                r_pick.prop(props, "selected_env_model", text="")
                r_pick.operator("adas.open_environments_folder", text="", icon="FOLDER_REDIRECT")

                # Presets & Scale
                s_box = box.box()
                s_box.label(text="Масштабирование и ориентация:", icon="SETTINGS")
                s_row = s_box.row(align=True)
                s_row.label(text="Быстрый масштаб:")
                op = s_row.operator("adas.set_env_scale_preset", text="SketchUp (дюймы)")
                op.scale_val = 0.0254
                op = s_row.operator("adas.set_env_scale_preset", text="1:1 (метры)")
                op.scale_val = 1.0
                op = s_row.operator("adas.set_env_scale_preset", text="CAD (мм)")
                op.scale_val = 0.001

                s_box.prop(props, "env_model_scale")
                s_box.prop(props, "env_model_rot_z")

                r_off = s_box.row(align=True)
                r_off.prop(props, "env_model_offset_x", text="X (влево/вправо)")
                r_off.prop(props, "env_model_offset_y", text="Y (вперед/назад)")
                r_off.prop(props, "env_model_offset_z", text="Z (высота)")

                c_box = box.box()
                c_box.label(text="Физика и камера:", icon="PHYSICS")
                c_box.prop(props, "env_add_collision")
                c_box.prop(props, "env_setup_camera")

                btn = box.row(align=True)
                btn.scale_y = 1.6
                btn.operator(
                    "adas.load_env_model",
                    text="🛣️ Загрузить 3D-окружение дороги",
                    icon="IMPORT",
                )

            else:
                box = layout.box()
                box.label(text="Выбор референсного кадра:", icon="IMAGE_DATA")
                col = box.column(align=True)
                col.prop(props, "reference_image")

                if props.reference_image:
                    ref_abs = bpy.path.abspath(props.reference_image)
                    if os.path.exists(ref_abs):
                        try:
                            from PIL import Image
                            with Image.open(ref_abs) as _im:
                                _w, _h = _im.size
                            info_box = col.box()
                            info_row = info_box.row()
                            info_row.label(text=f"Разрешение: {_w} × {_h} px", icon="CHECKMARK")
                            if _w < 1200:
                                btn_up = info_box.row(align=True)
                                btn_up.operator("adas.upscale_reference", text="Повысить четкость (4x AI/Lanczos)", icon="SHADERFX")
                        except Exception:
                            pass

                box.separator()
                box.label(text="Режим реконструкции:", icon="MOD_BUILD")
                box.prop(props, "reconstruction_mode", text="")

                if props.reconstruction_mode == "ROADCAM_25D":
                    sub = box.box()
                    sub.label(text="Параметры рельефа и камеры:", icon="SETTINGS")
                    sub.prop(props, "depth_strength")
                    sub.prop(props, "grid_resolution")
                    sub.prop(props, "roadcam_pitch")
                    r_dist = sub.row(align=True)
                    r_dist.prop(props, "near_road_distance")
                    r_dist.prop(props, "max_road_distance")
                    sub.prop(props, "use_pbr_shading")

                    sub.separator()
                    sub.label(text="Свод неба:", icon="LIGHT_SUN")
                    sub.prop(props, "clip_sky_roof")
                    if props.clip_sky_roof:
                        sub.prop(props, "sky_mode")
                        if props.sky_mode == "BACKDROP" or props.sky_cutoff > 0.0:
                            sub.prop(props, "sky_cutoff")

                    btn = box.row(align=True)
                    btn.scale_y = 1.6
                    btn.operator(
                        "adas.build_scene",
                        text="Построить 3D-сцену (RoadCam 2.5D)",
                        icon="SCENE_DATA",
                    )
                else:
                    sub = box.box()
                    sub.label(text="Параметры сегментированной дороги:", icon="SETTINGS")
                    sub.prop(props, "depth_scale")

                    btn = box.row(align=True)
                    btn.scale_y = 1.6
                    btn.operator(
                        "adas.build_scene",
                        text="Построить Twin (Дорога + Разметка)",
                        icon="SCENE_DATA",
                    )
                    btn_rev = box.row(align=True)
                    btn_rev.operator(
                        "adas.review_polygon",
                        text="Ручная правка контура (OpenCV)",
                        icon="GREASEPENCIL",
                    )

        # ═════════════════════════════════════════════════════════════
        # TAB 2: OBJECTS & SCENARIO SETUP
        # ═════════════════════════════════════════════════════════════
        elif props.ui_tab == "SCENARIO":
            # Ready placement templates
            tmpl_box = layout.box()
            tmpl_box.label(text="Начальная расстановка объектов (шаблоны):", icon="SNAP_ON")
            t_row = tmpl_box.row(align=True)
            t_row.prop(props, "scenario_template", text="")
            t_row.operator("adas.apply_template_scenario", text="Расставить", icon="FILE_REFRESH")

            # Objects list
            box = layout.box()
            box.label(text="Объекты на дороге:", icon="PHYSICS")

            row = box.row()
            row.template_list("ADAS_UL_scene_items", "", props, "scene_items",
                              props, "scene_items_index", rows=4)
            col_btns = row.column(align=True)
            col_btns.operator("adas.add_scene_item", icon="ADD", text="")
            col_btns.operator("adas.remove_scene_item", icon="REMOVE", text="")
            col_btns.separator()
            col_btns.operator("adas.focus_scene_item", icon="VIEWZOOM", text="")
            col_btns.operator("adas.drop_all_to_road", icon="SNAP_FACE", text="")

            # Selected item details
            if props.scene_items and 0 <= props.scene_items_index < len(props.scene_items):
                item = props.scene_items[props.scene_items_index]
                detail = box.box()
                detail.label(text=f"Свойства: {item.name}", icon="PREFERENCES")

                r_focus = detail.row(align=True)
                r_focus.scale_y = 1.25
                op_f = r_focus.operator("adas.focus_scene_item", text="🔍 Камера вокруг объекта (Фокус)", icon="VIEWZOOM")
                op_f.index = props.scene_items_index

                col = detail.column(align=True)
                col.prop(item, "obj_type")
                if item.obj_type == "CUSTOM":
                    col.prop(item, "custom_model_path")
                elif item.obj_type == "OBSTACLE":
                    col.prop(item, "subtype")

                # Snap buttons
                snap_box = col.box()
                snap_box.label(text="Быстрая привязка на дороге:", icon="SNAP_ON")
                snap_box.operator("adas.snap_to_cursor", icon="PIVOT_CURSOR")

                lane_row = snap_box.row(align=True)
                op_l = lane_row.operator("adas.snap_to_lane", text="Обочина Л.")
                op_l.lane = "LEFT_SHOULDER"
                op_l = lane_row.operator("adas.snap_to_lane", text="Левая")
                op_l.lane = "LEFT_LANE"
                op_l = lane_row.operator("adas.snap_to_lane", text="Правая")
                op_l.lane = "RIGHT_LANE"
                op_l = lane_row.operator("adas.snap_to_lane", text="Обочина П.")
                op_l.lane = "RIGHT_SHOULDER"

                snap_box.operator("adas.drop_to_road", text="⬇ Сбросить на дорогу (Коллизия)", icon="SNAP_FACE")

                col.separator()
                col.label(text="Точные координаты (метры):", icon="EMPTY_AXIS")
                r1 = col.row(align=True)
                r1.prop(item, "pos_x")
                r1.prop(item, "pos_y")
                r2 = col.row(align=True)
                r2.prop(item, "rotation_z")
                r2.prop(item, "scale_factor")
                if item.obj_type == "VEHICLE":
                    col.label(text="Авто-масштаб: 53% ширины полосы", icon="AUTO")

            btn = box.row()
            btn.scale_y = 1.3
            btn.operator("adas.spawn_scenario", text="Обновить объекты на дороге", icon="FILE_REFRESH")

        # ═════════════════════════════════════════════════════════════
        # TAB 3: MANUAL ANIMATION & BAKING STUDIO
        # ═════════════════════════════════════════════════════════════
        elif props.ui_tab == "PHYSICS":
            # 1. Timeline Controls
            box_time = layout.box()
            box_time.label(text="Управление таймлайном и кадрами:", icon="TIME")

            row_cur = box_time.row(align=True)
            row_cur.scale_y = 1.3
            row_cur.prop(context.scene, "frame_current", text="Текущий кадр")

            r_nav = box_time.row(align=True)
            r_nav.scale_y = 1.25
            op_start = r_nav.operator("adas.jump_frame", text="В начало", icon="REW")
            op_start.target = "START"
            r_nav.operator("adas.toggle_playback", text="Плей / Пауза (Space)", icon="PLAY")
            op_end = r_nav.operator("adas.jump_frame", text="В конец", icon="FF")
            op_end.target = "END"

            r_range = box_time.row(align=True)
            r_range.prop(props, "total_frames", text="Кадров всего")
            r_range.prop(props, "fps", text="FPS")

            # 2. Manual Animation & Keyframing
            box_key = layout.box()
            box_key.label(text="Ручная анимация (ключевые кадры):", icon="KEYFRAME_HLT")
            box_key.label(text="Перемещайте объект во вьюпорте (G, R, S) и ставьте ключи:", icon="INFO")

            r_keys = box_key.row(align=True)
            r_keys.scale_y = 1.25
            op_key = r_keys.operator("adas.keyframe_current", text="Ключ для выбранного (I)", icon="REC")
            op_key.target = "ACTIVE"
            r_keys.operator("adas.keyframe_delete", text="Удалить ключ", icon="X")

            r_all_key = box_key.row(align=True)
            op_all = r_all_key.operator("adas.keyframe_current", text="Записать ключ для ВСЕХ объектов", icon="DUPLICATE")
            op_all.target = "ALL"

            r_foc = box_key.row(align=True)
            r_foc.scale_y = 1.15
            op_f3 = r_foc.operator("adas.focus_scene_item", text="🔍 Сфокусировать камеру на объекте", icon="VIEWZOOM")
            op_f3.index = props.scene_items_index

            ts = context.scene.tool_settings
            auto_state = "ВКЛ (Ключи ставятся на лету)" if ts.use_keyframe_insert_auto else "ВЫКЛ"
            auto_icon = "REC" if ts.use_keyframe_insert_auto else "RADIOBUT_OFF"
            r_auto = box_key.row(align=True)
            r_auto.scale_y = 1.2
            r_auto.operator("adas.toggle_auto_keying", text=f"Авто-ключевание: {auto_state}", icon=auto_icon)

            r_clr = box_key.row(align=True)
            op_clr = r_clr.operator("adas.clear_animation", text="Очистить анимацию выбранного", icon="TRASH")
            op_clr.all_objects = False
            op_clr_all = r_clr.operator("adas.clear_animation", text="Сбросить у всех", icon="CANCEL")
            op_clr_all.all_objects = True

            # 3. Baking Studio
            box_bake = layout.box()
            box_bake.label(text="Запекание (Bake):", icon="ACTION")
            box_bake.label(text="Запекает движение объектов для точного рендера датасета:", icon="INFO")

            btn_bake = box_bake.row(align=True)
            btn_bake.scale_y = 1.45
            btn_bake.operator("adas.bake_action", text="🔥 Запечь анимацию объектов (Bake Action)", icon="ACTION")

            box_bake.separator()
            box_bake.label(text="Физика столкновений и падений:", icon="PHYSICS")
            col_p = box_bake.column(align=True)
            col_p.scale_y = 1.2
            col_p.operator("adas.apply_ragdoll", text="⚡ Запечь симуляцию физики/рэгдолла", icon="REC")
            col_p.operator("adas.toggle_live_physics", text="Живая физика (Space)", icon="PLAY")
            col_p.operator("adas.clear_ragdoll", text="Сбросить физику", icon="LOOP_BACK")

            # 4. Viewport Cameras
            box_cam = layout.box()
            box_cam.label(text="Ракурсы камеры (контроль движения):", icon="CAMERA_DATA")
            grid_cam = box_cam.grid_flow(columns=2, align=True)
            c1 = grid_cam.operator("adas.switch_view", text="Камера фиксации", icon="CAMERA_DATA")
            c1.mode = "REFERENCE"
            c2 = grid_cam.operator("adas.switch_view", text="Вид водителя", icon="AUTO")
            c2.mode = "DRIVER"
            c3 = grid_cam.operator("adas.switch_view", text="Вид сверху (Top)", icon="AXIS_TOP")
            c3.mode = "TOP"
            c4 = grid_cam.operator("adas.switch_view", text="Центр дороги", icon="VIEWZOOM")
            c4.mode = "CENTER"

        # ═════════════════════════════════════════════════════════════
        # TAB 4: EXPERIMENTS & YOLO DATASET
        # ═════════════════════════════════════════════════════════════
        elif props.ui_tab == "EXPERIMENT":
            # Experiment isolated folder management
            exp_box = layout.box()
            exp_box.label(text="Управление опытами (Архив тестов):", icon="FILE_FOLDER")
            col_e = exp_box.column(align=True)
            col_e.prop(props, "experiment_name")
            col_e.prop(props, "use_experiments_archive")

            r_exp = exp_box.row(align=True)
            r_exp.operator("adas.save_scenario_snapshot", icon="FILE_TICK", text="Сохранить scenario.json")
            r_exp.operator("adas.open_experiment_folder", icon="FOLDER_REDIRECT", text="Открыть папку опыта")

            # Dataset settings
            box = layout.box()
            box.label(text="Параметры рендеринга и аннотаций:", icon="OUTPUT")
            col = box.column(align=True)
            col.prop(props, "render_engine")
            col.prop(props, "render_resolution_scale")
            col.prop(props, "total_frames")
            col.prop(props, "fps")
            col.prop(props, "frame_step")

            box.separator()
            box.label(text="Пакетная рандомизация:", icon="DUPLICATE")
            col_v = box.column(align=True)
            col_v.prop(props, "batch_variations")
            col_v.prop(props, "randomize_positions")
            col_v.prop(props, "randomize_weather")
            col_v.prop(props, "export_qa_images")

            box.separator()
            row = box.row(align=True)
            row.operator("adas.preview_annotations", icon="VIEWZOOM")

            btn = box.row()
            btn.scale_y = 1.6
            btn.operator("adas.render_dataset", text="Рендер и экспорт датасета (YOLO)", icon="RENDER_ANIMATION")

        # ── Global Status Bar at Bottom ──
        layout.separator(factor=0.6)
        stat_box = layout.box()
        stat_row = stat_box.row(align=True)
        road_ready = any(
            o.name.startswith("RoadCam_") or o.name.startswith("ADAS_Road") or o.name.startswith("RoadScene_") or o.name.startswith("ADAS_Env_")
            for o in context.scene.objects
        )
        stat_icon = "CHECKMARK" if road_ready else "ERROR"
        stat_text = "Сцена: Готова" if road_ready else "Сцена: Не построена"
        stat_row.label(text=stat_text, icon=stat_icon)
        stat_row.label(text=f"Объектов: {len(props.scene_items)}", icon="OUTLINER_OB_MESH")


# ─────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────
classes = (
    ADAS_SceneItem,
    ADAS_Properties,
    ADAS_OT_LoadEnvironmentModel,
    ADAS_OT_OpenEnvironmentsFolder,
    ADAS_OT_SetEnvScalePreset,
    ADAS_OT_BuildScene,
    ADAS_OT_UpscaleReference,
    ADAS_OT_ReviewPolygon,
    ADAS_OT_SelectRoute,
    ADAS_OT_RebuildRoute,
    ADAS_OT_AddSceneItem,
    ADAS_OT_RemoveSceneItem,
    ADAS_OT_SpawnScenario,
    ADAS_OT_AnalyzeInfrastructure,
    ADAS_OT_BakePhysics,
    ADAS_OT_ApplyRagdoll,
    ADAS_OT_ToggleLivePhysics,
    ADAS_OT_ClearRagdoll,
    ADAS_OT_PreviewAnnotations,
    ADAS_OT_RenderDataset,
    ADAS_OT_KeyframeCurrent,
    ADAS_OT_KeyframeDelete,
    ADAS_OT_ClearAnimation,
    ADAS_OT_BakeAction,
    ADAS_OT_ToggleAutoKeying,
    ADAS_OT_JumpFrame,
    ADAS_OT_TogglePlayback,
    ADAS_OT_FocusSceneItem,
    ADAS_OT_SwitchView,
    ADAS_OT_SnapToCursor,
    ADAS_OT_SnapToLane,
    ADAS_OT_DropToRoad,
    ADAS_OT_DropAllToRoad,
    ADAS_OT_ApplyTemplateScenario,
    ADAS_OT_OpenExperimentFolder,
    ADAS_OT_SaveScenarioSnapshot,
    ADAS_UL_SceneItems,
    ADAS_PT_MainPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.adas_props = PointerProperty(type=ADAS_Properties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.adas_props


if __name__ == "__main__":
    register()
