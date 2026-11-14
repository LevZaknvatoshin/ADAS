"""
Environment Manager — Weather, sky, and lighting presets
========================================================
Provides atmospheric presets matching real-world conditions:
  OVERCAST — heavy cloud cover (matching reference frame)
  SUNNY    — clear day with sun
  SUNSET   — low sun, warm tones
  NIGHT    — dark sky, streetlights
  RAIN     — wet reflective asphalt, fog
"""

import bpy
import os
import math
from mathutils import Euler, Matrix, Vector


class EnvironmentManager:
    LIGHT_COL = "ADAS_Lights"

    def __init__(self):
        if self.LIGHT_COL not in bpy.data.collections:
            c = bpy.data.collections.new(self.LIGHT_COL)
            bpy.context.scene.collection.children.link(c)
        self.col = bpy.data.collections[self.LIGHT_COL]

    def _clear_lights(self):
        for obj in list(self.col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)

    def _setup_world(self, sky_color=None, sun_elev=45.0, sun_rot=70.0, strength=1.0):
        scene = bpy.context.scene
        world = scene.world
        if not world:
            world = bpy.data.worlds.new("ADAS_World")
            scene.world = world
        world.use_nodes = True
        nt = world.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputWorld")
        bg = nt.nodes.new("ShaderNodeBackground")

        if sky_color:
            bg.inputs["Color"].default_value = sky_color
            bg.inputs["Strength"].default_value = strength
        else:
            sky = nt.nodes.new("ShaderNodeTexSky")
            sky.sky_type = "NISHITA"
            sky.sun_elevation = math.radians(sun_elev)
            sky.sun_rotation = math.radians(sun_rot)
            sky.altitude = 50.0
            sky.air_density = 1.0
            sky.dust_density = 1.0
            sky.ozone_density = 1.0
            nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
            bg.inputs["Strength"].default_value = strength

        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    def _add_sun(self, name, elev, azimuth, energy, color=(1,1,1)):
        data = bpy.data.lights.new(name, "SUN")
        data.energy = energy
        data.color = color
        data.angle = math.radians(1.0)
        obj = bpy.data.objects.new(name, data)
        obj.rotation_euler = Euler((math.radians(90 - elev), 0, math.radians(azimuth)), "XYZ")
        obj.location = (0, 0, 20)
        self.col.objects.link(obj)

    def _set_asphalt_wetness(self, roughness):
        mat = bpy.data.materials.get("ADAS_Asphalt")
        if mat and mat.node_tree:
            for n in mat.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED":
                    n.inputs["Roughness"].default_value = roughness
                    break

    def apply_weather(self, preset, ref_analyzer=None):
        self._clear_lights()

        # If reference analyzer detected night, auto-bias towards night/ambient
        if ref_analyzer and ref_analyzer.is_night:
            preset = "NIGHT"

        if preset == "OVERCAST":
            sky_c = ref_analyzer.sky_color if ref_analyzer else (0.48, 0.52, 0.56, 1.0)
            self._setup_world(sky_color=sky_c, strength=1.6)
            self._add_sun("Sun", 60, 0, 1.2, color=(0.9, 0.92, 0.95))
            self._set_asphalt_wetness(0.78)

        elif preset == "SUNNY":
            self._setup_world(sun_elev=55, sun_rot=70, strength=1.0)
            self._add_sun("Sun", 55, 70, 3.5, color=(1.0, 0.98, 0.94))
            self._set_asphalt_wetness(0.75)

        elif preset == "SUNSET":
            self._setup_world(sun_elev=8, sun_rot=110, strength=1.2)
            self._add_sun("Sun", 8, 110, 4.0, color=(1.0, 0.65, 0.35))
            self._set_asphalt_wetness(0.7)

        elif preset == "NIGHT":
            sky_c = ref_analyzer.sky_color if ref_analyzer else (0.015, 0.018, 0.025, 1.0)
            self._setup_world(sky_color=sky_c, strength=0.08)
            self._add_sun("Moon", 45, -45, 0.15, color=(0.65, 0.75, 0.95))
            # Road lighting posts
            self._spawn_streetlights(y_start=15, y_end=170, step=35, x_offset=4.5, height=6.5)
            self._set_asphalt_wetness(0.18 if (ref_analyzer and ref_analyzer.is_wet) else 0.55)

        elif preset == "RAIN":
            sky_c = ref_analyzer.sky_color if ref_analyzer else (0.32, 0.35, 0.40, 1.0)
            self._setup_world(sky_color=sky_c, strength=1.0)
            self._add_sun("Sun", 50, 30, 1.0, color=(0.8, 0.85, 0.9))
            self._set_asphalt_wetness(0.12)
            self._add_fog()

    def _spawn_streetlights(self, y_start=10, y_end=180, step=40, x_offset=0.0, height=7.0):
        y = y_start
        i = 0
        while y <= y_end:
            for side in [1, -1]:
                data = bpy.data.lights.new(f"StreetLight_{i}_{side}", "SPOT")
                data.energy = 2000
                data.color = (1.0, 0.85, 0.65)
                data.spot_size = math.radians(70)
                data.spot_blend = 0.4
                obj = bpy.data.objects.new(f"StreetLight_{i}_{side}", data)
                obj.location = (side * 1.2 + x_offset, y, height)
                obj.rotation_euler = Euler((math.radians(70), 0, 0), "XYZ")
                self.col.objects.link(obj)
            y += step
            i += 1

    def _add_fog(self):
        """Add subtle atmospheric fog for rainy scenes"""
        world = bpy.context.scene.world
        if not world or not world.node_tree:
            return
        nt = world.node_tree
        # Add volume scatter to world
        out = None
        for n in nt.nodes:
            if n.type == "OUTPUT_WORLD":
                out = n
                break
        if out:
            scatter = nt.nodes.new("ShaderNodeVolumeScatter")
            scatter.inputs["Density"].default_value = 0.005
            scatter.inputs["Color"].default_value = (0.7, 0.75, 0.8, 1.0)
            nt.links.new(scatter.outputs["Volume"], out.inputs["Volume"])


# ─────────────────────────────────────────────────────────────────────
# 3D Road Environment Model Loader (from Окружения/ folder)
# ─────────────────────────────────────────────────────────────────────

def get_environments_dir() -> str:
    """Returns absolute path to the directory containing 3D environment models."""
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Окружения")),
        os.path.join(os.path.dirname(__file__), "environments"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def get_available_environment_models() -> list:
    """Lists all 3D environment model files found in the environments folder."""
    env_dir = get_environments_dir()
    if not os.path.exists(env_dir):
        return []
    valid_exts = {".glb", ".gltf", ".obj", ".fbx", ".blend"}
    models = []
    for f in sorted(os.listdir(env_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            fp = os.path.join(env_dir, f)
            models.append((f, fp, os.path.getsize(fp) / (1024 * 1024)))
    return models


class EnvironmentModelLoader:
    COLLECTION_NAME = "ADAS_Environment"

    @classmethod
    def clear_environment(cls):
        """Removes previously loaded environment objects and meshes."""
        col = bpy.data.collections.get(cls.COLLECTION_NAME)
        if col:
            for obj in list(col.objects):
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception:
                    pass

        # Also clean up any lingering objects with ADAS_Env_ prefix
        for obj in list(bpy.data.objects):
            if obj.name.startswith("ADAS_Env_"):
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception:
                    pass

    @classmethod
    def load_model(
        cls,
        filepath: str,
        scale: float = 0.0254,
        rot_z_deg: float = 90.0,
        offset: tuple = (0.0, 0.0, 0.0),
        add_collision: bool = True,
        setup_camera: bool = True,
    ) -> dict:
        """
        Imports a 3D road environment model into ADAS_Environment collection,
        scales and aligns it along the traffic direction (+Y), and sets up physics collision.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл окружения не найден: {filepath}")

        # Ensure collection exists and is linked
        col = bpy.data.collections.get(cls.COLLECTION_NAME)
        if not col:
            col = bpy.data.collections.new(cls.COLLECTION_NAME)
            bpy.context.scene.collection.children.link(col)

        cls.clear_environment()

        before_objs = set(bpy.data.objects)
        ext = os.path.splitext(filepath)[1].lower()

        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=filepath)
        elif ext == ".obj":
            try:
                bpy.ops.wm.obj_import(filepath=filepath)
            except AttributeError:
                bpy.ops.import_scene.obj(filepath=filepath)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=filepath)
        elif ext == ".blend":
            with bpy.data.libraries.load(filepath) as (data_from, data_to):
                data_to.objects = data_from.objects
            for obj in data_to.objects:
                if obj is not None:
                    col.objects.link(obj)
        else:
            raise ValueError(f"Неподдерживаемый формат модели: {ext}")

        imported_objs = [o for o in bpy.data.objects if o not in before_objs]
        if not imported_objs:
            raise RuntimeError("Не удалось импортировать объекты из модели.")

        # Organize into collection
        for obj in imported_objs:
            if obj.name not in col.objects:
                col.objects.link(obj)
            for c in list(obj.users_collection):
                if c != col:
                    c.objects.unlink(obj)

        # Find root object(s)
        root_objs = [o for o in imported_objs if o.parent is None]
        if not root_objs:
            root_objs = imported_objs

        scale_mat = Matrix.Scale(scale, 4)
        rot_mat = Matrix.Rotation(math.radians(rot_z_deg), 4, "Z")
        trans_mat = Matrix.Translation(Vector(offset))
        transform = trans_mat @ rot_mat @ scale_mat

        for r in root_objs:
            r.matrix_world = transform @ r.matrix_world

        bpy.context.view_layer.update()

        # Tag all imported objects (both empties and meshes) and configure colliders
        mesh_objs = []
        for o in imported_objs:
            o["adas_is_environment"] = True
            if not o.name.startswith("ADAS_Env_"):
                o.name = f"ADAS_Env_{o.name}"

            if o.type == "MESH":
                mesh_objs.append(o)
                o["role"] = "road_surface"

                if add_collision:
                    if not o.rigid_body:
                        bpy.context.view_layer.objects.active = o
                        bpy.ops.rigidbody.object_add()
                    if o.rigid_body:
                        o.rigid_body.type = "PASSIVE"
                        o.rigid_body.collision_shape = "MESH"
                        o.rigid_body.friction = 0.8
                        o.rigid_body.restitution = 0.05

        bpy.context.view_layer.update()

        # Setup road camera if requested
        cam_obj = None
        if setup_camera:
            cam_obj = cls._setup_road_camera()

        # Compute bounding dimensions
        min_c = [float("inf")] * 3
        max_c = [float("-inf")] * 3
        for obj in mesh_objs:
            for v in obj.bound_box:
                w_v = obj.matrix_world @ Vector(v)
                for i in range(3):
                    min_c[i] = min(min_c[i], w_v[i])
                    max_c[i] = max(max_c[i], w_v[i])

        bounds = {
            "min": min_c,
            "max": max_c,
            "width": max_c[0] - min_c[0] if mesh_objs else 0.0,
            "length": max_c[1] - min_c[1] if mesh_objs else 0.0,
            "height": max_c[2] - min_c[2] if mesh_objs else 0.0,
        }

        return {
            "total_objects": len(imported_objs),
            "mesh_objects": len(mesh_objs),
            "bounds": bounds,
            "camera": cam_obj,
        }

    @classmethod
    def _setup_road_camera(cls):
        scene = bpy.context.scene
        cam_name = "RoadCam_ReferenceCamera"
        cam_obj = scene.objects.get(cam_name)
        if not cam_obj:
            cam_data = bpy.data.cameras.new(cam_name)
            cam_obj = bpy.data.objects.new(cam_name, cam_data)
            scene.collection.objects.link(cam_obj)
        else:
            cam_data = cam_obj.data

        # Place camera at beginning of the road looking down +Y
        cam_obj.location = (0.0, -8.0, 3.5)
        cam_obj.rotation_euler = (math.radians(84.0), 0.0, 0.0)
        cam_data.lens = 35.0
        cam_data.clip_start = 0.1
        cam_data.clip_end = 1000.0
        scene.camera = cam_obj
        return cam_obj
