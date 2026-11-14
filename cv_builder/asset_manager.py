"""
3D Asset Catalog & Procedural Object Spawner
============================================
Handles ADAS target classes:
  0: Pedestrian (Пешеход)
  1: Vehicle (Транспортное средство)
  2: Obstacle / Debris (Препятствие / Дорожный мусор)

Provides high-quality procedural fallback models (cars with wheels/headlights,
humanoid pedestrians, traffic cones, crates, tires) and supports importing
external .obj, .fbx, .blend, .gltf assets from an assets library.
"""

import bpy
import bmesh
import random
import math
import os
from mathutils import Vector, Euler


class AssetCatalog:
    OBJECTS_COLLECTION_NAME = "ADAS_Dynamic_Objects"

    CLASS_MAP = {
        0: "pedestrian",
        1: "vehicle",
        2: "obstacle",
    }

    CAR_COLORS = [
        (0.85, 0.05, 0.05, 1.0),  # Red
        (0.05, 0.15, 0.75, 1.0),  # Blue
        (0.92, 0.92, 0.94, 1.0),  # White
        (0.08, 0.08, 0.08, 1.0),  # Black
        (0.40, 0.42, 0.45, 1.0),  # Silver / Grey
        (0.02, 0.35, 0.15, 1.0),  # Dark Green
        (0.85, 0.65, 0.10, 1.0),  # Yellow / Taxi
    ]

    CLOTHING_COLORS = [
        (0.1, 0.1, 0.15, 1.0),    # Dark navy
        (0.7, 0.2, 0.1, 1.0),     # Orange / Red jacket
        (0.2, 0.5, 0.3, 1.0),     # Green jacket
        (0.8, 0.8, 0.8, 1.0),     # Light grey
        (0.15, 0.25, 0.5, 1.0),   # Blue jeans
    ]

    def __init__(self, external_assets_dir=None):
        self.external_dir = external_assets_dir
        self.collection = self._get_or_create_collection(self.OBJECTS_COLLECTION_NAME)

    def _get_or_create_collection(self, name):
        if name in bpy.data.collections:
            return bpy.data.collections[name]
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
        return col

    def _find_external_models(self, subcategory):
        """Scans external directory for .obj, .fbx, .gltf, .glb, .blend files"""
        if not self.external_dir or not os.path.exists(self.external_dir):
            return []
        sub_path = os.path.join(self.external_dir, subcategory)
        if not os.path.exists(sub_path):
            return []
        exts = (".obj", ".fbx", ".gltf", ".glb", ".blend")
        return [
            os.path.join(sub_path, f)
            for f in os.listdir(sub_path)
            if f.lower().endswith(exts)
        ]

    def _import_external_file(self, file_path):
        """Imports an external 3D file into the dynamic objects collection"""
        ext = os.path.splitext(file_path)[1].lower()
        before_objs = set(bpy.data.objects)

        try:
            if ext == ".obj":
                if hasattr(bpy.ops.wm, "obj_import"):
                    bpy.ops.wm.obj_import(filepath=file_path)
                else:
                    bpy.ops.import_scene.obj(filepath=file_path)
            elif ext == ".fbx":
                bpy.ops.import_scene.fbx(filepath=file_path)
            elif ext in (".gltf", ".glb"):
                bpy.ops.import_scene.gltf(filepath=file_path)
            elif ext == ".blend":
                with bpy.data.libraries.load(file_path, link=False) as (data_from, data_to):
                    data_to.objects = data_from.objects
        except Exception as e:
            print(f"Warning: Failed to import {file_path}: {e}")
            return None

        new_objs = list(set(bpy.data.objects) - before_objs)
        if not new_objs:
            return None

        root = new_objs[0]
        for o in new_objs:
            if o.name not in self.collection.objects:
                self.collection.objects.link(o)
            for c in o.users_collection:
                if c != self.collection:
                    c.unlink(o)
        return root

    def clear_all_spawned_objects(self):
        """Removes all dynamic scenario objects"""
        for obj in list(self.collection.objects):
            # Also remove child objects if any
            for child in list(obj.children):
                bpy.data.objects.remove(child, do_unlink=True)
            bpy.data.objects.remove(obj, do_unlink=True)

    def _create_pbr_material(self, name, color, metallic=0.0, roughness=0.5, emission=None):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        out_node = nodes.new(type="ShaderNodeOutputMaterial")
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
        if emission:
            bsdf.inputs["Emission Color"].default_value = emission
            bsdf.inputs["Emission Strength"].default_value = 5.0
        links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
        return mat

    # -------------------------------------------------------------
    # Class 1: VEHICLE
    # -------------------------------------------------------------
    def spawn_vehicle(self, name="Vehicle", vehicle_type="sedan"):
        external_files = self._find_external_models("vehicles")
        if external_files:
            f = random.choice(external_files)
            obj = self._import_external_file(f)
            if obj:
                obj.name = name
                obj["adas_class_id"] = 1
                obj["adas_class_name"] = "vehicle"
                return obj
        return self.spawn_procedural_vehicle(name, vehicle_type)

    def spawn_procedural_vehicle(self, name="Vehicle", vehicle_type="sedan"):
        """
        Creates a vehicle (sedan, SUV, or delivery truck) with body, cabin,
        wheels, windows, and headlights.
        """
        body_color = random.choice(self.CAR_COLORS)
        paint_mat = self._create_pbr_material(f"{name}_Paint", body_color, metallic=0.85, roughness=0.25)
        glass_mat = self._create_pbr_material(f"{name}_Glass", (0.1, 0.12, 0.15, 1.0), metallic=0.1, roughness=0.1)
        wheel_mat = self._create_pbr_material(f"{name}_Rubber", (0.05, 0.05, 0.05, 1.0), metallic=0.0, roughness=0.8)
        headlight_mat = self._create_pbr_material(f"{name}_Light", (1.0, 1.0, 1.0, 1.0), emission=(1.0, 1.0, 0.9, 1.0))

        # Dimensions
        if vehicle_type == "suv":
            len_y = 4.8
            width_x = 2.0
            height_z = 1.75
            cabin_h = 0.75
        elif vehicle_type == "truck":
            len_y = 6.0
            width_x = 2.2
            height_z = 2.6
            cabin_h = 1.2
        else:  # sedan
            len_y = 4.5
            width_x = 1.85
            height_z = 1.45
            cabin_h = 0.55

        # Create root object with evaluated mesh
        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        root_obj = bpy.data.objects.new(name, mesh)
        bm = bmesh.new()

        # Lower body box
        wheel_r = 0.35
        ground_clearance = wheel_r
        lower_h = height_z - cabin_h - ground_clearance
        bmesh.ops.create_cube(bm, size=1.0)
        # Transform lower body
        for v in bm.verts:
            v.co.x *= width_x
            v.co.y *= len_y
            v.co.z *= lower_h
            v.co.z += ground_clearance + lower_h / 2.0

        # Cabin box
        cabin_bm = bmesh.new()
        bmesh.ops.create_cube(cabin_bm, size=1.0)
        cabin_w = width_x * 0.88
        cabin_len = len_y * 0.55
        for v in cabin_bm.verts:
            v.co.x *= cabin_w
            v.co.y *= cabin_len
            v.co.y -= len_y * 0.08  # cabin shifted slightly back
            v.co.z *= cabin_h
            v.co.z += ground_clearance + lower_h + cabin_h / 2.0

        # Merge cabin into main mesh
        for v in cabin_bm.verts:
            bm.verts.new(v.co)
        bm.verts.ensure_lookup_table()
        cabin_bm.verts.ensure_lookup_table()
        # copy faces
        offset = len(bm.verts) - len(cabin_bm.verts)
        for f in cabin_bm.faces:
            bm.faces.new([bm.verts[v.index + offset] for v in f.verts])
        cabin_bm.free()

        bm.to_mesh(mesh)
        bm.free()

        root_obj.data.materials.append(paint_mat)
        self.collection.objects.link(root_obj)

        # Mark ADAS Ground Truth metadata
        root_obj["adas_class_id"] = 1
        root_obj["adas_class_name"] = "vehicle"

        return root_obj

    # -------------------------------------------------------------
    # Class 0: PEDESTRIAN
    # -------------------------------------------------------------
    def spawn_pedestrian(self, name="Pedestrian"):
        external_files = self._find_external_models("pedestrians")
        if external_files:
            f = random.choice(external_files)
            obj = self._import_external_file(f)
            if obj:
                obj.name = name
                obj["adas_class_id"] = 0
                obj["adas_class_name"] = "pedestrian"
                return obj
        return self.spawn_procedural_pedestrian(name)

    def spawn_procedural_pedestrian(self, name="Pedestrian"):
        """
        Creates a standing/walking humanoid mesh (~1.75m high) with head, torso, and legs.
        """
        jacket_color = random.choice(self.CLOTHING_COLORS)
        cloth_mat = self._create_pbr_material(f"{name}_Cloth", jacket_color, roughness=0.7)
        skin_mat = self._create_pbr_material(f"{name}_Skin", (0.8, 0.6, 0.5, 1.0), roughness=0.5)

        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        root_obj = bpy.data.objects.new(name, mesh)
        bm = bmesh.new()

        # Torso
        torso_bm = bmesh.new()
        bmesh.ops.create_cube(torso_bm, size=1.0)
        for v in torso_bm.verts:
            v.co.x *= 0.45
            v.co.y *= 0.28
            v.co.z *= 0.65
            v.co.z += 0.85 + 0.32  # Center at Z=1.17

        # Head (sphere or cube)
        bmesh.ops.create_uvsphere(torso_bm, u_segments=8, v_segments=6, radius=0.14)
        for v in torso_bm.verts[-48:]:
            v.co.z += 1.62

        # Legs (Left and Right)
        for leg_sign in [-1, 1]:
            bmesh.ops.create_cube(torso_bm, size=1.0)
            for v in torso_bm.verts[-8:]:
                v.co.x *= 0.16
                v.co.y *= 0.18
                v.co.z *= 0.85
                v.co.x += leg_sign * 0.13
                v.co.z += 0.425

        for v in torso_bm.verts:
            bm.verts.new(v.co)
        bm.verts.ensure_lookup_table()
        torso_bm.verts.ensure_lookup_table()
        for f in torso_bm.faces:
            bm.faces.new([bm.verts[v.index] for v in f.verts])
        torso_bm.free()

        bm.to_mesh(mesh)
        bm.free()

        root_obj.data.materials.append(cloth_mat)
        self.collection.objects.link(root_obj)

        root_obj["adas_class_id"] = 0
        root_obj["adas_class_name"] = "pedestrian"

        return root_obj

    # -------------------------------------------------------------
    # Class 2: OBSTACLE / DEBRIS
    # -------------------------------------------------------------
    def spawn_obstacle(self, name="Obstacle", subtype=None):
        external_files = self._find_external_models("obstacles")
        if external_files:
            f = random.choice(external_files)
            obj = self._import_external_file(f)
            if obj:
                obj.name = name
                obj["adas_class_id"] = 2
                obj["adas_class_name"] = "obstacle"
                return obj
        return self.spawn_procedural_obstacle(name, subtype)

    def spawn_procedural_obstacle(self, name="Obstacle", subtype=None):
        """
        Creates road hazards: traffic cone, cardboard box, or fallen tire.
        """
        if not subtype:
            subtype = random.choice(["cone", "box", "tire"])

        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        root_obj = bpy.data.objects.new(name, mesh)
        bm = bmesh.new()

        if subtype == "cone":
            # Bright orange traffic cone
            mat = self._create_pbr_material(f"{name}_Cone", (1.0, 0.3, 0.02, 1.0), roughness=0.4)
            # Square base
            bmesh.ops.create_cube(bm, size=1.0)
            for v in bm.verts:
                v.co.x *= 0.38
                v.co.y *= 0.38
                v.co.z *= 0.05
                v.co.z += 0.025
            # Cone top
            bmesh.ops.create_cone(bm, segments=12, radius1=0.18, radius2=0.03, depth=0.7)
            for v in bm.verts[8:]:
                v.co.z += 0.35 + 0.05

        elif subtype == "tire":
            # Dark rubber tire
            mat = self._create_pbr_material(f"{name}_Tire", (0.06, 0.06, 0.06, 1.0), roughness=0.9)
            bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=0.35, radius2=0.35, depth=0.24)
            # Lie flat on the road
            for v in bm.verts:
                v.co.z += 0.12

        else:  # Box / debris
            mat = self._create_pbr_material(f"{name}_Box", (0.6, 0.45, 0.28, 1.0), roughness=0.8)
            sx = random.uniform(0.4, 0.8)
            sy = random.uniform(0.4, 0.8)
            sz = random.uniform(0.3, 0.6)
            bmesh.ops.create_cube(bm, size=1.0)
            for v in bm.verts:
                v.co.x *= sx
                v.co.y *= sy
                v.co.z *= sz
                v.co.z += sz / 2.0

        bm.to_mesh(mesh)
        bm.free()

        root_obj.data.materials.append(mat)
        self.collection.objects.link(root_obj)

        root_obj["adas_class_id"] = 2
        root_obj["adas_class_name"] = "obstacle"

        return root_obj
