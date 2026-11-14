#!/usr/bin/env python3
"""
road_twin_reconstructor.py — 3D Road Digital Twin Engine for Blender
===================================================================
Constructs a photorealistic 3D road digital twin from an input photograph,
semantic segmentation polygons, and monocular depth relief.
"""

import os
import math
import numpy as np

import bpy
from mathutils import Vector, geometry


COLLECTION_NAME = "ADAS_Road_Twin"


def clean_old_road_objects(scene=None):
    """Cleanly removes previously generated road twin objects and materials."""
    if COLLECTION_NAME in bpy.data.collections:
        coll = bpy.data.collections[COLLECTION_NAME]
        for obj in list(coll.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(coll)

    # Clean orphaned materials starting with ADAS_
    for mat in list(bpy.data.materials):
        if mat.name.startswith("ADAS_Road_") and mat.users == 0:
            bpy.data.materials.remove(mat)

    # Clean legacy lane guide collections if present
    if "ADAS_Lane_Lines" in bpy.data.collections:
        coll = bpy.data.collections["ADAS_Lane_Lines"]
        for obj in list(coll.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(coll)

    # Clean default Blender starter objects, legacy extra driver cameras, and legacy route line
    for def_name in ["Cube", "Light", "Camera", "RoadCam_DriverViewCamera", "ADAS_DriverCamera", "ADAS_Vehicle_Route"]:
        if def_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[def_name], do_unlink=True)


def get_or_create_collection(scene):
    if COLLECTION_NAME in bpy.data.collections:
        return bpy.data.collections[COLLECTION_NAME]
    coll = bpy.data.collections.new(COLLECTION_NAME)
    scene.collection.children.link(coll)
    return coll


def make_photo_plate_material(name: str, image):
    """Unlit emission material for 100% photo background fidelity."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Linear"

    mat.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
    mat.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_pbr_road_material(name: str, image):
    """
    Principled BSDF material combining the camera photo projection
    with procedural asphalt micro-texture, roughness, and bump.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    # Photo texture node
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Linear"

    # Procedural asphalt bump
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 80.0
    noise.inputs["Detail"].default_value = 4.0

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.15
    bump.inputs["Distance"].default_value = 0.05

    links = mat.node_tree.links
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.75
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.35

    return mat


def make_marking_material(name: str, color=(0.95, 0.95, 0.95, 1.0)):
    """Reflective road paint material."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.25
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (color[0] * 0.1, color[1] * 0.1, color[2] * 0.1, 1.0)

    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_guardrail_material(name: str):
    """Galvanized steel guardrail material."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.75, 0.78, 0.82, 1.0)
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.95
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.25

    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def interpolate_depth(depth_map: np.ndarray | None, u: float, v: float, scale: float) -> float:
    """Samples depth map with bilinear interpolation."""
    if depth_map is None:
        return float((1.0 - v) * scale)

    dh, dw = depth_map.shape[:2]
    px = float(np.clip(u * (dw - 1), 0, dw - 1))
    py = float(np.clip(v * (dh - 1), 0, dh - 1))

    x0 = int(math.floor(px))
    x1 = min(x0 + 1, dw - 1)
    y0 = int(math.floor(py))
    y1 = min(y0 + 1, dh - 1)

    fx = px - x0
    fy = py - y0

    val = (
        depth_map[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + depth_map[y0, x1] * fx * (1.0 - fy)
        + depth_map[y1, x0] * (1.0 - fx) * fy
        + depth_map[y1, x1] * fx * fy
    )
    return float(val * scale)


def point_in_polygon(u: float, v: float, poly: list) -> bool:
    """Ray casting point-in-polygon test in normalized (u, v) space."""
    n = len(poly)
    inside = False
    p1u, p1v = poly[0]
    for i in range(1, n + 1):
        p2u, p2v = poly[i % n]
        if v > min(p1v, p2v):
            if v <= max(p1v, p2v):
                if u <= max(p1u, p2u):
                    if p1v != p2v:
                        u_cross = (v - p1v) * (p2u - p1u) / (p2v - p1v) + p1u
                    if p1u == p2u or u <= u_cross:
                        inside = not inside
        p1u, p1v = p2u, p2v
    return inside


def build_3d_road_mesh(
    name: str,
    poly: list,
    depth_map: np.ndarray | None,
    depth_scale: float,
    aspect: float,
    material,
    collection,
    grid_res: int = 60,
):
    poly_arr = np.array(poly)
    min_u, min_v = float(poly_arr[:, 0].min()), float(poly_arr[:, 1].min())
    max_u, max_v = float(poly_arr[:, 0].max()), float(poly_arr[:, 1].max())

    u_vals = np.linspace(min_u, max_u, grid_res)
    v_vals = np.linspace(min_v, max_v, grid_res)

    verts = []
    uvs = []
    vert_map = {}

    for j, v in enumerate(v_vals):
        for i, u in enumerate(u_vals):
            if point_in_polygon(u, v, poly):
                z_depth = -interpolate_depth(depth_map, u, v, depth_scale)
                x = (u - 0.5) * aspect
                y = 0.5 - v
                vert_idx = len(verts)
                verts.append((x, y, z_depth))
                uvs.append((u, 1.0 - v))
                vert_map[(i, j)] = vert_idx

    faces = []
    for j in range(grid_res - 1):
        for i in range(grid_res - 1):
            corners = [
                vert_map.get((i, j)),
                vert_map.get((i + 1, j)),
                vert_map.get((i + 1, j + 1)),
                vert_map.get((i, j + 1)),
            ]
            if all(c is not None for c in corners):
                faces.append((corners[0], corners[1], corners[2]))
                faces.append((corners[0], corners[2], corners[3]))

    # Fallback to polygon boundary tessellation
    if len(faces) == 0:
        verts, uvs = [], []
        for u, v in poly:
            z_depth = -interpolate_depth(depth_map, u, v, depth_scale)
            verts.append(((u - 0.5) * aspect, 0.5 - v, z_depth))
            uvs.append((u, 1.0 - v))
        face_indices = geometry.tessellate_polygon([[Vector(v) for v in verts]])
        faces = [tuple(verts.index(tuple(v)) for v in tri) for tri in face_indices]

    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="PhotoProjection")
    for poly_face in mesh.polygons:
        for loop_idx in poly_face.loop_indices:
            v_idx = mesh.loops[loop_idx].vertex_index
            uv_layer.data[loop_idx].uv = uvs[v_idx]

    for f in mesh.polygons:
        f.use_smooth = True

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    obj["role"] = "road_surface_3d"
    return obj


def build_road_ribbon(
    name: str,
    curve_points: list,
    depth_map: np.ndarray | None,
    depth_scale: float,
    aspect: float,
    material,
    collection,
    width_m: float = 0.008,
    z_offset: float = 0.003,
    is_dashed: bool = False,
):
    if len(curve_points) < 2:
        return None

    verts, uvs, faces = [], [], []
    for k in range(0, len(curve_points) - 1):
        if is_dashed and ((k // 2) % 2 == 1):
            continue

        p1 = curve_points[k]
        p2 = curve_points[min(k + 1, len(curve_points) - 1)]

        u1, v1 = p1
        u2, v2 = p2

        z1 = -interpolate_depth(depth_map, u1, v1, depth_scale) + z_offset
        z2 = -interpolate_depth(depth_map, u2, v2, depth_scale) + z_offset

        x1 = (u1 - 0.5) * aspect
        y1 = 0.5 - v1
        x2 = (u2 - 0.5) * aspect
        y2 = 0.5 - v2

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        nx = -dy / length * (width_m / 2.0)
        ny = dx / length * (width_m / 2.0)

        v_idx = len(verts)
        verts.extend([
            (x1 - nx, y1 - ny, z1),
            (x1 + nx, y1 + ny, z1),
            (x2 + nx, y2 + ny, z2),
            (x2 - nx, y2 - ny, z2),
        ])
        uvs.extend([(0, 0), (1, 0), (1, 1), (0, 1)])
        faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))

    if not faces:
        return None

    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly_face in mesh.polygons:
        for loop_idx in poly_face.loop_indices:
            v_idx = mesh.loops[loop_idx].vertex_index
            uv_layer.data[loop_idx].uv = uvs[v_idx]

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    obj["role"] = "road_marking"
    return obj


def build_guardrail_barrier(
    name: str,
    boundary_points: list,
    depth_map: np.ndarray | None,
    depth_scale: float,
    aspect: float,
    material,
    collection,
    rail_height: float = 0.04,
    rail_offset_z: float = 0.02,
):
    if len(boundary_points) < 4:
        return None

    sampled = boundary_points[::2]
    if len(sampled) < 2:
        return None

    verts, faces = [], []
    for k in range(len(sampled) - 1):
        u1, v1 = sampled[k]
        u2, v2 = sampled[k + 1]

        z1 = -interpolate_depth(depth_map, u1, v1, depth_scale) + rail_offset_z
        z2 = -interpolate_depth(depth_map, u2, v2, depth_scale) + rail_offset_z

        x1 = (u1 - 0.5) * aspect
        y1 = 0.5 - v1
        x2 = (u2 - 0.5) * aspect
        y2 = 0.5 - v2

        v_idx = len(verts)
        verts.extend([
            (x1, y1, z1),
            (x1, y1, z1 + rail_height),
            (x2, y2, z2 + rail_height),
            (x2, y2, z2),
        ])
        faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))

        if k % 3 == 0:
            post_idx = len(verts)
            pw = 0.005
            verts.extend([
                (x1 - pw, y1 - pw, z1 - rail_offset_z),
                (x1 + pw, y1 - pw, z1 - rail_offset_z),
                (x1 + pw, y1 + pw, z1 + rail_height),
                (x1 - pw, y1 + pw, z1 + rail_height),
            ])
            faces.append((post_idx, post_idx + 1, post_idx + 2, post_idx + 3))

    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    obj["role"] = "guardrail_barrier"
    return obj


def build_road_digital_twin(
    image_path: str,
    polygon_data: dict,
    depth_map: np.ndarray | None = None,
    depth_scale: float = 2.0,
    scene=None,
) -> dict:
    """
    Main entry point for Blender add-on.
    Constructs the complete 3D digital twin in the active Blender scene.
    """
    if scene is None:
        scene = bpy.context.scene

    image_path = os.path.abspath(image_path)
    clean_old_road_objects(scene)
    coll = get_or_create_collection(scene)

    # Load reference image into Blender
    image = bpy.data.images.load(image_path, check_existing=True)
    w, h = image.size
    aspect = w / float(h)

    # Set scene resolution
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100

    # 1. Source Camera (exact match to photo plate)
    cam_name = "ADAS_SourceCamera"
    if cam_name in bpy.data.objects:
        source_cam = bpy.data.objects[cam_name]
    else:
        cam_data = bpy.data.cameras.new(cam_name)
        source_cam = bpy.data.objects.new(cam_name, cam_data)
        coll.objects.link(source_cam)

    source_cam.location = (0, 0, 3)
    source_cam.rotation_euler = (0, 0, 0)
    source_cam.data.type = "ORTHO"
    source_cam.data.ortho_scale = aspect
    scene.camera = source_cam


    # 3. Photo Reference Plate
    plate_mat = make_photo_plate_material("ADAS_Road_PhotoPlate_Mat", image)
    z_plate = -depth_scale - 0.05
    half_w = aspect * 0.5
    plate_verts = [
        (-half_w, -0.5, z_plate),
        (half_w, -0.5, z_plate),
        (half_w, 0.5, z_plate),
        (-half_w, 0.5, z_plate),
    ]
    plate_faces = [(0, 1, 2, 3)]
    plate_uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    plate_mesh = bpy.data.meshes.new("ADAS_PhotoReference_Plate_Mesh")
    plate_mesh.from_pydata(plate_verts, [], plate_faces)
    plate_mesh.update()

    uv_layer = plate_mesh.uv_layers.new(name="UVMap")
    for poly_face in plate_mesh.polygons:
        for loop_idx in poly_face.loop_indices:
            v_idx = plate_mesh.loops[loop_idx].vertex_index
            uv_layer.data[loop_idx].uv = plate_uvs[v_idx]

    plate = bpy.data.objects.new("ADAS_PhotoReference_Plate", plate_mesh)
    coll.objects.link(plate)
    plate.data.materials.append(plate_mat)
    plate["role"] = "camera_reference_plate"

    # 4. Materials
    road_pbr_mat = make_pbr_road_material("ADAS_Road_PBR_Mat", image)
    marking_mat = make_marking_material("ADAS_Road_Marking_Mat")
    barrier_mat = make_guardrail_material("ADAS_Road_Guardrail_Mat")

    # 5. Build 3D Road Meshes
    polygons = polygon_data.get("polygons", [])
    road_objs = []
    for idx, poly in enumerate(polygons, start=1):
        if len(poly) >= 3:
            obj = build_3d_road_mesh(
                f"ADAS_RoadSurface_3D_{idx:02d}",
                poly,
                depth_map,
                depth_scale,
                aspect,
                road_pbr_mat,
                coll,
            )
            road_objs.append(obj)

    # 6. Road Markings
    centerline = polygon_data.get("centerline", [])
    if centerline:
        build_road_ribbon(
            "ADAS_RoadMarking_Centerline",
            centerline,
            depth_map,
            depth_scale,
            aspect,
            marking_mat,
            coll,
            width_m=0.008,
            is_dashed=True,
        )

    left_bound = polygon_data.get("left_boundary", [])
    if left_bound:
        build_road_ribbon(
            "ADAS_RoadMarking_Left_Boundary",
            left_bound,
            depth_map,
            depth_scale,
            aspect,
            marking_mat,
            coll,
            width_m=0.006,
        )

    right_bound = polygon_data.get("right_boundary", [])
    if right_bound:
        build_road_ribbon(
            "ADAS_RoadMarking_Right_Boundary",
            right_bound,
            depth_map,
            depth_scale,
            aspect,
            marking_mat,
            coll,
            width_m=0.006,
        )

    # 7. Guardrails
    features = polygon_data.get("features", {})
    if features.get("has_fences", False) or right_bound:
        if right_bound and len(right_bound) >= 4:
            build_guardrail_barrier(
                "ADAS_RoadBarrier_Right",
                right_bound,
                depth_map,
                depth_scale,
                aspect,
                barrier_mat,
                coll,
            )

    # 8. Sun Light
    sun_name = "ADAS_SunLight"
    if sun_name not in bpy.data.objects:
        sun_data = bpy.data.lights.new(name=sun_name, type="SUN")
        sun_data.energy = 2.5
        sun_obj = bpy.data.objects.new(sun_name, sun_data)
        coll.objects.link(sun_obj)
        sun_obj.rotation_euler = (math.radians(45), math.radians(25), math.radians(15))

    return {
        "status": "SUCCESS",
        "road_objects": len(road_objs),
        "aspect": aspect,
        "width": w,
        "height": h,
    }


def detect_sky_horizon(depth_map: np.ndarray, poly_data: dict | None = None) -> float:
    """
    Detects the normalized v-coordinate (0.0 to 1.0) of the sky horizon line.
    Rows with v < v_horizon correspond to the sky / overhead canopy.
    """
    dh = depth_map.shape[0]

    # 1. Check if JSON metadata has calibration horizon line directly
    if poly_data and "calibration" in poly_data and "horizon_normalized_v" in poly_data["calibration"]:
        calib_v = float(poly_data["calibration"]["horizon_normalized_v"])
        if 0.10 <= calib_v <= 0.85:
            return calib_v

    # 2. Check if road polygon data provides an upper boundary for the road
    if poly_data and "polygons" in poly_data and len(poly_data["polygons"]) > 0:
        min_vs = []
        for poly in poly_data["polygons"]:
            if len(poly) > 0:
                min_vs.append(min(p[1] for p in poly))
        if min_vs:
            road_top_v = min(min_vs)
            if 0.10 <= road_top_v <= 0.85:
                return float(np.clip(road_top_v - 0.02, 0.15, 0.60))

    # 3. Check depth map gradient: transition between sky and distant road
    row_means = depth_map.mean(axis=1)
    r_start = int(0.15 * dh)
    r_end = int(0.65 * dh)
    if r_end > r_start + 5:
        diffs = np.abs(np.diff(row_means[r_start:r_end]))
        if len(diffs) > 0:
            max_idx = r_start + int(np.argmax(diffs))
            detected_v = max_idx / float(dh - 1)
            if 0.20 <= detected_v <= 0.55:
                return float(detected_v)

    # Default to 0.35 (standard horizon line for road cameras)
    return 0.35


def build_roadcam_25d_scene(
    image_path: str,
    depth_map: np.ndarray,
    depth_strength: float = 1.0,
    grid_w: int = 240,
    base_distance: float = 20.0,
    cam_pitch: float = 82.0,
    scene=None,
    use_pbr: bool = False,
    clip_sky_roof: bool = True,
    sky_mode: str = "REMOVE",
    sky_cutoff: float = 0.0,
    poly_data: dict | None = None,
    max_road_distance: float = 250.0,
    near_road_distance: float = 6.0,
) -> dict:
    """
    Constructs a 2.5D camera-matched reprojection mesh based on RoadCam MVP.
    Reconstructs entire environment (road, trees, sky, buildings, guardrails)
    from depth map with exact camera perspective alignment.

    Oriented horizontally along Blender's ground plane (XY plane, stretching
    forward along +Y, Z up) so the road lies naturally on the floor.
    Eliminates the overhead roof/cave canopy above the road (clip_sky_roof).
    """
    if scene is None:
        scene = bpy.context.scene

    image_path = os.path.abspath(image_path)
    clean_old_road_objects(scene)
    coll = get_or_create_collection(scene)

    image = bpy.data.images.load(image_path, check_existing=True)
    w, h = image.size[:]
    aspect = w / float(h)

    # Set scene render settings
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100

    # Grid dimensions
    grid_w = max(32, int(grid_w))
    grid_h = max(16, round(grid_w * h / float(w)))

    # Resample depth map to grid
    xs = np.linspace(0, depth_map.shape[1] - 1, grid_w).astype(np.int32)
    ys = np.linspace(0, depth_map.shape[0] - 1, grid_h).astype(np.int32)
    dsmall = depth_map[np.ix_(ys, xs)]

    # Reference camera — oriented forward (+Y) with natural downward pitch
    cam_name = "RoadCam_ReferenceCamera"
    if cam_name in bpy.data.objects:
        cam = bpy.data.objects[cam_name]
    else:
        cam_data = bpy.data.cameras.new(cam_name)
        cam = bpy.data.objects.new(cam_name, cam_data)
        coll.objects.link(cam)

    d_near = max(1.0, float(near_road_distance))
    d_far = max(20.0, float(max_road_distance))

    cam.location = (0, 0, 0)
    cam.rotation_euler = (math.radians(cam_pitch), 0, 0)
    cam.data.lens = 50.0
    cam.data.sensor_width = 36.0
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.clip_start = 0.1
    cam.data.clip_end = max(1000.0, d_far * 3.0)

    # Configure camera background image for seamless camera view
    cam.data.show_background_images = True
    if not cam.data.background_images:
        bg = cam.data.background_images.new()
    else:
        bg = cam.data.background_images[0]
    bg.image = image
    bg.frame_method = "FIT"
    bg.display_depth = "BACK"

    scene.camera = cam
    bpy.context.view_layer.update()

    # Tangent of half horizontal FOV
    tan_half_hfov = cam.data.sensor_width / (2.0 * cam.data.lens)

    # Determine horizon line to eliminate the overhead roof canopy
    if sky_cutoff > 0.01:
        horizon_v = float(sky_cutoff)
    else:
        horizon_v = detect_sky_horizon(depth_map, poly_data)

    horizon_j = int(round(horizon_v * (grid_h - 1)))
    horizon_j = max(0, min(horizon_j, grid_h - 5))

    # Determine rows to generate based on sky_mode
    if clip_sky_roof and sky_mode == "REMOVE":
        # Pure open scene: only generate rows from the horizon to the bottom (road/ground)
        # Completely eliminates any overhead roof, arch, or cave ceiling!
        start_j = horizon_j
        end_j = grid_h
    else:
        start_j = 0
        end_j = grid_h

    # ── Natural 2.5D Depth Relief without extreme horizon stretching ──
    # The previous power-law stretching (to 250m) caused severe texture smearing / needle distortion.
    # Restoring proportional depth keeps the road crisp, realistic and compact (~20-30m length).
    d_sub = dsmall[start_j:end_j, :]
    d_p2 = float(np.percentile(d_sub, 2))
    d_p98 = float(np.percentile(d_sub, 98))
    d_norm = np.clip((d_sub - d_p2) / max(1e-5, d_p98 - d_p2), 0.0, 1.0)

    # Use base_distance with AI depth relief
    # Road stays crisp and naturally proportioned without longitudinal smearing:
    road_distances = base_distance + (d_norm - 0.5) * (base_distance * float(depth_strength))
    road_distances = np.maximum(4.0, road_distances)

    # Target horizon Y distance for vertical backdrop mode
    y_horizon_target = base_distance * 1.5

    verts_world = []
    uvs = []
    for j in range(start_j, end_j):
        j_local = j - start_j
        v_img = j / float(grid_h - 1)
        for i in range(grid_w):
            u = i / float(grid_w - 1)
            d = float(road_distances[j_local, i])
            x = (u - 0.5) * 2.0 * d * tan_half_hfov
            y = (0.5 - v_img) * 2.0 * d * tan_half_hfov * (float(h) / float(w))
            p_cam = Vector((x, y, -d))
            p_world = cam.matrix_world @ p_cam

            # In BACKDROP mode, project sky rays onto the vertical horizon wall
            if clip_sky_roof and sky_mode == "BACKDROP" and j < horizon_j:
                ray_dir = (p_world - cam.location).normalized()
                if ray_dir.y > 1e-4:
                    scale = (y_horizon_target - cam.location.y) / ray_dir.y
                    p_world = cam.location + ray_dir * scale

            verts_world.append(p_world)
            uvs.append((u, 1.0 - v_img))

    # Grounding: align the nearest road surface (bottom rows of image) with Z = 0.0 (Blender floor)
    bottom_start_j = max(0, len(verts_world) - 3 * grid_w)
    z_ground = min(verts_world[idx].z for idx in range(bottom_start_j, len(verts_world)))

    # Center map in workspace: align the center of the road to Y = 0.0 (Blender origin)
    # Placing the road at the center of the 3D grid makes viewing, orbiting and object placement intuitive and natural.
    # Camera shifts synchronously so ReferenceCamera view remains 100% pixel-perfect matching the photo.
    y_vals = [p.y for p in verts_world]
    y_center = (min(y_vals) + max(y_vals)) / 2.0

    verts = [(p.x, p.y - y_center, p.z - z_ground) for p in verts_world]
    cam.location.y -= y_center
    cam.location.z -= z_ground
    bpy.context.view_layer.update()

    num_rows = end_j - start_j
    faces = []
    for j in range(num_rows - 1):
        for i in range(grid_w - 1):
            a = j * grid_w + i
            b = a + 1
            c = a + grid_w + 1
            d = a + grid_w
            # Use front-facing / upward winding order (a, d, c, b)
            faces.append((a, d, c, b))

    mesh = bpy.data.meshes.new("RoadCam_DepthMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="PhotoUV")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            uv_layer.data[li].uv = uvs[vi]

    for f in mesh.polygons:
        f.use_smooth = True

    obj = bpy.data.objects.new("RoadCam_2.5D_SceneMesh", mesh)
    coll.objects.link(obj)

    # Material
    if use_pbr:
        mat = make_pbr_road_material("RoadCam_PBR_Material", image)
    else:
        mat = make_photo_plate_material("RoadCam_Emission_Material", image)
    mesh.materials.append(mat)
    obj["role"] = "roadcam_2.5d_reconstruction"

    # Physics Static Collider: enables real physical drop & collision on road surface
    try:
        if not obj.rigid_body:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.rigidbody.object_add()
        if obj.rigid_body:
            obj.rigid_body.type = "PASSIVE"
            obj.rigid_body.collision_shape = "MESH"
            obj.rigid_body.friction = 0.85
            obj.rigid_body.restitution = 0.05
    except Exception:
        pass


    # Ambient World
    if not scene.world:
        scene.world = bpy.data.worlds.new("RoadCam_World")
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.18, 0.22, 0.28, 1.0)
        bg_node.inputs["Strength"].default_value = 0.5

    return {
        "status": "SUCCESS",
        "mesh_name": obj.name,
        "vertices": len(verts),
        "faces": len(faces),
        "aspect": aspect,
        "width": w,
        "height": h,
    }


def make_route_marker_material(name="ADAS_Route_Marker_Material"):
    """Glowing cyan-emerald material for 3D road trajectory guide."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emission = nt.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.05, 0.95, 0.70, 1.0)
    emission.inputs["Strength"].default_value = 2.5
    nt.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def build_vehicle_route_curve(scene, road_mesh_obj=None, coll=None, num_points=8):
    """
    Constructs an editable 3D Bezier curve along the center line of the reconstructed road.
    The route is grounded directly on the road surface from near the car to the horizon.
    """
    if coll is None:
        coll = get_or_create_collection(scene)

    curve_name = "ADAS_Vehicle_Route"
    if curve_name in bpy.data.objects:
        old_obj = bpy.data.objects[curve_name]
        bpy.data.objects.remove(old_obj, do_unlink=True)

    if road_mesh_obj is None:
        road_mesh_obj = (
            bpy.data.objects.get("RoadCam_2.5D_SceneMesh") or
            bpy.data.objects.get("RoadScene_2.5D_Reconstruction") or
            bpy.data.objects.get("ADAS_Carriageway_Mesh")
        )

    pts = []
    if road_mesh_obj and road_mesh_obj.type == "MESH":
        # Extract drivable road center vertices (|X| < 0.4 and Z < 4.5)
        verts = [road_mesh_obj.matrix_world @ v.co for v in road_mesh_obj.data.vertices
                 if abs(v.co.x) < 0.4 and v.co.z < 4.5]
        verts.sort(key=lambda p: p.y)
        if len(verts) >= num_points:
            step = max(1, len(verts) // num_points)
            pts = [verts[i] for i in range(0, len(verts), step)][:num_points]

    if not pts or len(pts) < 2:
        # Fallback road center line along +Y on road plane (centered at Y=0)
        pts = [
            Vector((0.0, -5.0, 0.1)),
            Vector((0.0, -1.0, 0.5)),
            Vector((0.0, 2.0, 1.0)),
            Vector((0.0, 5.0, 1.5)),
        ]

    # Lift points slightly above asphalt so curve is clearly visible (4cm above road)
    pts = [Vector((p.x, p.y, p.z + 0.04)) for p in pts]

    curve_data = bpy.data.curves.new(curve_name + "_Data", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.05
    curve_data.resolution_u = 4

    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(pts) - 1)
    for i, pt in enumerate(pts):
        bp = spline.bezier_points[i]
        bp.co = pt
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"

    curve_obj = bpy.data.objects.new(curve_name, curve_data)
    coll.objects.link(curve_obj)
    curve_obj["role"] = "vehicle_route"

    mat = make_route_marker_material()
    curve_data.materials.append(mat)

    return curve_obj
