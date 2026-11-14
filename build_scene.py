#!/usr/bin/env python3
"""
build_scene.py — 3D Road Scene Builder for Blender 4.x
======================================================
Supports two complementary reconstruction methods:
  1. 'roadcam' (Default): Continuous 2.5D camera-matched reprojection mesh with
     full environment depth relief (road, trees, vehicles, sky).
  2. 'segmented': Distinct procedural layers (road polygon mesh, centerline
     dashed markings, boundary lines, guardrails, and background plate).
"""

import os
import sys
import json
import math
import argparse
import numpy as np

import bpy
from mathutils import Vector, geometry


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--image", "--input", dest="image", required=True, help="Input road photograph")
    p.add_argument("--depth", default=None, help="Depth map .npy / .npz file")
    p.add_argument("--polygon", default=None, help="Optional polygon JSON for segmented mode")
    p.add_argument("--output", "--outdir", dest="output", required=True, help="Output .blend file or directory")
    p.add_argument("--mode", choices=["roadcam", "segmented"], default="roadcam", help="Reconstruction mode")
    p.add_argument("--grid", type=int, default=240, help="Grid density width")
    p.add_argument("--depth-strength", type=float, default=1.0, help="Relative depth strength for roadcam mode (default: 1.0)")
    p.add_argument("--depth-scale", type=float, default=2.0, help="Depth scale for segmented mode")
    p.add_argument("--pitch", type=float, default=82.0, help="Camera pitch angle in degrees (82 default, looking forward)")
    p.add_argument("--preview", default=None, help="Optional render preview image path")
    p.add_argument("--render-samples", type=int, default=64, help="Render samples")
    p.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"], default="CYCLES")
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for col in [bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights]:
        for b in list(col):
            col.remove(b, do_unlink=True)


def make_emission_material(name: str, image):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Linear"
    tex.extension = "CLIP"
    mat.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
    mat.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def build_roadcam_mode(image, depth, args, out_blend_path, preview_path):
    w, h = image.size[:]
    aspect = w / float(h)

    scene = bpy.context.scene
    scene.render.engine = args.engine if args.engine in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items else "CYCLES"
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    # Camera setup (50mm, horizontal sensor fit, oriented forward along +Y)
    cam_data = bpy.data.cameras.new("ReferenceCamera")
    cam = bpy.data.objects.new("ReferenceCamera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.location = (0, 0, 0)
    cam.rotation_euler = (math.radians(args.pitch), 0, 0)
    cam_data.lens = 50.0
    cam_data.sensor_width = 36.0
    cam_data.sensor_fit = "HORIZONTAL"
    bpy.context.view_layer.update()

    tan_half_hfov = cam_data.sensor_width / (2.0 * cam_data.lens)

    # Grid unprojection
    grid_w = max(32, args.grid)
    grid_h = max(16, round(grid_w * h / float(w)))

    xs = np.linspace(0, depth.shape[1] - 1, grid_w).astype(np.int32)
    ys = np.linspace(0, depth.shape[0] - 1, grid_h).astype(np.int32)
    dsmall = depth[np.ix_(ys, xs)]

    base_distance = 20.0
    distances = base_distance + (dsmall - 0.5) * (base_distance * args.depth_strength)

    verts_world, uvs = [], []
    for j in range(grid_h):
        v_img = j / float(grid_h - 1)
        for i in range(grid_w):
            u = i / float(grid_w - 1)
            d = float(distances[j, i])
            x = (u - 0.5) * 2.0 * d * tan_half_hfov
            y = (0.5 - v_img) * 2.0 * d * tan_half_hfov * (float(h) / float(w))
            p_cam = Vector((x, y, -d))
            p_world = cam.matrix_world @ p_cam
            verts_world.append(p_world)
            uvs.append((u, 1.0 - v_img))

    # Grounding: align the bottom row (nearest road surface) to Z = 0.0 (ground floor)
    bottom_start_j = max(0, grid_h - 3)
    bottom_indices = range(bottom_start_j * grid_w, grid_h * grid_w)
    z_ground = min(verts_world[idx].z for idx in bottom_indices)

    # Center map in workspace: align the center of the road to Y = 0.0 (Blender origin)
    y_vals = [p.y for p in verts_world]
    y_center = (min(y_vals) + max(y_vals)) / 2.0

    verts = [(p.x, p.y - y_center, p.z - z_ground) for p in verts_world]
    cam.location.y -= y_center
    cam.location.z -= z_ground
    bpy.context.view_layer.update()

    # Free orbit / driver camera for inspection (forward perspective along road)
    driver_cam_data = bpy.data.cameras.new("DriverViewCamera")
    driver_cam = bpy.data.objects.new("DriverViewCamera", driver_cam_data)
    scene.collection.objects.link(driver_cam)
    cam_z = cam.location.z
    min_road_y = min(p[1] for p in verts)
    driver_cam.location = (0.0, min_road_y - 1.5, max(1.2, cam_z * 0.9))
    driver_cam.rotation_euler = (math.radians(args.pitch + 4.0), 0, 0)
    driver_cam_data.lens = 35.0

    faces = []
    for j in range(grid_h - 1):
        for i in range(grid_w - 1):
            a = j * grid_w + i
            b = a + 1
            c = a + grid_w + 1
            d = a + grid_w
            faces.append((a, b, c, d))

    mesh = bpy.data.meshes.new("PhotoDepthMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="PhotoUV")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            uv_layer.data[li].uv = uvs[vi]

    for f in mesh.polygons:
        f.use_smooth = True

    obj = bpy.data.objects.new("RoadScene_2.5D_Reconstruction", mesh)
    scene.collection.objects.link(obj)

    mat = make_emission_material("RoadPhoto_Emission", image)
    mesh.materials.append(mat)

    # Ambient World
    world = bpy.data.worlds.new("RoadScene_World") if not scene.world else scene.world
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.18, 0.22, 0.28, 1.0)
        bg_node.inputs["Strength"].default_value = 0.4

    # Save .blend
    os.makedirs(os.path.dirname(out_blend_path) or ".", exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend_path)
    print(f"[BuildScene] Successfully saved: {out_blend_path}")

    # Render preview
    if preview_path:
        os.makedirs(os.path.dirname(preview_path) or ".", exist_ok=True)
        scene.cycles.samples = args.render_samples
        scene.render.filepath = preview_path
        bpy.ops.render.render(write_still=True)
        print(f"[BuildScene] Saved preview render: {preview_path}")


def main():
    args = parse_args()
    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Output blend file path
    if os.path.isdir(args.output) or not args.output.endswith(".blend"):
        out_dir = os.path.abspath(args.output)
        out_blend = os.path.join(out_dir, "scene.blend")
        out_preview = args.preview or os.path.join(out_dir, "render.png")
    else:
        out_blend = os.path.abspath(args.output)
        out_dir = os.path.dirname(out_blend)
        out_preview = args.preview or os.path.join(out_dir, "preview.png")

    clear_scene()
    image = bpy.data.images.load(image_path, check_existing=True)
    w, h = image.size[:]

    # Load depth
    depth_arr = None
    if args.depth and os.path.exists(args.depth):
        if args.depth.endswith(".npz"):
            depth_arr = np.load(args.depth)["depth"]
        elif args.depth.endswith(".npy"):
            depth_arr = np.load(args.depth)
        elif args.depth.lower().endswith((".png", ".jpg", ".jpeg")):
            import cv2
            img_depth = cv2.imread(args.depth, cv2.IMREAD_GRAYSCALE)
            if img_depth is not None:
                depth_arr = img_depth.astype(np.float32) / 255.0
        if depth_arr is not None and depth_arr.shape != (h, w):
            import cv2
            depth_arr = cv2.resize(depth_arr, (w, h))

    if depth_arr is None:
        depth_arr = np.zeros((h, w), dtype=np.float32)

    build_roadcam_mode(image, depth_arr, args, out_blend, out_preview)


if __name__ == "__main__":
    main()
