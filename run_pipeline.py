#!/usr/bin/env python3
"""
run_pipeline.py — One-Command Road Photo to 3D Blender Scene Pipeline
====================================================================
End-to-end CPU neural pipeline based on RoadCam 2.5D:
  1. Monocular Depth Estimation (Depth Anything V2 Small)
  2. 3D Camera-Matched Reprojection in Blender (Mesh, UVs, ReferenceCamera, DriverCamera)
  3. Automatic Preview Rendering

Usage:
  python run_pipeline.py --image samples/roadcam_highway.jpg --output output_highway/scene.blend
"""

import os
import sys
import shutil
import argparse
import subprocess

from depth_estimator import estimate_depth


def find_blender() -> str | None:
    """Finds Blender 4.x or 3.x executable."""
    which_b = shutil.which("blender")
    if which_b:
        return which_b

    candidates = [
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\Blender 4.1\blender.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_pipeline(
    image_path: str,
    output_target: str = "output/scene.blend",
    grid: int = 240,
    depth_strength: float = 1.0,
    render_samples: int = 64,
    blender_path: str | None = None,
    render_preview: bool = True,
):
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if output_target.endswith(".blend"):
        output_blend = os.path.abspath(output_target)
        output_dir = os.path.dirname(output_blend) or "."
    else:
        output_dir = os.path.abspath(output_target)
        output_blend = os.path.join(output_dir, "scene.blend")

    os.makedirs(output_dir, exist_ok=True)
    depth_npy = os.path.join(output_dir, "depth.npy")
    preview_png = os.path.join(output_dir, "render.png") if render_preview else None
    depth_preview_png = os.path.join(output_dir, "depth_preview.png")

    print("=" * 70)
    print(" ROADCAM PHOTO -> 3D BLENDER PIPELINE")
    print("=" * 70)
    print(f" Input Image:       {image_path}")
    print(f" Output Scene:      {output_blend}")
    print(f" Grid Resolution:   {grid}")
    print(f" Depth Strength:    {depth_strength}")
    print("-" * 70)

    # ── Stage 1: Depth Estimation ──────────────────────────────────────
    print("\n[STAGE 1/2] Estimating Monocular Depth (Depth Anything V2 Small)...")
    estimate_depth(image_path, output_npy=depth_npy, preview_png=depth_preview_png)

    # ── Stage 2: Blender 3D Reconstruction ─────────────────────────────
    print("\n[STAGE 2/2] Generating 3D Scene in Blender...")
    blender_bin = blender_path or find_blender()
    if not blender_bin:
        raise FileNotFoundError("Blender executable not found! Please specify --blender path.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    builder_script = os.path.join(script_dir, "build_scene.py")

    cmd = [
        blender_bin,
        "-b",
        "-P", builder_script,
        "--",
        "--image", image_path,
        "--depth", depth_npy,
        "--output", output_blend,
        "--grid", str(grid),
        "--depth-strength", str(depth_strength),
        "--render-samples", str(render_samples),
    ]
    if preview_png:
        cmd.extend(["--preview", preview_png])

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[ERROR] Blender builder failed with code {ret.returncode}")
        sys.exit(ret.returncode)

    print("\n" + "=" * 70)
    print(" [OK] PIPELINE SUCCESSFULLY COMPLETED!")
    print(f"  - Blender Scene:   {output_blend}")
    if preview_png and os.path.exists(preview_png):
        print(f"  - Render Preview:  {preview_png}")
    print(f"  - Depth Map NPY:   {depth_npy}")
    print(f"  - Depth Preview:   {depth_preview_png}")
    print("=" * 70)


def main():
    p = argparse.ArgumentParser(description="One-Command RoadCam Photo to 3D Blender Scene")
    p.add_argument("--image", "--input", "-i", dest="image", required=True, help="Input road photograph path")
    p.add_argument("--output", "--outdir", "-o", dest="output", default="output/scene.blend", help="Output .blend or directory")
    p.add_argument("--grid", "-g", type=int, default=240, help="Grid resolution width (default: 240)")
    p.add_argument("--depth-strength", "-d", dest="depth_strength", type=float, default=1.0, help="Depth relief strength (default: 1.0)")
    p.add_argument("--render-samples", "-s", type=int, default=64, help="Cycles render samples (default: 64)")
    p.add_argument("--blender", default=None, help="Path to blender.exe")
    p.add_argument("--no_render", action="store_true", help="Skip rendering verification preview image")
    args = p.parse_args()

    run_pipeline(
        image_path=args.image,
        output_target=args.output,
        grid=args.grid,
        depth_strength=args.depth_strength,
        render_samples=args.render_samples,
        blender_path=args.blender,
        render_preview=not args.no_render,
    )


if __name__ == "__main__":
    main()
