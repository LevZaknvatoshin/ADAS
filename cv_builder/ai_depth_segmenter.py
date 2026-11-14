"""
AI Depth & Semantic Segmentation Module
=======================================
Integrates Depth Anything V2 and SegFormer (Cityscapes) for state-of-the-art
monocular depth estimation and pixel-perfect semantic masking on any road photo:
  - Depth Anything V2: dense depth / disparity map (hills, slopes, road contour)
  - SegFormer: semantic masks (Road, Grass/Terrain, Sky, Buildings, Poles)
  - Sky dome extraction for environment lighting
  - 2D pole detection for procedural 3D object instancing
"""

import os
import sys
import json
import subprocess
import numpy as np
import cv2
from PIL import Image


class AIDepthSegmenter:
    """Preprocesses input image into depth map and semantic category masks"""

    CATEGORIES = ["road", "terrain", "sky", "poles", "buildings"]

    def __init__(self, image_path, cache_dir=None):
        self.image_path = os.path.abspath(image_path)
        if cache_dir is None:
            try:
                from .project_paths import get_project_paths
                paths = get_project_paths()
                self.cache_dir = paths.get_environment_dir(self.image_path)
            except Exception:
                base_dir = os.path.dirname(self.image_path)
                stem = os.path.splitext(os.path.basename(self.image_path))[0]
                self.cache_dir = os.path.join(base_dir, f".cache_ai_{stem}")
        else:
            self.cache_dir = os.path.abspath(cache_dir)

        os.makedirs(self.cache_dir, exist_ok=True)

        self.depth_map_path = os.path.join(self.cache_dir, "depth_map.png")
        self.mask_paths = {
            "road": os.path.join(self.cache_dir, "mask_road.png"),
            "terrain": os.path.join(self.cache_dir, "mask_terrain.png"),
            "sky": os.path.join(self.cache_dir, "mask_sky.png"),
            "poles": os.path.join(self.cache_dir, "mask_poles.png"),
            "buildings": os.path.join(self.cache_dir, "mask_buildings.png"),
        }
        self.sky_dome_path = os.path.join(self.cache_dir, "sky_dome.png")
        self.metadata_path = os.path.join(self.cache_dir, "metadata.json")

        self.metadata = {}
        self.poles_2d = []

    def is_cached(self):
        """Check if precomputed depth and all masks already exist"""
        if not os.path.exists(self.depth_map_path):
            return False
        for path in self.mask_paths.values():
            if not os.path.exists(path):
                return False
        return os.path.exists(self.metadata_path)

    def process(self, force=False):
        """Processes image with AI models or loads from cache"""
        if not force and self.is_cached():
            print(f"[AIDepthSegmenter] Loading cached results from: {self.cache_dir}")
            self._load_cached_metadata()
            return self.get_results()

        print(f"[AIDepthSegmenter] Running AI models for: {self.image_path}")
        # Try direct in-process inference (if transformers/torch available)
        try:
            import torch
            from transformers import pipeline
            self._run_inference_direct()
        except (ImportError, ModuleNotFoundError):
            # Run via system python subprocess
            print("[AIDepthSegmenter] Running via system Python subprocess...")
            self._run_inference_subprocess()

        self._load_cached_metadata()
        return self.get_results()

    def _run_inference_direct(self):
        from transformers import pipeline

        img_pil = Image.open(self.image_path).convert("RGB")
        w, h = img_pil.size

        # 1. Depth Anything V2
        print("[AIDepthSegmenter] Estimating Depth via Depth-Anything-V2-Small...")
        depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
        depth_pil = depth_pipe(img_pil)["depth"]
        # Ensure matches original size
        if depth_pil.size != (w, h):
            depth_pil = depth_pil.resize((w, h), Image.Resampling.BILINEAR)
        depth_pil.save(self.depth_map_path)

        # 2. SegFormer (Cityscapes)
        print("[AIDepthSegmenter] Segmenting Semantics via SegFormer Cityscapes...")
        seg_pipe = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        seg_res = seg_pipe(img_pil)

        mask_road = np.zeros((h, w), dtype=np.uint8)
        mask_terrain = np.zeros((h, w), dtype=np.uint8)
        mask_sky = np.zeros((h, w), dtype=np.uint8)
        mask_poles = np.zeros((h, w), dtype=np.uint8)
        mask_buildings = np.zeros((h, w), dtype=np.uint8)

        for r in seg_res:
            lbl = r["label"]
            m = np.array(r["mask"].resize((w, h), Image.Resampling.NEAREST))
            if lbl in ["road", "sidewalk"]:
                mask_road = np.maximum(mask_road, m)
            elif lbl in ["vegetation", "terrain"]:
                mask_terrain = np.maximum(mask_terrain, m)
            elif lbl == "sky":
                mask_sky = np.maximum(mask_sky, m)
            elif lbl in ["pole", "traffic sign", "traffic light"]:
                mask_poles = np.maximum(mask_poles, m)
            elif lbl in ["building", "wall", "fence"]:
                mask_buildings = np.maximum(mask_buildings, m)

        # Post-process masks
        self._postprocess_and_save(img_pil, depth_pil, mask_road, mask_terrain, mask_sky, mask_poles, mask_buildings)

    def _run_inference_subprocess(self):
        """Invoke external script with system python if current python lacks transformers"""
        runner_script_path = os.path.join(self.cache_dir, "_ai_runner.py")
        runner_code = f"""# -*- coding: utf-8 -*-
import sys, os, numpy as np, json, cv2
from PIL import Image
from transformers import pipeline

img_path = r"{self.image_path}"
cache_dir = r"{self.cache_dir}"
os.makedirs(cache_dir, exist_ok=True)

img_pil = Image.open(img_path).convert("RGB")
w, h = img_pil.size

print("Running Depth Anything V2...")
depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
depth_pil = depth_pipe(img_pil)["depth"]
if depth_pil.size != (w, h):
    depth_pil = depth_pil.resize((w, h), Image.Resampling.BILINEAR)
depth_map_path = os.path.join(cache_dir, "depth_map.png")
depth_pil.save(depth_map_path)

print("Running SegFormer...")
seg_pipe = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
seg_res = seg_pipe(img_pil)

mask_road = np.zeros((h, w), dtype=np.uint8)
mask_terrain = np.zeros((h, w), dtype=np.uint8)
mask_sky = np.zeros((h, w), dtype=np.uint8)
mask_poles = np.zeros((h, w), dtype=np.uint8)
mask_buildings = np.zeros((h, w), dtype=np.uint8)

for r in seg_res:
    lbl = r["label"]
    m = np.array(r["mask"].resize((w, h), Image.Resampling.NEAREST))
    if lbl in ["road", "sidewalk"]:
        mask_road = np.maximum(mask_road, m)
    elif lbl in ["vegetation", "terrain"]:
        mask_terrain = np.maximum(mask_terrain, m)
    elif lbl == "sky":
        mask_sky = np.maximum(mask_sky, m)
    elif lbl in ["pole", "traffic sign", "traffic light"]:
        mask_poles = np.maximum(mask_poles, m)
    elif lbl in ["building", "wall", "fence"]:
        mask_buildings = np.maximum(mask_buildings, m)

img_rgb = np.array(img_pil)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

Image.fromarray(mask_road).save(os.path.join(cache_dir, "mask_road.png"))
Image.fromarray(mask_terrain).save(os.path.join(cache_dir, "mask_terrain.png"))
Image.fromarray(mask_sky).save(os.path.join(cache_dir, "mask_sky.png"))
Image.fromarray(mask_poles).save(os.path.join(cache_dir, "mask_poles.png"))
Image.fromarray(mask_buildings).save(os.path.join(cache_dir, "mask_buildings.png"))

# Sky dome texture
sky_dome = np.zeros_like(img_bgr)
sky_bin = mask_sky > 64
if sky_bin.sum() > 100:
    sky_dome[sky_bin] = img_bgr[sky_bin]
    # Inpaint or extend downward
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    sky_dil = cv2.dilate(mask_sky, kernel)
    inpaint_mask = ((sky_dil > 0) & (~sky_bin)).astype(np.uint8) * 255
    sky_dome = cv2.inpaint(sky_dome, inpaint_mask, 5, cv2.INPAINT_TELEA)
else:
    # Night or indoor: dark gradient
    sky_dome = np.full_like(img_bgr, 15)

sky_dome_rgb = cv2.cvtColor(sky_dome, cv2.COLOR_BGR2RGB)
Image.fromarray(sky_dome_rgb).save(os.path.join(cache_dir, "sky_dome.png"))

# Extract pole 2D positions
poles_bin = (mask_poles > 64).astype(np.uint8)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(poles_bin, connectivity=8)
poles_list = []
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > 40:
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        base_x = bx + bw // 2
        base_y = by + bh  # bottom base of pole
        poles_list.append({{"base_x": base_x, "base_y": base_y, "width": bw, "height": bh, "area": int(area)}})

# Scene brightness / night detection
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
is_night = bool(gray.mean() < 65.0)

meta = {{
    "width": w,
    "height": h,
    "is_night": is_night,
    "road_pixels": int((mask_road > 64).sum()),
    "terrain_pixels": int((mask_terrain > 64).sum()),
    "sky_pixels": int((mask_sky > 64).sum()),
    "poles_count": len(poles_list),
    "poles_2d": poles_list,
}}
with open(os.path.join(cache_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("Inference completed successfully.")
"""
        with open(runner_script_path, "w", encoding="utf-8") as f_scr:
            f_scr.write(runner_code)

        candidates = [
            r"C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe",
            "python",
        ]
        py_exe = None
        for c in candidates:
            if os.path.exists(c):
                py_exe = c
                break
        if not py_exe:
            py_exe = "python"

        cmd = [py_exe, runner_script_path]
        print(f"[AIDepthSegmenter] Launching: {cmd[0]} {cmd[1]}")
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0:
            print(f"[AIDepthSegmenter ERROR] stderr:\n{res.stderr}")
            print(f"[AIDepthSegmenter ERROR] stdout:\n{res.stdout}")
            raise RuntimeError(f"AI Depth & Segmentation failed with code {res.returncode}:\n{res.stderr}")
        print(res.stdout)

    def _postprocess_and_save(self, img_pil, depth_pil, mask_road, mask_terrain, mask_sky, mask_poles, mask_buildings):
        w, h = img_pil.size
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        Image.fromarray(mask_road).save(self.mask_paths["road"])
        Image.fromarray(mask_terrain).save(self.mask_paths["terrain"])
        Image.fromarray(mask_sky).save(self.mask_paths["sky"])
        Image.fromarray(mask_poles).save(self.mask_paths["poles"])
        Image.fromarray(mask_buildings).save(self.mask_paths["buildings"])

        # Sky dome extraction
        sky_bin = mask_sky > 64
        sky_dome = np.zeros_like(img_bgr)
        if sky_bin.sum() > 100:
            sky_dome[sky_bin] = img_bgr[sky_bin]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            sky_dil = cv2.dilate(mask_sky, kernel)
            inpaint_mask = ((sky_dil > 0) & (~sky_bin)).astype(np.uint8) * 255
            sky_dome = cv2.inpaint(sky_dome, inpaint_mask, 5, cv2.INPAINT_TELEA)
        else:
            sky_dome = np.full_like(img_bgr, 15)
        Image.fromarray(cv2.cvtColor(sky_dome, cv2.COLOR_BGR2RGB)).save(self.sky_dome_path)

        # Detect pole base positions
        poles_bin = (mask_poles > 64).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(poles_bin, connectivity=8)
        self.poles_2d = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 40:
                bx = int(stats[i, cv2.CC_STAT_LEFT])
                by = int(stats[i, cv2.CC_STAT_TOP])
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                base_x = bx + bw // 2
                base_y = by + bh
                self.poles_2d.append({
                    "base_x": base_x,
                    "base_y": base_y,
                    "width": bw,
                    "height": bh,
                    "area": int(area),
                })

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        is_night = bool(gray.mean() < 65.0)

        self.metadata = {
            "width": w,
            "height": h,
            "is_night": is_night,
            "road_pixels": int((mask_road > 64).sum()),
            "terrain_pixels": int((mask_terrain > 64).sum()),
            "sky_pixels": int((mask_sky > 64).sum()),
            "poles_count": len(self.poles_2d),
            "poles_2d": self.poles_2d,
        }

        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _load_cached_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
                self.poles_2d = self.metadata.get("poles_2d", [])

    def get_results(self):
        return {
            "depth_map": self.depth_map_path,
            "masks": self.mask_paths,
            "sky_dome": self.sky_dome_path,
            "metadata": self.metadata,
            "poles_2d": self.poles_2d,
        }
