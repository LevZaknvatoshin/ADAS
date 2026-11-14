#!/usr/bin/env python3
"""
depth_estimator.py — Depth Anything V2 Small Monocular Depth Estimation
========================================================================
Based on RoadCam MVP:
Estimates relative monocular depth map for a road-camera image.
Uses Hugging Face Transformers and Depth Anything V2 Small (Apache-2.0).
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def estimate_depth(
    image_path: str,
    output_npy: str | None = None,
    preview_png: str | None = None,
) -> np.ndarray:
    """
    Computes normalized relative depth map (0 = near, 1 = far).
    Returns float32 numpy array.
    """
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DepthEstimator] Loading {MODEL_ID} on {device} ({w}x{h})...")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        pred = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().float().cpu().numpy()

    # Monocular depth is relative: robust percentile normalization
    lo, hi = np.percentile(pred, [2, 98])
    normalized = np.clip((pred - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)

    # Invert so 0.0 = near, 1.0 = far
    normalized = 1.0 - normalized

    if output_npy:
        os.makedirs(os.path.dirname(os.path.abspath(output_npy)) or ".", exist_ok=True)
        if output_npy.endswith(".npz"):
            np.savez_compressed(output_npy, depth=normalized)
        else:
            np.save(output_npy, normalized)
        print(f"[DepthEstimator] Saved normalized depth: {output_npy}")

    if preview_png:
        os.makedirs(os.path.dirname(os.path.abspath(preview_png)) or ".", exist_ok=True)
        Image.fromarray((normalized * 255).astype(np.uint8)).save(preview_png)
        print(f"[DepthEstimator] Saved depth preview: {preview_png}")

    return normalized


def main():
    p = argparse.ArgumentParser(description="Estimate relative depth map using Depth Anything V2")
    p.add_argument("--input", "-i", required=True, help="Input road photo path")
    p.add_argument("--output", "-o", default="depth.npy", help="Output .npy or .npz path")
    p.add_argument("--preview", "-p", default=None, help="Optional preview PNG path")
    args = p.parse_args()

    estimate_depth(args.input, args.output, args.preview)


if __name__ == "__main__":
    main()
