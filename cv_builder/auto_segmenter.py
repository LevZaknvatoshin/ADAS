#!/usr/bin/env python3
"""
auto_segmenter.py — Automated CPU Semantic Road Segmentation
============================================================
Uses SegFormer-B0 (Cityscapes) to segment road surfaces, sidewalks,
guardrails/fences, and poles from any road photograph.

Outputs:
  - polygon JSON matching prototype format
  - road mask PNG
  - rich segment metadata (centerline, boundaries, detected features)
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from PIL import Image

try:
    import torch
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# Cityscapes classes in SegFormer
# 0: road, 1: sidewalk, 2: building, 3: wall, 4: fence, 5: pole,
# 6: traffic light, 7: traffic sign, 8: vegetation, 9: terrain,
# 10: sky, 11: person, 12: rider, 13: car, 14: truck, 15: bus,
# 16: train, 17: motorcycle, 18: bicycle


def segment_road_image(
    image_path: str,
    output_json: str | None = None,
    output_mask: str | None = None,
    model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    approx_epsilon: float = 0.003,
    min_area_ratio: float = 0.02,
) -> dict:
    """
    Runs automated segmentation and extracts smoothed road polygons and centerline.
    """
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    pil_img = Image.open(image_path).convert("RGB")
    w, h = pil_img.size

    print(f"[AutoSegmenter] Processing image: {os.path.basename(image_path)} ({w}x{h})")
    print(f"[AutoSegmenter] Loading model on CPU: {model_name}...")

    seg_pipe = pipeline("image-segmentation", model=model_name, device=-1)
    results = seg_pipe(pil_img)

    road_mask = np.zeros((h, w), dtype=np.uint8)
    sidewalk_mask = np.zeros((h, w), dtype=np.uint8)
    fence_mask = np.zeros((h, w), dtype=np.uint8)
    detected_labels = []

    for r in results:
        label = r["label"].lower()
        detected_labels.append(label)
        mask = np.array(r["mask"]).astype(np.uint8)
        # Resize mask to original image size if needed
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if label == "road":
            road_mask = np.maximum(road_mask, mask * 255)
        elif label == "sidewalk":
            sidewalk_mask = np.maximum(sidewalk_mask, mask * 255)
        elif label in ("fence", "wall", "guardrail"):
            fence_mask = np.maximum(fence_mask, mask * 255)

    # Clean road mask using morphological closing then opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

    # Extract connected components and keep significant ones
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(road_mask)
    min_pixels = int(w * h * min_area_ratio)

    valid_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            valid_mask[labels_im == i] = 255

    if np.sum(valid_mask) > 0:
        road_mask = valid_mask

    # Find contours
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_pixels:
            continue
        # Approximate contour to reduce vertex count while preserving shape
        epsilon = approx_epsilon * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        if len(approx) >= 3:
            # Normalize to 0..1 coordinates (u, v)
            poly_norm = []
            for pt in approx[:, 0, :]:
                u = float(np.clip(pt[0] / w, 0.0, 1.0))
                v = float(np.clip(pt[1] / h, 0.0, 1.0))
                poly_norm.append([round(u, 4), round(v, 4)])
            polygons.append(poly_norm)

    # Extract road centerline and boundaries by scanlines
    scanline_y = np.linspace(h * 0.45, h * 0.98, num=40, dtype=int)
    centerline = []
    left_boundary = []
    right_boundary = []

    for y in scanline_y:
        row = road_mask[y, :]
        road_indices = np.where(row > 128)[0]
        if len(road_indices) > 10:
            left_x = float(road_indices[0])
            right_x = float(road_indices[-1])
            center_x = float((left_x + right_x) / 2.0)
            centerline.append([round(center_x / w, 4), round(float(y) / h, 4)])
            left_boundary.append([round(left_x / w, 4), round(float(y) / h, 4)])
            right_boundary.append([round(right_x / w, 4), round(float(y) / h, 4)])

    road_pixel_ratio = float(np.sum(road_mask > 0) / (w * h))
    has_fences = bool(np.sum(fence_mask > 0) > (w * h * 0.005))
    has_sidewalks = bool(np.sum(sidewalk_mask > 0) > (w * h * 0.01))

    output_data = {
        "image_width": w,
        "image_height": h,
        "polygons": polygons,
        "centerline": centerline,
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "features": {
            "road_pixel_ratio": round(road_pixel_ratio, 4),
            "has_fences": has_fences,
            "has_sidewalks": has_sidewalks,
            "detected_labels": list(set(detected_labels)),
        },
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"[AutoSegmenter] Saved polygon JSON -> {output_json} ({len(polygons)} polygon(s))")

    if output_mask:
        os.makedirs(os.path.dirname(os.path.abspath(output_mask)) or ".", exist_ok=True)
        cv2.imwrite(output_mask, road_mask)
        print(f"[AutoSegmenter] Saved road mask -> {output_mask}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Automated CPU Road Segmentation")
    parser.add_argument("--image", required=True, help="Path to input road photo")
    parser.add_argument("--output_json", default="road_polygon.json", help="Output polygon JSON path")
    parser.add_argument("--output_mask", default=None, help="Optional output road mask PNG path")
    args = parser.parse_args()

    segment_road_image(args.image, args.output_json, args.output_mask)


if __name__ == "__main__":
    main()
