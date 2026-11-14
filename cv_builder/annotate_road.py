#!/usr/bin/env python3
"""
annotate_road.py — Interactive Road Polygon Annotation & Review Tool
===================================================================
Visualizes AI-generated road polygons or allows manual polygon drawing.
CPU-only OpenCV GUI.

Controls:
  - Left Click: Add polygon vertex
  - Enter: Finish current polygon
  - S: Save polygons and exit
  - R: Clear/reset current polygon
  - C: Clear all polygons
  - Esc: Quit without saving
"""

import os
import sys
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

points = []
polygons = []
display_scale = 1.0
image = None
canvas = None


def redraw():
    global canvas
    canvas = image.copy()
    h, w = image.shape[:2]

    # Draw saved / pre-existing polygons (Green)
    for idx, poly in enumerate(polygons):
        pts = np.array([(int(x * w), int(y * h)) for x, y in poly], dtype=np.int32)
        if len(pts) >= 3:
            # Semi-transparent overlay
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], (0, 180, 0))
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
            cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
            cv2.putText(
                canvas, f"Poly #{idx+1}", (pts[0][0], pts[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

    # Draw currently active points and lines (Yellow / Red)
    curr_pts = [(int(x * w), int(y * h)) for x, y in points]
    for p in curr_pts:
        cv2.circle(canvas, p, 5, (0, 0, 255), -1)
    if len(curr_pts) >= 2:
        cv2.polylines(canvas, [np.array(curr_pts, dtype=np.int32)], False, (0, 255, 255), 2)

    # Instructions overlay at top
    header_text = f"Polygons: {len(polygons)} | LeftClick=Add | Enter=Close Poly | S=Save & Quit | R=Reset | C=Clear All | Esc=Quit"
    cv2.rectangle(canvas, (0, 0), (w, 35), (20, 20, 20), -1)
    cv2.putText(canvas, header_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    view = cv2.resize(canvas, None, fx=display_scale, fy=display_scale)
    cv2.imshow("Road Annotation Review", view)


def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and image is not None:
        h, w = image.shape[:2]
        u = float(np.clip(x / display_scale / w, 0.0, 1.0))
        v = float(np.clip(y / display_scale / h, 0.0, 1.0))
        points.append((u, v))
        redraw()


def review_or_annotate(
    image_path: str,
    output_json: str = "road_polygon.json",
    initial_json: str | None = None,
    max_width: int = 1400,
) -> list:
    global image, display_scale, polygons, points
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]
    display_scale = min(1.0, max_width / float(w))
    polygons = []
    points = []

    # Preload polygons if provided
    load_path = initial_json or (output_json if os.path.exists(output_json) else None)
    if load_path and os.path.exists(load_path):
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                loaded_polys = data.get("polygons", [])
                if loaded_polys:
                    polygons = loaded_polys
                    print(f"[AnnotateRoad] Preloaded {len(polygons)} polygon(s) from {load_path}")
        except Exception as e:
            print(f"[AnnotateRoad] Warning loading initial JSON: {e}")

    cv2.namedWindow("Road Annotation Review", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Road Annotation Review", mouse)
    redraw()

    saved = False
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):  # Enter: close current polygon
            if len(points) >= 3:
                polygons.append(points.copy())
                points.clear()
                redraw()
                print(f"[AnnotateRoad] Polygon closed. Total: {len(polygons)}")
            else:
                print("[AnnotateRoad] Need at least 3 points to close polygon.")
        elif key in (ord("s"), ord("S")):  # S: Save & exit
            if len(points) >= 3:
                polygons.append(points.copy())
                points.clear()
            out_data = {
                "image_width": w,
                "image_height": h,
                "polygons": polygons,
            }
            os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)
            print(f"[AnnotateRoad] Saved {len(polygons)} polygon(s) to {output_json}")
            saved = True
            break
        elif key in (ord("r"), ord("R")):  # R: Reset current points
            points.clear()
            redraw()
        elif key in (ord("c"), ord("C")):  # C: Clear all
            points.clear()
            polygons.clear()
            redraw()
            print("[AnnotateRoad] Cleared all polygons.")
        elif key == 27:  # Esc: cancel
            print("[AnnotateRoad] Cancelled without saving changes.")
            break

    cv2.destroyAllWindows()
    return polygons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default="road_polygon.json")
    ap.add_argument("--initial", default=None)
    ap.add_argument("--max_width", type=int, default=1400)
    args = ap.parse_args()

    review_or_annotate(args.image, args.output, args.initial, args.max_width)


if __name__ == "__main__":
    main()
