"""
Ground Truth QA Visualizer
==========================
Renders visual confirmation overlays with 2D bounding boxes and class tags
directly on top of the rendered images to verify annotation accuracy.
"""

import os

COLOR_MAP = {
    0: (0.1, 0.9, 0.2),   # Pedestrian: Green
    1: (0.2, 0.5, 1.0),   # Vehicle: Blue
    2: (1.0, 0.3, 0.1),   # Obstacle: Red/Orange
}

CLASS_NAMES = {
    0: "pedestrian",
    1: "vehicle",
    2: "obstacle",
}


def draw_bboxes_on_image(image_path, annotation_lines, output_path):
    """
    Draws 2D bounding boxes on image_path and saves to output_path.
    Uses PIL if available, otherwise falls back to Blender's native image pixel buffer.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        _draw_with_pil(image_path, annotation_lines, output_path)
    except ImportError:
        _draw_with_blender(image_path, annotation_lines, output_path)


def _draw_with_pil(image_path, annotation_lines, output_path):
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Responsive line thickness based on image resolution
    line_thickness = max(2, int(round(w / 400.0)))
    label_pad = max(12, int(round(h / 35.0)))

    for line in annotation_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])

        # Ignore tiny stray edge clipping slivers (< 1.5% of frame)
        if bw < 0.015 and bh < 0.015:
            continue

        x_min = int((xc - bw / 2.0) * w)
        y_min = int((yc - bh / 2.0) * h)
        x_max = int((xc + bw / 2.0) * w)
        y_max = int((yc + bh / 2.0) * h)

        c = COLOR_MAP.get(cls_id, (1.0, 1.0, 0.0))
        rgb = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

        # Draw thick rectangle
        for offset in range(line_thickness):
            draw.rectangle(
                [x_min - offset, y_min - offset, x_max + offset, y_max + offset],
                outline=rgb,
            )

        # Label tag
        label_text = f"{CLASS_NAMES.get(cls_id, 'class')}_{cls_id}"
        draw.text((x_min + 3, max(0, y_min - label_pad)), label_text, fill=rgb)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, quality=98)


def _draw_with_blender(image_path, annotation_lines, output_path):
    import bpy

    # Load image into Blender
    img = bpy.data.images.load(image_path)
    w = img.size[0]
    h = img.size[1]
    pixels = list(img.pixels)  # float array of size w * h * 4

    for line in annotation_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])

        x_min = max(0, min(w - 1, int((xc - bw / 2.0) * w)))
        x_max = max(0, min(w - 1, int((xc + bw / 2.0) * w)))
        # Invert Y for Blender coordinate system (0 is bottom)
        y_top = max(0, min(h - 1, int((1.0 - (yc - bh / 2.0)) * h)))
        y_bottom = max(0, min(h - 1, int((1.0 - (yc + bh / 2.0)) * h)))

        c = COLOR_MAP.get(cls_id, (1.0, 1.0, 0.0))
        r, g, b = c[0], c[1], c[2]

        def set_pixel(px, py):
            if 0 <= px < w and 0 <= py < h:
                idx = (py * w + px) * 4
                pixels[idx] = r
                pixels[idx + 1] = g
                pixels[idx + 2] = b
                pixels[idx + 3] = 1.0

        # Draw horizontal borders
        for x in range(x_min, x_max + 1):
            for t in range(-1, 2):
                set_pixel(x, y_bottom + t)
                set_pixel(x, y_top + t)

        # Draw vertical borders
        for y in range(y_bottom, y_top + 1):
            for t in range(-1, 2):
                set_pixel(x_min + t, y)
                set_pixel(x_max + t, y)

    img.pixels = pixels
    img.filepath_raw = output_path
    img.file_format = "JPEG"
    img.save()
    bpy.data.images.remove(img)
