"""
Batch Renderer & Dataset Exporter
=================================
Executes frame-by-frame rendering and annotations saving:
  - Generates data.yaml for YOLOv8/v9/v10
  - Handles variations and position randomization for massive dataset generation
  - Saves QA images with visual bounding boxes
"""

import bpy
import os
import random
from .camera_rig import CameraRig
from .environment import EnvironmentManager
from .obstacles import ObstacleManager
from .annotator import YOLOAnnotator
from .visualizer import draw_bboxes_on_image


class BatchRenderer:
    def __init__(self, props=None):
        self.props = props or bpy.context.scene.adas_props
        self.annotator = YOLOAnnotator()
        self.obs_mgr = ObstacleManager()
        self.env_mgr = EnvironmentManager()

    def _resolve_dir(self, dir_path):
        if not dir_path or dir_path.strip() == "":
            dir_path = "//adas_dataset"

        if dir_path.startswith("//"):
            blend_dir = bpy.path.abspath("//")
            # If scene is unsaved, bpy.path.abspath("//") defaults to Blender exe folder (Program Files) which is read-only
            if not blend_dir or blend_dir == "//" or "Program Files" in blend_dir:
                user_home = os.path.expanduser("~")
                blend_dir = os.path.join(user_home, "Desktop")
                if not os.path.exists(blend_dir):
                    blend_dir = user_home
            dir_path = os.path.join(blend_dir, dir_path[2:])
        return os.path.abspath(dir_path)

    def _setup_directories(self, base_dir):
        dirs = {
            "base": base_dir,
            "train_images": os.path.join(base_dir, "images", "train"),
            "val_images": os.path.join(base_dir, "images", "val"),
            "train_labels": os.path.join(base_dir, "labels", "train"),
            "val_labels": os.path.join(base_dir, "labels", "val"),
            "vis": os.path.join(base_dir, "visualizations"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        return dirs

    def _write_yaml(self, base_dir):
        norm_path = base_dir.replace("\\", "/")
        yaml_content = f"""# ADAS Road Obstacle Dataset for YOLOv8 / YOLOv9 / YOLOv10
path: {norm_path}
train: images/train
val: images/val

# Target Classes
nc: 3
names:
  0: person
  1: vehicle
  2: obstacle
"""
        yaml_path = os.path.join(base_dir, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        return yaml_path

    def _configure_engine(self, scene):
        # Ensure session temp directory exists so Cycles tiles write without error
        if hasattr(bpy.app, "tempdir") and not os.path.exists(bpy.app.tempdir):
            try:
                os.makedirs(bpy.app.tempdir, exist_ok=True)
            except Exception:
                pass

        engine = self.props.render_engine
        scene.render.engine = engine

        # Resolution super-sampling
        res_scale = int(getattr(self.props, "render_resolution_scale", "200"))
        scene.render.resolution_percentage = res_scale

        if "CYCLES" in engine:
            scene.cycles.samples = 64
            scene.cycles.preview_samples = 32
            scene.cycles.use_denoising = True
            if hasattr(scene.cycles, "tile_size"):
                scene.cycles.tile_size = 2048
        else:
            if hasattr(scene, "eevee"):
                scene.eevee.taa_render_samples = 64
                scene.eevee.use_gtao = True

        scene.render.image_settings.file_format = "JPEG"
        scene.render.image_settings.quality = 98

    def run(self):
        scene = bpy.context.scene
        if not scene.camera:
            cam = (scene.objects.get("RoadCam_ReferenceCamera") or
                   scene.objects.get("ADAS_SourceCamera") or
                   scene.objects.get("ReferenceCamera") or
                   scene.objects.get("ADAS_Camera"))
            if cam:
                scene.camera = cam
            else:
                raise RuntimeError("В сцене нет активной камеры для рендера! Перейдите во вкладку камер или перестройте сцену.")

        base_dir = self._resolve_dir(self.props.output_dir)
        dirs = self._setup_directories(base_dir)
        self._write_yaml(base_dir)
        self._configure_engine(scene)

        total_variations = max(1, self.props.batch_variations)
        frames_step = max(1, self.props.frame_step)

        print(f"\n=======================================================")
        print(f"🎬 ADAS Simulation Batch Render: {total_variations} variations")
        print(f"📁 Destination: {base_dir}")
        print(f"=======================================================\n")

        weather_options = ["OVERCAST", "SUNNY", "SUNSET", "NIGHT", "RAIN"]
        original_items = list(self.props.scene_items)

        total_frames_rendered = 0
        total_annotations_saved = 0

        for var_idx in range(1, total_variations + 1):
            split = "val" if (var_idx % 6 == 0) else "train"
            img_dir = dirs[f"{split}_images"]
            lbl_dir = dirs[f"{split}_labels"]

            # 1. Weather variation if enabled
            if self.props.randomize_weather:
                w = random.choice(weather_options)
                self.env_mgr.apply_weather(w)
            else:
                self.env_mgr.apply_weather(self.props.weather)

            # 2. Objects in scene
            col = bpy.data.collections.get("ADAS_Objects")
            existing_objs = [o for o in col.objects if o.type == 'MESH' or o.children] if col else []

            # If objects are not yet in the scene, spawn them once from scenario list
            if not existing_objs:
                self.obs_mgr.clear()
                for item_idx, item in enumerate(original_items):
                    self.obs_mgr.spawn_item(item, index=item_idx, total_frames=self.props.total_frames, fps=self.props.fps)
            # If objects already exist with manual keyframes or baked physics,
            # KEEP THEM INTACT so the user's manual animations are preserved 1:1.

            # 3. Render frames
            start_f = scene.frame_start
            end_f = min(scene.frame_end, self.props.total_frames) if self.props.total_frames > 0 else scene.frame_end

            for f in range(start_f, end_f + 1, frames_step):
                scene.frame_set(f)

                frame_name = f"var_{var_idx:04d}_frame_{f:04d}"
                img_path = os.path.join(img_dir, f"{frame_name}.jpg")
                lbl_path = os.path.join(lbl_dir, f"{frame_name}.txt")

                scene.render.filepath = img_path
                if hasattr(bpy.app, "tempdir") and not os.path.exists(bpy.app.tempdir):
                    try:
                        os.makedirs(bpy.app.tempdir, exist_ok=True)
                    except Exception:
                        pass
                bpy.ops.render.render(write_still=True)
                total_frames_rendered += 1

                # YOLO Ground Truth
                annotations = self.annotator.get_frame_annotations(scene)
                yolo_lines = self.annotator.format_yolo_lines(annotations)

                with open(lbl_path, "w", encoding="utf-8") as f_lbl:
                    f_lbl.writelines(yolo_lines)
                total_annotations_saved += len(annotations)

                # QA Visual Overlay (first frame of each variation)
                if self.props.export_qa_images and (f == start_f or f % 20 == 0):
                    vis_path = os.path.join(dirs["vis"], f"{frame_name}_vis.jpg")
                    try:
                        draw_bboxes_on_image(img_path, yolo_lines, vis_path)
                    except Exception as e:
                        print(f"Warning QA visualization: {e}")

            print(f"   ✓ Variation {var_idx}/{total_variations} completed (Split: {split})")

        print(f"\n=======================================================")
        print(f"✅ Batch Render Finished!")
        print(f"📊 Frames Saved: {total_frames_rendered}")
        print(f"🏷️  Annotations: {total_annotations_saved}")
        print(f"📦 Config File: {os.path.join(base_dir, 'data.yaml')}")
        print(f"=======================================================\n")
