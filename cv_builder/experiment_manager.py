"""
Experiment Manager for ADAS CV Builder
=======================================
Manages experiment lifecycle:
  - Create new experiments with auto-numbered directories
  - Save full scenario state (environment, scene items, render settings)
  - Load and restore scenarios from JSON
  - List all existing experiments

Each experiment directory contains:
  - scenario.json    — full scene configuration
  - renders/         — rendered frame images
  - yolo_dataset/    — YOLO-format labels and images
  - qa_preview/      — QA visualization images
"""

import os
import json
from datetime import datetime

from .project_paths import ProjectPaths


class ExperimentManager:
    """Manages experiment lifecycle: create, save scenario, load scenario."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.current_dir = None  # Set when experiment is created/loaded

    def create_experiment(self, name: str = "") -> str:
        """
        Creates a new experiment directory, returns path.

        The directory is auto-numbered as experiments/exp_NNN_<name>/.
        Sub-directories (renders/, yolo_dataset/, qa_preview/) are also created.
        """
        self.current_dir = self.paths.get_next_experiment_dir(name)

        # Create standard sub-directories
        os.makedirs(self.renders_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.qa_dir, exist_ok=True)

        return self.current_dir

    def save_scenario(self, scene, props) -> str:
        """
        Saves current scenario state to scenario.json in current experiment.

        Includes: environment image path, all scene items (type, position,
        rotation, scale, animation params), render settings, timestamp.

        Args:
            scene: bpy.context.scene — the Blender scene object
            props: scene.adas_props — the ADAS property group

        Returns:
            Path to the saved scenario.json file.

        Raises:
            RuntimeError: If no experiment directory is set (call create_experiment first).
        """
        if not self.current_dir:
            raise RuntimeError(
                "No experiment directory set. Call create_experiment() first."
            )

        # Gather environment info
        ref_image = getattr(props, "reference_image", "")
        env_source = os.path.basename(ref_image) if ref_image else ""
        env_stem = os.path.splitext(env_source)[0] if env_source else ""
        env_dir_rel = f"environments/{env_stem}" if env_stem else ""

        environment_data = {
            "source_image": env_source,
            "environment_dir": env_dir_rel,
            "depth_strength": float(getattr(props, "depth_strength", 1.0)),
            "grid_resolution": int(getattr(props, "grid_resolution", 240)),
            "cam_pitch": float(getattr(props, "roadcam_pitch",
                                       getattr(props, "cam_pitch", 82.0))),
        }

        # Gather scene items
        scene_items_data = []
        for item in props.scene_items:
            item_data = {
                "name": item.name,
                "obj_type": str(getattr(item, "obj_type", "OBSTACLE")),
                "subtype": str(getattr(item, "subtype", "")),
                "pos_x": float(item.pos_x),
                "pos_y": float(item.pos_y),
                "rotation_z": float(item.rotation_z),
                "scale_factor": float(item.scale_factor),
                "animated": bool(item.animated),
                "end_x": float(item.end_x),
                "end_y": float(item.end_y),
                "speed_kmh": float(item.speed_kmh),
                "scenario_behavior": str(
                    getattr(item, "scenario_behavior", "STATIC")
                ),
            }

            # Include custom model path if applicable
            custom_path = getattr(item, "custom_model_path", "")
            if custom_path:
                item_data["custom_model_path"] = custom_path

            scene_items_data.append(item_data)

        # Gather render settings
        render_settings = {
            "engine": str(getattr(props, "render_engine", scene.render.engine)),
            "total_frames": int(getattr(props, "total_frames", 60)),
            "fps": int(getattr(props, "fps", 30)),
            "resolution_scale": int(
                getattr(props, "render_resolution_scale", "100")
            ),
        }

        # Build full scenario document
        scenario = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "environment": environment_data,
            "scene_items": scene_items_data,
            "render_settings": render_settings,
        }

        scenario_path = os.path.join(self.current_dir, "scenario.json")
        with open(scenario_path, "w", encoding="utf-8") as f:
            json.dump(scenario, f, indent=2, ensure_ascii=False)

        print(f"[ExperimentManager] Scenario saved: {scenario_path}")
        return scenario_path

    def load_scenario(self, experiment_dir: str, scene, props):
        """
        Loads scenario.json and restores all scene items and settings.

        Args:
            experiment_dir: Path to the experiment directory containing scenario.json
            scene: bpy.context.scene — the Blender scene object
            props: scene.adas_props — the ADAS property group

        Raises:
            FileNotFoundError: If scenario.json doesn't exist in the directory.
        """
        self.current_dir = os.path.abspath(experiment_dir)
        scenario_path = os.path.join(self.current_dir, "scenario.json")

        if not os.path.isfile(scenario_path):
            raise FileNotFoundError(
                f"No scenario.json found in: {self.current_dir}"
            )

        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario = json.load(f)

        # Restore environment settings
        env = scenario.get("environment", {})
        if hasattr(props, "depth_strength") and "depth_strength" in env:
            props.depth_strength = float(env["depth_strength"])
        if hasattr(props, "grid_resolution") and "grid_resolution" in env:
            props.grid_resolution = int(env["grid_resolution"])
        # Restore cam_pitch (check both property names)
        cam_pitch_val = env.get("cam_pitch")
        if cam_pitch_val is not None:
            if hasattr(props, "roadcam_pitch"):
                props.roadcam_pitch = float(cam_pitch_val)
            elif hasattr(props, "cam_pitch"):
                props.cam_pitch = float(cam_pitch_val)

        # Restore render settings
        render = scenario.get("render_settings", {})
        if "engine" in render and hasattr(props, "render_engine"):
            try:
                props.render_engine = render["engine"]
            except TypeError:
                pass  # Enum value might not match available options
        if "total_frames" in render and hasattr(props, "total_frames"):
            props.total_frames = int(render["total_frames"])
        if "fps" in render and hasattr(props, "fps"):
            props.fps = int(render["fps"])
        if "resolution_scale" in render and hasattr(props, "render_resolution_scale"):
            try:
                props.render_resolution_scale = str(render["resolution_scale"])
            except TypeError:
                pass

        # Restore scene items
        props.scene_items.clear()
        for item_data in scenario.get("scene_items", []):
            item = props.scene_items.add()
            item.name = item_data.get("name", "Object")

            # Set obj_type first (may trigger update callbacks)
            obj_type = item_data.get("obj_type", "OBSTACLE")
            try:
                item.obj_type = obj_type
            except TypeError:
                item.obj_type = "OBSTACLE"

            # Set subtype if property exists
            subtype = item_data.get("subtype", "")
            if subtype and hasattr(item, "subtype"):
                try:
                    item.subtype = subtype
                except TypeError:
                    pass

            item.pos_x = float(item_data.get("pos_x", 0.0))
            item.pos_y = float(item_data.get("pos_y", 0.0))
            item.rotation_z = float(item_data.get("rotation_z", 0.0))
            item.scale_factor = float(item_data.get("scale_factor", 1.0))
            item.animated = bool(item_data.get("animated", False))
            item.end_x = float(item_data.get("end_x", 0.0))
            item.end_y = float(item_data.get("end_y", 5.0))
            item.speed_kmh = float(item_data.get("speed_kmh", 5.0))

            behavior = item_data.get("scenario_behavior", "STATIC")
            if hasattr(item, "scenario_behavior"):
                try:
                    item.scenario_behavior = behavior
                except TypeError:
                    pass

            custom_path = item_data.get("custom_model_path", "")
            if custom_path and hasattr(item, "custom_model_path"):
                item.custom_model_path = custom_path

        props.scene_items_index = 0 if len(props.scene_items) > 0 else -1

        print(
            f"[ExperimentManager] Scenario loaded from: {scenario_path} "
            f"({len(props.scene_items)} items)"
        )

    def list_experiments(self) -> list:
        """
        Returns list of (dir_name, scenario_json_path, timestamp)
        for all experiments found in the experiments/ directory.

        Experiments are sorted by directory name (i.e., by number).
        """
        experiments = []
        exp_root = self.paths.experiments_dir

        if not os.path.isdir(exp_root):
            return experiments

        for entry in sorted(os.listdir(exp_root)):
            entry_path = os.path.join(exp_root, entry)
            if not os.path.isdir(entry_path):
                continue

            scenario_path = os.path.join(entry_path, "scenario.json")
            timestamp = ""
            if os.path.isfile(scenario_path):
                try:
                    with open(scenario_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    timestamp = data.get("timestamp", "")
                except (json.JSONDecodeError, OSError):
                    pass

            experiments.append((entry, scenario_path, timestamp))

        return experiments

    @property
    def renders_dir(self) -> str:
        """<experiment>/renders/"""
        if not self.current_dir:
            raise RuntimeError("No experiment directory set.")
        return os.path.join(self.current_dir, "renders")

    @property
    def dataset_dir(self) -> str:
        """<experiment>/yolo_dataset/"""
        if not self.current_dir:
            raise RuntimeError("No experiment directory set.")
        return os.path.join(self.current_dir, "yolo_dataset")

    @property
    def qa_dir(self) -> str:
        """<experiment>/qa_preview/"""
        if not self.current_dir:
            raise RuntimeError("No experiment directory set.")
        return os.path.join(self.current_dir, "qa_preview")
