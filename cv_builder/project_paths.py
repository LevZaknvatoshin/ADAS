"""
Centralized Path Management for ADAS CV Builder
=================================================
Provides a single source of truth for all project directories:
  - assets/      — 3D models (.glb), textures
  - environments/ — cached depth maps, masks per source photo
  - experiments/  — individual test runs with renders and datasets

Auto-detects project root from Blender context or module location.
"""

import os
import re
import shutil
from pathlib import Path


class ProjectPaths:
    """Centralized path management for ADAS CV Builder project."""

    def __init__(self, project_root=None):
        """
        Auto-detect project root:
          1. If project_root is given, use it
          2. Try directory of current .blend file (bpy.data.filepath)
          3. Try parent of cv_builder module directory
          4. Fallback to user's Desktop
        """
        if project_root:
            self._root = os.path.abspath(project_root)
        else:
            self._root = self._auto_detect_root()
        self.ensure_dirs()

    def _auto_detect_root(self) -> str:
        """Attempts to locate the project root automatically."""
        # Strategy 2: Blender file directory
        try:
            import bpy
            blend_path = bpy.data.filepath
            if blend_path:
                blend_dir = os.path.dirname(os.path.abspath(blend_path))
                # Check if cv_builder/ exists relative to blend file dir
                if os.path.isdir(os.path.join(blend_dir, "cv_builder")):
                    return blend_dir
                # Also accept if .blend is inside the project tree
                parent = os.path.dirname(blend_dir)
                if os.path.isdir(os.path.join(parent, "cv_builder")):
                    return parent
        except (ImportError, Exception):
            pass

        # Strategy 3: Parent of cv_builder module directory
        module_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(module_dir)
        if os.path.basename(module_dir) == "cv_builder":
            return parent_dir

        # Strategy 4: Fallback to user's Desktop
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        fallback = os.path.join(desktop, "CV_Builder_Project")
        if not os.path.exists(fallback):
            os.makedirs(fallback, exist_ok=True)
        return fallback

    @property
    def root(self) -> str:
        """Project root directory."""
        return self._root

    @property
    def assets_dir(self) -> str:
        """<project>/assets/ — 3D models, textures."""
        return os.path.join(self._root, "assets")

    @property
    def vehicles_dir(self) -> str:
        """<project>/assets/vehicles/ — vehicle models."""
        return os.path.join(self.assets_dir, "vehicles")

    @property
    def pedestrians_dir(self) -> str:
        """<project>/assets/pedestrians/ — pedestrian models."""
        return os.path.join(self.assets_dir, "pedestrians")

    @property
    def obstacles_dir(self) -> str:
        """<project>/assets/obstacles/ — obstacle models."""
        return os.path.join(self.assets_dir, "obstacles")

    @property
    def environments_dir(self) -> str:
        """<project>/environments/ — cached depth maps, masks per photo."""
        return os.path.join(self._root, "environments")

    @property
    def experiments_dir(self) -> str:
        """<project>/experiments/ — individual test runs."""
        return os.path.join(self._root, "experiments")

    def get_environment_dir(self, image_path: str) -> str:
        """
        Returns environments/<stem>/ for a given source image.
        Creates the directory if it doesn't exist.
        Copies source image into it as 'source.<ext>' if not already there.
        """
        image_path = os.path.abspath(image_path)
        stem = Path(image_path).stem
        ext = Path(image_path).suffix or ".jpg"
        env_dir = os.path.join(self.environments_dir, stem)
        os.makedirs(env_dir, exist_ok=True)

        # Copy source image into environment dir if not present
        source_dest = os.path.join(env_dir, f"source{ext}")
        if not os.path.exists(source_dest) and os.path.isfile(image_path):
            try:
                shutil.copy2(image_path, source_dest)
            except (OSError, shutil.SameFileError):
                pass

        return env_dir

    def get_next_experiment_dir(self, name_hint: str = "") -> str:
        """
        Returns experiments/exp_NNN_<name>/ with auto-incrementing number.
        Creates the directory.
        """
        os.makedirs(self.experiments_dir, exist_ok=True)

        # Scan existing exp_NNN_* folders
        max_num = 0
        pattern = re.compile(r"^exp_(\d{3,})")
        if os.path.isdir(self.experiments_dir):
            for entry in os.listdir(self.experiments_dir):
                match = pattern.match(entry)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num

        next_num = max_num + 1
        # Sanitize name hint: keep only alphanumeric, underscore, hyphen
        safe_name = re.sub(r"[^\w\-]", "_", name_hint).strip("_") if name_hint else ""
        if safe_name:
            dir_name = f"exp_{next_num:03d}_{safe_name}"
        else:
            dir_name = f"exp_{next_num:03d}"

        exp_dir = os.path.join(self.experiments_dir, dir_name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def find_model_file(self, filename: str) -> str | None:
        """
        Searches for a model file in assets/ subdirectories and legacy locations.

        Search order:
          1. assets/vehicles/
          2. assets/pedestrians/
          3. assets/obstacles/
          4. assets/ (root)
          5. cv_builder/models/ (legacy)
          6. Модели/ (legacy Russian-named folder)
        """
        candidates = [
            os.path.join(self.vehicles_dir, filename),
            os.path.join(self.pedestrians_dir, filename),
            os.path.join(self.obstacles_dir, filename),
            os.path.join(self.assets_dir, filename),
            os.path.join(self._root, "cv_builder", "models", filename),
            os.path.join(self._root, "Модели", filename),
        ]

        # Also search from the module directory (for addon installed in Blender)
        module_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(module_dir, "models", filename))

        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
        return None

    def ensure_dirs(self):
        """Creates all top-level directories if they don't exist."""
        dirs = [
            self.assets_dir,
            self.vehicles_dir,
            self.pedestrians_dir,
            self.obstacles_dir,
            self.environments_dir,
            self.experiments_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def __repr__(self) -> str:
        return f"ProjectPaths(root='{self._root}')"


# Module-level singleton (lazy-initialized)
_default_paths = None


def get_project_paths(project_root=None) -> ProjectPaths:
    """
    Returns a module-level ProjectPaths singleton.
    If project_root is provided on first call, uses that root.
    Subsequent calls return the cached instance.
    """
    global _default_paths
    if _default_paths is None or project_root is not None:
        _default_paths = ProjectPaths(project_root=project_root)
    return _default_paths
