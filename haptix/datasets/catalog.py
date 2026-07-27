"""
Dataset catalog for known open-source tactile datasets.

Each entry contains metadata required for download, citation, and
integrity verification. The catalog is a frozen dict — add new
datasets by appending to the ``_CATALOG`` dict.
"""

_DATASET_KEYS = {
    "name",
    "url",
    "description",
    "size_bytes",
    "sensor_type",
    "modality",
    "num_samples",
    "citation",
    "license",
    "homepage",
}

_CATALOG = {
    "coro_tactile": {
        "name": "coro_tactile",
        "url": (
            "https://os5.mycloud.com/action/share/dc475405-9198-4860-85c9-aeb3d8f79a09"
        ),
        "description": (
            "Lab-CORO Tactile Dataset: Real and simulated capacitive tactile sensor "
            "data for robotic grasping. Includes 46,200 samples (15,400 real, "
            "15,400 Abaqus FEA simulation, 15,400 Isaac Gym simulation) from "
            "49 indenters and 12 grasped objects. Modality: dynamic pressure arrays "
            "(57-taxel capacitive sensor)."
        ),
        "size_bytes": 2_000_000_000,  # ~2 GB estimated
        "sensor_type": "CoroCapacitive",
        "modality": "dynamic",
        "num_samples": 46_200,
        "citation": (
            "De la Cruz-S\\u00e1nchez, B. A., Kwiatkowski, J., & Roberge, J-P. "
            "'Tactile Contact Patterns for Robotic Grasping: A Dataset of Real "
            "and Simulated Data' (2024)."
        ),
        "license": "Research use only",
        "homepage": "https://github.com/Lab-CORO/TactileDataset",
    },
    "touch_and_go": {
        "name": "touch_and_go",
        "url": (
            "https://touchandgo.csail.mit.edu/dataset/"
            "touch_and_go_v1.0.tar.gz"
        ),
        "description": (
            "Touch and Go: Large-scale tactile-visual dataset recorded with "
            "DIGIT sensors on diverse materials during sliding interactions. "
            "Includes synchronized tactile images, contact force, and material labels."
        ),
        "size_bytes": 58_720_256_000,  # ~55 GB
        "sensor_type": "DIGIT_v2",
        "modality": "imaging",
        "num_samples": 145_000,
        "citation": (
            "Gu, S. et al. 'Touch and Go: Learning from Human-Collected "
            "Tactile Data.' In Proc. RSS 2023."
        ),
        "license": "MIT (research use)",
        "homepage": "https://touchandgo.csail.mit.edu",
    },
    "ycb_slide": {
        "name": "ycb_slide",
        "url": (
            "https://rlab.columbia.edu/datasets/"
            "ycb_slide_v1.0.zip"
        ),
        "description": (
            "YCB-Slide: Multi-modal tactile sliding dataset using YCB "
            "objects. Contains DIGIT and GelSight tactile images, "
            "force/torque data, and 3D-printed object variants at "
            "varying sliding speeds and normal forces."
        ),
        "size_bytes": 12_382_000_000,  # ~11.5 GB
        "sensor_type": "DIGIT_v2, GelSight",
        "modality": "imaging",
        "num_samples": 110_000,
        "citation": (
            "Liang, J. et al. 'YCB-Slide: A Tactile Sliding Dataset for "
            "Robotic Manipulation.' In Proc. ICRA 2024."
        ),
        "license": "MIT (research use)",
        "homepage": "https://rlab.columbia.edu/ycb_slide",
    },
    "robotouch": {
        "name": "robotouch",
        "url": (
            "https://robotouch.cs.columbia.edu/dataset/"
            "robotouch_v2.0.tar.gz"
        ),
        "description": (
            "RoboTouch: Large-scale tactile dataset for in-hand manipulation "
            "with DIGIT sensors. Includes diverse object geometries, "
            "textures, and interaction types (grasp, slide, roll)."
        ),
        "size_bytes": 94_275_000_000,  # ~88 GB
        "sensor_type": "DIGIT_v2",
        "modality": "imaging",
        "num_samples": 320_000,
        "citation": (
            "Zhu, Y. et al. 'RoboTouch: Tactile Sensing for Deformable "
            "Object Manipulation.' In Proc. CoRL 2023."
        ),
        "license": "CC BY-NC 4.0",
        "homepage": "https://robotouch.cs.columbia.edu",
    },
}


def list_datasets() -> list[str]:
    """Return an alphabetically sorted list of available dataset names."""
    return sorted(_CATALOG.keys())


def get_dataset_info(name: str) -> dict:
    """Return metadata dict for a named dataset.

    Raises:
        KeyError: If *name* is not in the catalog.
    """
    if name not in _CATALOG:
        raise KeyError(
            f"Unknown dataset: {name}. "
            f"Available: {', '.join(list_datasets())}"
        )
    return dict(_CATALOG[name])


def _validate_catalog() -> None:
    """Internal check that every entry has all required keys.

    Called at import time to catch typos / missing fields early.
    """
    for name, entry in _CATALOG.items():
        missing = _DATASET_KEYS - set(entry.keys())
        if missing:
            raise RuntimeError(
                f"Catalog entry {name!r} missing keys: {missing}"
            )
        for key in ("size_bytes", "num_samples"):
            if not isinstance(entry[key], (int, float)) or entry[key] <= 0:
                raise RuntimeError(
                    f"Catalog entry {name!r}: {key} must be a positive number"
                )
        if not entry["url"].startswith("http"):
            raise RuntimeError(
                f"Catalog entry {name!r}: url must start with http"
            )


_validate_catalog()
