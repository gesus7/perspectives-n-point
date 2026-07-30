"""Run via the project venv (not Blender's). Spawns blender headless processes."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.camera import load_camera
from utils.config import load_config


def _blender_executable() -> str:
    exe = shutil.which("blender")
    if exe:
        return exe
    raise RuntimeError(
        "`blender` not found on PATH. Install Blender 3.6+/4.x and ensure "
        "`blender --version` works in your shell."
    )


def _run_split(blender_exe: str, cfg: dict, K, split: str, count: int, out_root: Path):
    images_dir = (out_root / "images" / split).resolve()
    labels_dir = (out_root / "labels" / split).resolve()
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    render = cfg["render"]
    model = cfg["model"]
    hdri = render.get("hdri_strength", [0.3, 1.0])
    scene_weights = render.get("scene_weights", [0.25, 0.25, 0.5])
    top_dist = render.get("top_camera_distance", [8.0, 20.0])
    cmd = [
        blender_exe, "--background", "--python",
        str(Path("data_synth/blender_render.py").resolve()), "--",
        "--obj", str(Path(model["obj"]).resolve()),
        "--keypoints", str(Path(model["keypoints_3d"]).resolve()),
        "--obj-top", str(Path(model["obj_top"]).resolve()),
        "--keypoints-top", str(Path(model["keypoints_3d_top"]).resolve()),
        "--scene-weights", ",".join(str(x) for x in scene_weights),
        "--top-dist-min", str(top_dist[0]),
        "--top-dist-max", str(top_dist[1]),
        "--top-tilt-jitter-deg", str(render.get("top_tilt_jitter_deg", 8.0)),
        "--out-images", str(images_dir),
        "--out-labels", str(labels_dir),
        "--count", str(count),
        "--width", str(render["image_size"][0]),
        "--height", str(render["image_size"][1]),
        "--fx", f"{float(K[0, 0]):.6f}",
        "--sensor-width-mm", str(render["sensor_width_mm"]),
        "--dist-min", str(render["camera_distance"][0]),
        "--dist-max", str(render["camera_distance"][1]),
        "--pitch-min-deg", str(render["pitch_deg"][0]),
        "--pitch-max-deg", str(render["pitch_deg"][1]),
        "--roll-jitter-deg", str(render["roll_jitter_deg"]),
        "--sun-min", str(render["sun_strength"][0]),
        "--sun-max", str(render["sun_strength"][1]),
        "--hdri-min", str(hdri[0]),
        "--hdri-max", str(hdri[1]),
        "--ground-size", str(render["ground_size"]),
        "--backgrounds-dir", str(Path(render["backgrounds_dir"]).resolve()) if render["backgrounds_dir"] else "",
        "--textures-dir", str(Path(render["textures_dir"]).resolve()) if render["textures_dir"] else "",
        "--occlusion-eps-m", str(render["occlusion_eps_m"]),
        "--min-visible", str(render["min_visible_keypoints"]),
        "--min-bbox-ratio", str(render["min_bbox_area_ratio"]),
        "--seed", "0" if split == "train" else "1",
        "--start-index", "0",
    ]
    print(f"[orchestrator] launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def write_data_yaml(out_root: Path, num_kpts: int, names: dict):
    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
        "kpt_shape": [num_kpts, 3],
        "flip_idx": [],
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"[orchestrator] wrote {out_root / 'data.yaml'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="dataset")
    p.add_argument("--num-train", type=int, default=None)
    p.add_argument("--num-val", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    K, _ = load_camera(cfg["camera"]["path"])

    expected_w = cfg["render"]["image_size"][0]
    expected_h = cfg["render"]["image_size"][1]
    if abs(K[0, 2] - expected_w / 2) > 1.0 or abs(K[1, 2] - expected_h / 2) > 1.0:
        raise SystemExit(
            f"camera.yaml principal point ({K[0,2]}, {K[1,2]}) does not match "
            f"image center ({expected_w/2}, {expected_h/2}). Update config.render.image_size "
            f"or your camera.yaml."
        )

    n_train = cfg["render"]["num_train"] if args.num_train is None else args.num_train
    n_val = cfg["render"]["num_val"] if args.num_val is None else args.num_val

    out_root = Path(args.out)
    blender = _blender_executable()
    _run_split(blender, cfg, K, "train", n_train, out_root)
    _run_split(blender, cfg, K, "val", n_val, out_root)
    names = cfg["model"].get("classes", {0: "TowerBase", 1: "TowerTop"})
    write_data_yaml(out_root, cfg["model"]["num_keypoints"], names)


if __name__ == "__main__":
    main()
