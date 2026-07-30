import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("blender") is None, reason="blender not on PATH"
)


def test_render_5_frames_produces_labels(tmp_path):
    out_root = tmp_path / "ds"
    cmd = [
        sys.executable, "data_synth/render_dataset.py",
        "--config", "config.yaml",
        "--out", str(out_root),
        "--num-train", "5",
        "--num-val", "2",
    ]
    subprocess.run(cmd, check=True)
    train_images = list((out_root / "images" / "train").glob("*.png"))
    train_labels = list((out_root / "labels" / "train").glob("*.txt"))
    assert len(train_images) == 5
    assert len(train_labels) == 5

    # Multi-class, multi-instance dataset: each label file holds 1+ lines (one per
    # visible object), except background frames which are empty. Every non-empty
    # line is `cls cx cy bw bh + 12*(x,y,vis)` = 41 tokens, cls in {0,1}.
    saw_line = False
    for lbl in train_labels:
        for line in lbl.read_text().strip().splitlines():
            toks = line.split()
            if not toks:
                continue
            saw_line = True
            assert len(toks) == 41, toks
            cls = int(toks[0])
            assert cls in {0, 1}, cls
            for v in toks[1:5]:
                assert 0.0 <= float(v) <= 1.0
            for i in range(12):
                vis = toks[5 + i * 3 + 2]
                assert vis in {"0", "1", "2"}, vis
    assert saw_line, "no labeled instances produced"
    assert (out_root / "data.yaml").exists()
