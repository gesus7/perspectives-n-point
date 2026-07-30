import numpy as np

from infer.relative_pose import compute_relative_pose, flight_instruction


def test_relative_pose_identity_when_same():
    """Top at the same camera pose as base -> identity relative pose, aligned."""
    R = np.eye(3)
    t = np.array([1.0, 2.0, 30.0])
    rel = compute_relative_pose({"R": R, "t": t}, {"R": R, "t": t})
    assert np.allclose(rel["R_rel"], np.eye(3), atol=1e-9)
    assert np.allclose(rel["t_rel"], 0, atol=1e-9)
    assert rel["dist_m"] < 1e-9
    assert flight_instruction(rel["move_cam"])["en"] == "aligned"


def test_flight_instruction_camera_axes():
    """move_cam = t_base - t_top. Top must move toward base.

    Camera frame: +X right, +Y down-in-image(back), +Z deeper(down).
    """
    # Top is to the LEFT of base in camera X (base_x > top_x) -> helicopter moves +X = 向右.
    base = {"R": np.eye(3), "t": np.array([5.0, 0.0, 30.0])}
    top = {"R": np.eye(3), "t": np.array([0.0, 0.0, 30.0])}
    rel = compute_relative_pose(base, top)
    instr = flight_instruction(rel["move_cam"])
    assert "向右" in instr["zh"], instr
    assert "right" in instr["en"]

    # Top is FARTHER (larger Z) than base -> base_z < top_z -> move -Z = 向上 (climb).
    base = {"R": np.eye(3), "t": np.array([0.0, 0.0, 20.0])}
    top = {"R": np.eye(3), "t": np.array([0.0, 0.0, 30.0])}
    instr = flight_instruction(compute_relative_pose(base, top)["move_cam"])
    assert "向上" in instr["zh"], instr


def test_flight_instruction_deadzone():
    """Sub-deadzone offsets report aligned."""
    instr = flight_instruction(np.array([0.1, -0.2, 0.3]), deadzone_m=0.5)
    assert instr["zh"] == "已对齐"


def test_relative_pose_translation_in_base_frame():
    """With both orientations = identity, t_rel equals the camera-frame offset
    top - base (base frame == camera frame when R_base = I)."""
    base = {"R": np.eye(3), "t": np.array([1.0, 1.0, 25.0])}
    top = {"R": np.eye(3), "t": np.array([3.0, 1.0, 28.0])}
    rel = compute_relative_pose(base, top)
    assert np.allclose(rel["t_rel"], [2.0, 0.0, 3.0], atol=1e-9)
