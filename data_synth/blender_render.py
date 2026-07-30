"""Run inside Blender via:
    blender --background --python data_synth/blender_render.py -- <args>

Renders `count` images containing the transmission-tower BASE segment
(class 0, ground-fixed) and/or the TOP segment (class 1, hanging in the air
below the helicopter). Each frame randomly contains base-only, top-only or both
(weighted by --scene-weights). When the top is present it is kept within
--top-dist-max metres of the camera. Writes multi-instance YOLO-pose labels (one
line per visible object: `cls cx cy bw bh + 12*(x,y,vis)`) and PNG images.
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector, Euler, Matrix


# A dedicated RNG for background ordering, independent of the global `random`
# stream that drives camera/lighting jitter. Seeded separately so the background
# order can be re-shuffled every run (fresh entropy) while scene perturbation
# stays reproducible from --seed.
_bg_rng = random.Random()


class BackgroundDeck:
    """Draw backgrounds WITHOUT replacement: every image is used once before any
    repeats, giving even coverage. The deck reshuffles when exhausted, and the
    shuffle order is controlled by `rng` so each run can differ."""

    def __init__(self, candidates, rng):
        self._candidates = list(candidates)
        self._rng = rng
        self._deck = []

    def __bool__(self):
        return bool(self._candidates)

    def draw(self):
        if not self._candidates:
            return None
        if not self._deck:
            self._deck = list(self._candidates)
            self._rng.shuffle(self._deck)
        return self._deck.pop()


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--obj", required=True, help="base segment OBJ (class 0)")
    p.add_argument("--keypoints", required=True, help="base keypoints JSON")
    p.add_argument("--obj-top", required=True, help="top segment OBJ (class 1)")
    p.add_argument("--keypoints-top", required=True, help="top keypoints JSON")
    p.add_argument("--out-images", required=True)
    p.add_argument("--out-labels", required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fx", type=float, required=True)
    p.add_argument("--sensor-width-mm", type=float, default=36.0)
    p.add_argument("--dist-min", type=float, default=20.0,
                   help="base orbit min distance (m)")
    p.add_argument("--dist-max", type=float, default=80.0,
                   help="base orbit max distance (m)")
    p.add_argument("--top-dist-min", type=float, default=8.0,
                   help="camera->top min distance (m); top stays this close")
    p.add_argument("--top-dist-max", type=float, default=20.0,
                   help="camera->top max distance (m); hard cap on top range")
    p.add_argument("--top-tilt-jitter-deg", type=float, default=8.0,
                   help="rope-sway tilt of the floating top segment")
    p.add_argument("--scene-weights", default="0.25,0.25,0.5",
                   help="weights for base_only,top_only,both")
    p.add_argument("--pitch-min-deg", type=float, default=60.0)
    p.add_argument("--pitch-max-deg", type=float, default=90.0)
    p.add_argument("--roll-jitter-deg", type=float, default=5.0)
    p.add_argument("--sun-min", type=float, default=2.0)
    p.add_argument("--sun-max", type=float, default=6.0)
    p.add_argument("--hdri-min", type=float, default=0.3,
                   help="min ambient/sky (world) fill strength")
    p.add_argument("--hdri-max", type=float, default=1.0,
                   help="max ambient/sky (world) fill strength")
    p.add_argument("--ground-size", type=float, default=200.0)
    p.add_argument("--backgrounds-dir", default="")
    p.add_argument("--textures-dir", default="")
    p.add_argument("--occlusion-eps-m", type=float, default=0.01)
    p.add_argument("--min-visible", type=int, default=4)
    p.add_argument("--min-bbox-ratio", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0,
                   help="seed for camera/lighting perturbation (reproducible scene jitter)")
    p.add_argument("--bg-seed", type=int, default=-1,
                   help="seed for background-order shuffle; <0 = fresh OS entropy each run "
                        "(different background order every run)")
    p.add_argument("--start-index", type=int, default=0)
    return p.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    engine_items = [
        i.identifier
        for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    ]
    if "BLENDER_EEVEE_NEXT" in engine_items:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in engine_items:
        bpy.context.scene.render.engine = "BLENDER_EEVEE"


def load_obj(path: str, name: str):
    """Import an OBJ, join its meshes into one object, and rename it. The object
    is NOT moved/rotated here — placement is per-frame (see place_* helpers)."""
    existing = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        bpy.ops.import_scene.obj(filepath=path)
    new = [o for o in bpy.context.scene.objects if o not in existing]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh imported from OBJ: {path}")
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    return obj


def bbox_center_world(obj) -> Vector:
    """World-space centre of an object's local bounding box (uses matrix_world)."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return sum(corners, Vector((0, 0, 0))) / 8.0


def bbox_radius(obj) -> float:
    """Half the bounding-box diagonal in world units (rotation-invariant for a
    rigid body): the max distance from the bbox centre to any mesh corner. Used
    to keep the WHOLE top segment within --top-dist-max of the camera, not just
    its centre."""
    center = bbox_center_world(obj)
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return max((c - center).length for c in corners)


def set_object_active(obj, active: bool):
    """Show/hide an object for BOTH render and the depsgraph (ray_cast). Hiding
    in the viewport too keeps an inactive object from occluding keypoints."""
    obj.hide_render = not active
    obj.hide_viewport = not active


def place_top(top, center_world: Vector, tilt_jitter_deg: float):
    """Orient the top segment upright (its native +Y axis -> world +Z, apex up),
    add a random yaw about Z and a small rope-sway tilt, then translate so its
    bounding-box centre lands at `center_world`."""
    upright = Matrix.Rotation(math.radians(90.0), 4, "X")   # native +Y -> world +Z
    yaw = Matrix.Rotation(random.uniform(0, 2 * math.pi), 4, "Z")
    tx = math.radians(random.uniform(-tilt_jitter_deg, tilt_jitter_deg))
    ty = math.radians(random.uniform(-tilt_jitter_deg, tilt_jitter_deg))
    sway = Matrix.Rotation(tx, 4, "X") @ Matrix.Rotation(ty, 4, "Y")
    total = sway @ yaw @ upright
    top.rotation_euler = total.to_euler()
    top.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    cur = bbox_center_world(top)
    top.location = center_world - cur
    bpy.context.view_layer.update()


def add_ground(size: float, image_dir: str, texture_dir: str):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, -9.6))
    plane = bpy.context.active_object
    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    candidates = []
    for d in (image_dir, texture_dir):
        if d and Path(d).is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                candidates.extend(sorted(Path(d).glob(ext)))

    # Backgrounds are drawn without replacement from a shuffled deck (see
    # BackgroundDeck) for even coverage and a fresh order each run.
    deck = BackgroundDeck(candidates, _bg_rng)
    plane.data.materials.append(mat)
    _randomize_ground(mat, deck)
    return plane, mat, deck


def _randomize_ground(mat, deck):
    """Pick a new background texture (or color) for the ground plane, with
    randomized UV mapping (scale/rotation/offset) and roughness so the same
    texture never looks identical twice."""
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes["Principled BSDF"]

    # Remove any existing texture / mapping nodes
    for node in list(nodes):
        if node.type in ("TEX_IMAGE", "MAPPING", "TEX_COORD"):
            nodes.remove(node)

    # Randomized micro-surface so lighting response varies per frame.
    bsdf.inputs["Roughness"].default_value = random.uniform(0.35, 0.95)

    img_path = deck.draw()
    if img_path is not None:
        img_path = str(img_path)
        img_name = Path(img_path).name
        if img_name in bpy.data.images:
            tex_img = bpy.data.images[img_name]
        else:
            tex_img = bpy.data.images.load(img_path)
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = tex_img
        tex_node.extension = "REPEAT"

        # Randomized UV mapping: vary scale, planar rotation and offset so the
        # texture's framing/orientation differs every render.
        coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        scale = random.uniform(0.5, 3.0)
        mapping.inputs["Location"].default_value = (
            random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), 0.0)
        mapping.inputs["Rotation"].default_value = (
            0.0, 0.0, random.uniform(0.0, 2 * math.pi))
        mapping.inputs["Scale"].default_value = (scale, scale, 1.0)
        links.new(coord.outputs["UV"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], tex_node.inputs["Vector"])
        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (
            random.uniform(0.2, 0.6),
            random.uniform(0.3, 0.6),
            random.uniform(0.2, 0.5), 1.0,
        )


def add_sun(strength: float):
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 100))
    sun = bpy.context.active_object
    sun.data.energy = strength
    _randomize_sun(sun, strength)
    return sun


def _randomize_sun(sun, strength: float):
    """Randomize sun direction, energy and a warm/cool color tint per frame."""
    sun.data.energy = strength
    sun.rotation_euler = (
        math.radians(random.uniform(-30, 30)),
        math.radians(random.uniform(-30, 30)),
        math.radians(random.uniform(0, 360)),
    )
    # Warm (sunset) <-> cool (overcast) tint, kept subtle to stay realistic.
    warmth = random.uniform(-1.0, 1.0)
    r = 1.0 + 0.15 * max(warmth, 0.0)
    b = 1.0 + 0.15 * max(-warmth, 0.0)
    g = 1.0
    sun.data.color = (min(r, 1.0), g, min(b, 1.0)) if warmth >= 0 else (r, g, b)


def set_world_ambient(strength: float):
    """Set uniform sky/ambient fill with a slight random tint. Provides the
    soft fill that real outdoor scenes get from the sky dome, and softens the
    hard sun shadows so keypoints in shadow remain visible."""
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is None:
        bg = world.node_tree.nodes.new("ShaderNodeBackground")
    tint = random.uniform(-0.08, 0.08)
    bg.inputs["Color"].default_value = (
        max(0.0, 0.5 + tint), 0.5, max(0.0, 0.5 - tint), 1.0)
    bg.inputs["Strength"].default_value = strength


def setup_camera(width: int, height: int, fx_px: float, sensor_width_mm: float):
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.lens_unit = "MILLIMETERS"
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = sensor_width_mm
    cam.data.lens = fx_px * sensor_width_mm / width
    cam.data.shift_x = 0.0
    cam.data.shift_y = 0.0
    bpy.context.scene.camera = cam
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "PNG"
    return cam


def aim_camera_at(cam, target: Vector, roll_jitter_deg: float):
    """Point the camera at `target` (from its current location) with roll jitter."""
    direction = (target - cam.location).normalized()
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()
    roll = math.radians(random.uniform(-roll_jitter_deg, roll_jitter_deg))
    eul = cam.rotation_euler
    cam.rotation_euler = Euler((eul.x, eul.y, eul.z + roll), "XYZ")


def sample_camera_pose(cam, target: Vector, dist_min, dist_max,
                       pitch_min_deg, pitch_max_deg, roll_jitter_deg):
    """Orbit the camera around `target` at a random distance/pitch/yaw and aim it."""
    d = random.uniform(dist_min, dist_max)
    pitch = math.radians(random.uniform(pitch_min_deg, pitch_max_deg))
    yaw = math.radians(random.uniform(0, 360))
    x = d * math.cos(pitch) * math.cos(yaw)
    y = d * math.cos(pitch) * math.sin(yaw)
    z = d * math.sin(pitch)
    cam.location = target + Vector((x, y, z))
    aim_camera_at(cam, target, roll_jitter_deg)


def project_world_to_pixel(world_co: Vector, cam, scene) -> tuple[float, float, bool]:
    from bpy_extras.object_utils import world_to_camera_view
    co_ndc = world_to_camera_view(scene, cam, world_co)
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    px = co_ndc.x * w
    py = (1.0 - co_ndc.y) * h
    in_img = (0.0 <= px < w) and (0.0 <= py < h) and (co_ndc.z > 0)
    return px, py, in_img


def keypoint_visibility(world_co: Vector, cam, scene, depsgraph, eps_m: float) -> int:
    px, py, in_img = project_world_to_pixel(world_co, cam, scene)
    if not in_img:
        return 0
    cam_loc = cam.matrix_world.translation
    direction = (world_co - cam_loc).normalized()
    hit, loc, _, _, _, _ = scene.ray_cast(depsgraph, cam_loc, direction)
    if not hit:
        return 2
    dist_hit = (loc - cam_loc).length
    dist_kp = (world_co - cam_loc).length
    if dist_hit + eps_m < dist_kp:
        return 1
    return 2


def load_keypoints(path: str) -> list[Vector]:
    with open(path, "r") as f:
        data = json.load(f)
    kpts = sorted(data["keypoints"], key=lambda k: k["id"])
    return [Vector((k["x"], k["y"], k["z"])) for k in kpts]


def annotate_object(scene, cam, depsgraph, obj, kpts_local, class_id, args) -> str | None:
    """Project one object's keypoints in the already-rendered scene and build a
    YOLO-pose label line (`cls cx cy bw bh + per-kpt x y vis`). Returns None if
    the object fails the min-visible / min-bbox-ratio gates.

    NOTE: keypoints keep FIXED physical ids (no position-based relabeling).
    Symmetric-object ambiguity is handled at training time by the symmetry-aware
    pose loss, which keeps the 2D<->3D correspondence intact so PnP stays solvable.
    """
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    obj_world_matrix = obj.matrix_world

    annotations = []
    for kp in kpts_local:
        world_co = obj_world_matrix @ kp
        v = keypoint_visibility(world_co, cam, scene, depsgraph, args.occlusion_eps_m)
        px, py, _ = project_world_to_pixel(world_co, cam, scene)
        annotations.append((px, py, v))

    visible = [a for a in annotations if a[2] >= 1]
    if len(visible) < args.min_visible:
        return None

    xs = [a[0] for a in visible]
    ys = [a[1] for a in visible]
    x_min, x_max = max(min(xs), 0), min(max(xs), w - 1)
    y_min, y_max = max(min(ys), 0), min(max(ys), h - 1)
    bw = x_max - x_min
    bh = y_max - y_min
    x_min = max(x_min - 0.05 * bw, 0)
    x_max = min(x_max + 0.05 * bw, w - 1)
    y_min = max(y_min - 0.05 * bh, 0)
    y_max = min(y_max + 0.05 * bh, h - 1)
    bw = x_max - x_min
    bh = y_max - y_min
    if bw * bh < args.min_bbox_ratio * w * h:
        return None

    cx = (x_min + x_max) / 2.0 / w
    cy = (y_min + y_max) / 2.0 / h
    nw = bw / w
    nh = bh / h
    parts = [f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]
    for (px, py, v) in annotations:
        if v == 0:
            parts.append(f"0.000000 0.000000 0")
        else:
            parts.append(f"{px / w:.6f} {py / h:.6f} {v}")
    return " ".join(parts)


def place_scene(scene_type, base, top, cam, base_target, args):
    """Position objects + camera for one frame according to `scene_type`
    (base_only / top_only / both). Returns the list of (obj, kpts, class_id)
    that are active and should be annotated."""
    if scene_type == "base_only":
        set_object_active(base, True)
        set_object_active(top, False)
        sample_camera_pose(cam, base_target, args.dist_min, args.dist_max,
                           args.pitch_min_deg, args.pitch_max_deg, args.roll_jitter_deg)
        return [("base",)]

    if scene_type == "top_only":
        set_object_active(base, False)
        set_object_active(top, True)
        # Float the top above the ground; camera orbits it at the near range.
        top_center = Vector((0.0, 0.0, random.uniform(-2.0, 6.0)))
        place_top(top, top_center, args.top_tilt_jitter_deg)
        # Pull the orbit in by the top's radius so its farthest point (not just
        # its centre) stays within top_dist_max of the camera.
        radius = bbox_radius(top)
        d_max = max(args.top_dist_min, args.top_dist_max - radius)
        sample_camera_pose(cam, top_center, args.top_dist_min, d_max,
                           args.pitch_min_deg, args.pitch_max_deg, args.roll_jitter_deg)
        return [("top",)]

    # both: base fixed on ground, top floating on the camera->base view ray so it
    # stays within [top_dist_min, top_dist_max] of the camera and hangs in front
    # of the base (helicopter looking down past the suspended top to the base).
    set_object_active(base, True)
    set_object_active(top, True)
    sample_camera_pose(cam, base_target, args.dist_min, args.dist_max,
                       args.pitch_min_deg, args.pitch_max_deg, args.roll_jitter_deg)
    cam_loc = cam.location.copy()
    view_dir = (base_target - cam_loc).normalized()
    place_top(top, cam_loc + view_dir * args.top_dist_min, args.top_tilt_jitter_deg)
    radius = bbox_radius(top)
    # Build a basis perpendicular to the view ray for a small lateral offset so
    # the top doesn't perfectly occlude the base.
    up = Vector((0.0, 0.0, 1.0))
    right = view_dir.cross(up)
    if right.length < 1e-6:
        right = Vector((1.0, 0.0, 0.0))
    right.normalize()
    up_perp = right.cross(view_dir).normalized()
    lateral = right * random.uniform(-2.0, 2.0) + up_perp * random.uniform(-2.0, 2.0)
    # Keep the WHOLE top within top_dist_max of the camera: the centre distance is
    # sqrt(along^2 + |lateral|^2), so reserve the radius AND the lateral component
    # when choosing the along-ray distance.
    budget = max(args.top_dist_min, args.top_dist_max - radius)
    along_max = math.sqrt(max(args.top_dist_min ** 2, budget ** 2 - lateral.length_squared))
    along = random.uniform(args.top_dist_min, max(args.top_dist_min, along_max))
    top_center = cam_loc + view_dir * along + lateral
    place_top(top, top_center, args.top_tilt_jitter_deg)
    return [("base",), ("top",)]


def main():
    args = parse_args()
    random.seed(args.seed)
    # Background order: fresh OS entropy by default (different every run), or a
    # fixed seed when reproducibility is wanted.
    if args.bg_seed < 0:
        _bg_rng.seed()
        print("[render] background order: fresh entropy (new shuffle this run)")
    else:
        _bg_rng.seed(args.bg_seed)
        print(f"[render] background order: seeded ({args.bg_seed})")

    weights = [float(x) for x in args.scene_weights.split(",")]
    if len(weights) != 3 or sum(weights) <= 0:
        raise SystemExit("--scene-weights must be 3 positive-ish numbers: base_only,top_only,both")
    scene_types = ["base_only", "top_only", "both"]

    reset_scene()
    bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True

    base = load_obj(args.obj, "TowerBase")
    top = load_obj(args.obj_top, "TowerTop")
    _, ground_mat, ground_deck = add_ground(
        args.ground_size, args.backgrounds_dir, args.textures_dir)
    sun_strength = random.uniform(args.sun_min, args.sun_max)
    add_sun(sun_strength)
    set_world_ambient(random.uniform(args.hdri_min, args.hdri_max))
    cam = setup_camera(args.width, args.height, args.fx, args.sensor_width_mm)

    kpts_base = load_keypoints(args.keypoints)
    kpts_top = load_keypoints(args.keypoints_top)
    obj_lookup = {
        "base": (base, kpts_base, 0),
        "top": (top, kpts_top, 1),
    }
    base_target_world = base.matrix_world @ Vector((0, 0, 1.0))

    Path(args.out_images).mkdir(parents=True, exist_ok=True)
    Path(args.out_labels).mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    sun = next(o for o in bpy.context.scene.objects if o.type == "LIGHT")

    i = args.start_index
    produced = 0
    attempts = 0
    max_attempts = args.count * 5
    while produced < args.count and attempts < max_attempts:
        attempts += 1
        out_img = str(Path(args.out_images) / f"img_{i:06d}.png")

        _randomize_ground(ground_mat, ground_deck)
        _randomize_sun(sun, random.uniform(args.sun_min, args.sun_max))
        set_world_ambient(random.uniform(args.hdri_min, args.hdri_max))

        # Every 50th image (global index): background-only frame, no object.
        if i % 50 == 0:
            set_object_active(base, False)
            set_object_active(top, False)
            bpy.context.view_layer.update()
            scene.render.filepath = out_img
            bpy.ops.render.render(write_still=True)
            open(Path(args.out_labels) / f"img_{i:06d}.txt", "w").close()  # empty label
            produced += 1
            i += 1
            print(f"[render] background-only frame idx={i - 1}  produced {produced}/{args.count}")
            continue

        scene_type = random.choices(scene_types, weights=weights, k=1)[0]
        active = place_scene(scene_type, base, top, cam, base_target_world, args)
        bpy.context.view_layer.update()

        scene.render.filepath = out_img
        bpy.ops.render.render(write_still=True)

        # Re-fetch depsgraph AFTER objects moved so ray_cast occlusion is current.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        lines = []
        for (key,) in active:
            obj, kpts_local, class_id = obj_lookup[key]
            line = annotate_object(scene, cam, depsgraph, obj, kpts_local, class_id, args)
            if line is not None:
                lines.append(line)

        if not lines:
            try:
                os.remove(out_img)
            except OSError:
                pass
            continue

        with open(Path(args.out_labels) / f"img_{i:06d}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        produced += 1
        i += 1
        print(f"[render] produced {produced}/{args.count} (idx={i - 1}) "
              f"scene={scene_type} objs={len(lines)}")

    if produced < args.count:
        print(f"[render] WARNING: only produced {produced}/{args.count} after "
              f"{attempts} attempts.")


if __name__ == "__main__":
    main()
