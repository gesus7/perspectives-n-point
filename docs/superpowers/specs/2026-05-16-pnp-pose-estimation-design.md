# PnP 位姿估计端到端方案设计

**日期**：2026-05-16
**目标**：基于 YOLO26-pose 关键点检测 + OpenCV PnP 的输电塔位姿估计端到端系统。

## 1. 问题与场景

- **目标物**：直立在地面的输电塔（`Base.obj`，单位为米）。物体姿态固定。
- **相机**：内参固定（`camera.yaml`：fx=fy=2828.3, cx=960, cy=540, 1920×1080, 无畸变），从目标上方 20–80 m 俯拍。仅相机位置/朝向变化。
- **已有资产**：`Base.obj`、`Base.mtl`、`camera.yaml`、`keypoints_3d.json`（12 个关键点：4 底角 + 4 中部关节 + 4 上部关节，单位为米）。
- **最终输入**：视频流；中间验证阶段支持单图 / 图像文件夹。
- **输出**：每帧每个检测目标的旋转 R、平移 t（目标坐标系 → 相机坐标系），以及可视化叠加图。

## 2. 总体架构

模块化三阶段流水线，共享一份根目录 `config.yaml`：

```
data_synth (Blender headless) → dataset/ (YOLO-pose 格式) → train (ultralytics) → infer (PnP)
```

每个阶段独立 CLI 入口，可单独运行、单独调试、单独重跑。

### 2.1 目录结构

```
perspectives-n-point/
├── Base.obj, Base.mtl, camera.yaml, keypoints_3d.json   (已有)
├── config.yaml                            (全局参数)
├── backgrounds/                           (航拍背景图)
├── textures/                              (程序化纹理素材)
├── data_synth/
│   ├── render_dataset.py                  (调度入口：起 Blender headless)
│   └── blender_render.py                  (Blender Python 内执行的渲染脚本)
├── dataset/                               (渲染产物)
│   ├── images/{train,val}/*.png
│   ├── labels/{train,val}/*.txt
│   └── data.yaml                          (ultralytics 训练用)
├── train/
│   └── train.py
├── runs/                                  (ultralytics 默认输出，模型权重落这里)
├── infer/
│   ├── pose_solver.py                     (PnP 核心)
│   ├── visualize.py                       (画轴/重投影点/3D bbox)
│   ├── infer_image.py                     (单图/文件夹)
│   └── infer_video.py                     (视频文件/摄像头)
└── utils/
    ├── camera.py                          (camera.yaml 加载)
    └── keypoints.py                       (keypoints_3d.json 加载)
```

## 3. 模块设计

### 3.1 `utils/camera.py`

- `load_camera(path) -> (K: np.ndarray (3,3), dist: np.ndarray (5,), image_size: (w,h))`
- 单一职责：解析 YAML、返回 numpy 数组。下游所有模块（渲染、PnP）都通过它读 K。

### 3.2 `utils/keypoints.py`

- `load_keypoints_3d(path) -> (points: np.ndarray (N,3), names: list[str])`
- 按 id 升序排序，保证 PnP 输入与 YOLO 输出的关键点顺序一致。
- N=12 是数据集决定的，不在代码里硬编码。

### 3.3 `data_synth/render_dataset.py`（调度层）

- CLI 参数：`--config config.yaml --num-train 1000 --num-val 200 --out dataset/`。
- 职责：解析 config，调用 `blender --background --python data_synth/blender_render.py -- <args>` 起渲染。
- 不依赖 `bpy`，纯 Python，可以在普通解释器里跑。
- 渲染完成后：写出 `dataset/data.yaml`（ultralytics 训练用）。

### 3.4 `data_synth/blender_render.py`（在 Blender 内执行）

**相机内参对齐**（关键）：

- 固定 `sensor_width = 36mm`，由 `fx, image_width` 反推 `f_mm = fx × sensor_width / image_width ≈ 53.03mm`。
- 渲染分辨率严格设为 1920×1080。
- 主点严格 (cx=960, cy=540) → 不偏移（Blender 默认主点在图像中心，K 也是中心，匹配）。
- 校验：渲染一张测试图后，用代码生成的 K 与 `camera.yaml` 对比，逐元素差异 < 1e-3。

**场景搭建**：

- 加载 `Base.obj`，使其原点位于世界原点、姿态直立（Z 向上为 obj 坐标系约定，需检查 .obj 实际坐标轴并对齐到世界 Z）。
- 地面：一个大平面（200×200 m），位于塔基（z = -9.6m 附近，即关键点 0/1/2/3 的高度）。
- 背景：50% 概率使用 `backgrounds/` 中随机一张航拍图作为平面贴图；50% 概率使用程序化纹理（`textures/` 中的草地/土地/路面）。两类素材尺寸/光照差异较大可提升泛化。

**相机姿态采样**（仅相机动，目标固定）：

- 距离 d ∈ Uniform(20, 80) m（注：是相机到塔顶中心的距离）。
- 俯仰角 θ ∈ Uniform(60°, 90°)（90°=正下方俯拍，60°=斜视，覆盖典型无人机视角）。
- 方位角 φ ∈ Uniform(0°, 360°)。
- 相机 look_at 目标中心，up 向量加微小扰动 ±5°（roll 随机化）。

**光照**：

- 一个太阳光（强度随机 2–6，角度随机），加一个 HDRI 世界环境光（强度 0.3–1.0）。HDRI 可选用 Blender 自带或单独下载。

**标签输出**（关键）：

对每张渲染图：
1. 12 个关键点的 3D 世界坐标 → 用 Blender 相机矩阵投影到 2D 像素坐标。
2. **可见性 v** 按 ultralytics 标准三值：
   - `v=0`：投影点不在 [0, W)×[0, H) 内。
   - `v=1`：在图内但被几何遮挡。判定：`scene.ray_cast` 从相机原点向关键点世界坐标发射射线，命中点距离比关键点距离短 > 1cm 即被挡。
   - `v=2`：在图内且未被挡。
3. bbox：取 v≥1 的关键点像素坐标的轴对齐包围盒，再四边扩 5% 余量并裁剪到图像范围。
4. 写 `labels/<split>/<frame>.txt`，一行：
   ```
   0 cx cy w h x1 y1 v1 x2 y2 v2 ... x12 y12 v12
   ```
   所有 xy/wh 归一化到 [0,1]；v 保持整数 0/1/2。

**容错**：

- 若 v≥1 的关键点 < 4：直接丢弃此帧（PnP 无解）。
- 若 bbox 面积 < 图像面积 0.5%：丢弃（目标太小）。
- 调度层在丢弃后自动补采，直到达到目标数量。

### 3.5 `dataset/data.yaml`（ultralytics 训练配置）

```yaml
path: <abs_path>/dataset
train: images/train
val: images/val
names: {0: transmission_tower}
kpt_shape: [12, 3]   # 12 keypoints, (x, y, v)
flip_idx: []         # 无水平翻转对称（输电塔左右不对称结构无需指定）
```

> `flip_idx` 留空意味着训练时禁用水平翻转增强；ultralytics 在 `flip_idx=[]` 时会自动关闭 hflip 对关键点的影响。如果实测发现需要 hflip 增强，再回头根据 4 角点对称关系填写。

### 3.6 `train/train.py`

```python
from ultralytics import YOLO
model = YOLO('yolo26n-pose.pt')
model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=1280,
    batch=8,
    device=0,
    project='runs/pose',
    name='tower',
)
```

- 参数全部从 `config.yaml.train` 读，命令行不重复设置。
- `imgsz=1280`：俯拍 80m 时关键点像素跨度小，分辨率宁高勿低。
- 模型从 `yolo26n` 起步；如 mAP 不足，配置改成 `yolo26s`/`yolo26m` 重训。

### 3.7 `infer/pose_solver.py`（PnP 核心）

```python
def solve_pose(
    kpts_2d: np.ndarray,    # (N, 2)
    kpts_conf: np.ndarray,  # (N,)
    kpts_3d: np.ndarray,    # (N, 3) 与 kpts_2d 顺序对齐
    K: np.ndarray,
    dist: np.ndarray,
    conf_thresh: float = 0.5,
) -> dict | None:
    """返回 {'R': (3,3), 't': (3,), 'inliers': idx[], 'reproj_err': float} 或 None。"""
```

流程：
1. 用 `conf >= conf_thresh` 筛选有效关键点；若 < 4 返回 None。
2. `cv2.solvePnPRansac(objectPoints, imagePoints, K, dist, flags=cv2.SOLVEPNP_SQPNP, reprojectionError=4.0, iterationsCount=200)` → 拿到 rvec, tvec, inliers。
3. 若 inliers < 4：返回 None。
4. `cv2.solvePnPRefineLM` 在 inliers 上精修。
5. 计算 inliers 上的平均重投影误差（像素）。
6. `R = cv2.Rodrigues(rvec)[0]`，返回 dict。

> SQPNP 在 n≥3 时给出全局最优解，对噪声鲁棒；RANSAC 外层进一步抗外点；LM 精修提升精度。是当前 OpenCV PnP 的推荐组合。

### 3.8 `infer/visualize.py`

提供：
- `draw_keypoints(img, kpts_2d, kpts_conf, conf_thresh)`：红点画检测出的关键点，绿点画 PnP 重投影点（直观看误差）。
- `draw_axes(img, R, t, K, dist, axis_len=5.0)`：画物体局部坐标系 X/Y/Z（红/绿/蓝），单位为米。
- `draw_bbox3d(img, R, t, K, dist, bounds)`：在物体坐标系下用 12 关键点的 AABB 画 3D 包围盒。
- `put_pose_text(img, R, t)`：左上角文字显示距离 `||t||`、欧拉角。

### 3.9 `infer/infer_image.py` & `infer/infer_video.py`

- 共享一个 `run_on_frame(model, K, dist, kpts_3d, frame) -> annotated_frame` 函数（建议放到 `infer/pipeline.py` 复用）。
- `infer_image.py`：参数 `--input <path>`（单文件或目录）、`--output <dir>`，输出标注图 + 每帧一个 `*_pose.json`（R 矩阵、t、重投影误差）。
- `infer_video.py`：参数 `--input <video|0>`、`--output <video>`，逐帧推理并写视频；按 `q` 退出实时窗口。

## 4. 数据流

```
Base.obj + keypoints_3d.json + camera.yaml
        │
        ▼
[blender_render.py] —— sample camera pose ──▶ render image + project keypoints
        │
        ▼
dataset/images/*.png, dataset/labels/*.txt  (YOLO-pose 格式)
        │
        ▼
[train.py] ultralytics YOLO26-pose ──▶ runs/pose/tower/weights/best.pt
        │
        ▼
[infer_video.py] frame ──▶ model.predict ──▶ kpts_2d
                              │
                              ▼
                       [pose_solver.solve_pose] ──▶ R, t
                              │
                              ▼
                       [visualize] ──▶ annotated frame
```

## 5. 配置（`config.yaml`）

```yaml
camera:
  path: camera.yaml          # K, dist 从此读
model:
  obj: Base.obj
  keypoints_3d: keypoints_3d.json
  num_keypoints: 12
render:
  num_train: 1000
  num_val: 200
  image_size: [1920, 1080]
  sensor_width_mm: 36.0
  camera_distance: [20.0, 80.0]
  pitch_deg: [60.0, 90.0]
  yaw_deg: [0.0, 360.0]
  roll_jitter_deg: 5.0
  sun_strength: [2.0, 6.0]
  hdri_strength: [0.3, 1.0]
  backgrounds_dir: backgrounds
  textures_dir: textures
  ground_size: 200.0
  occlusion_eps_m: 0.01
  min_visible_keypoints: 4
  min_bbox_area_ratio: 0.005
train:
  weights: yolo26n-pose.pt
  epochs: 100
  imgsz: 1280
  batch: 8
  device: 0
  project: runs/pose
  name: tower
infer:
  weights: runs/pose/tower/weights/best.pt
  conf_thresh: 0.5            # 关键点置信度阈值
  det_conf_thresh: 0.25       # 检测置信度阈值
  ransac_reproj_err: 4.0
  ransac_iters: 200
  axis_len_m: 5.0
```

## 6. 错误处理与边界

| 情况 | 处理 |
|------|------|
| 渲染时 v≥1 关键点 < 4 | 该帧丢弃，调度层补采。 |
| 渲染时 bbox 太小 | 同上。 |
| 推理时检测置信度 < 阈值 | 该目标无 R/t，仅画检测框/关键点（如果有的话）。 |
| 推理时有效关键点 < 4 | 同上。 |
| PnP RANSAC inliers < 4 | 同上。 |
| 视频流单帧解码失败 | 跳过此帧，继续下一帧。 |
| `K` 与渲染分辨率不一致 | 启动时断言失败，直接报错退出。 |

## 7. 测试策略

**渲染端**：
- 单元：`utils/camera.py`、`utils/keypoints.py` 加载结果与原始文件对比。
- 集成：渲染 10 张，校验：(a) labels 中所有 v=2 的关键点像素坐标 ∈ [0,1] 内；(b) 用渲染时计算的 R/t 反投影关键点 3D 到 2D，与 labels 中的 2D 误差 < 0.5 像素。

**PnP 端**：
- 合成测试：取已知 R/t，把 keypoints_3d 投影到像素，加入像素噪声 σ=1，调用 `solve_pose`，断言估计的 t 误差 < 0.5m、旋转误差 < 2°。
- 加入 2 个外点（错配），断言 RANSAC 把它们剔出 inliers。

**端到端**：
- 在验证集上跑训好的模型，按 PnP 链路计算 R/t 与渲染时的 GT 比对，统计中位数 / P95 误差，作为模型质量指标。

## 8. 依赖

- Python 3.10+
- `ultralytics`（YOLO26-pose）
- `opencv-python`（PnP、视频 IO、可视化）
- `numpy`, `pyyaml`, `tqdm`
- Blender 3.6+ 或 4.x（headless，命令行调用）

## 9. 分阶段交付里程碑

1. **M1 工具与配置**：`utils/`、`config.yaml`、目录骨架。
2. **M2 渲染**：`data_synth/` 跑通，产出 50 张图人工抽查可见性、bbox、重投影一致性。
3. **M3 全量数据**：跑 1000+200 张，写 `data.yaml`。
4. **M4 训练**：跑通 100 epochs，得到 `best.pt`，校验 val mAP。
5. **M5 PnP & 单图推理**：`pose_solver.py` + `visualize.py` + `infer_image.py` 跑通，肉眼检查重投影叠加效果。
6. **M6 视频推理**：`infer_video.py` 跑通视频文件 + 摄像头。

## 10. 非目标（YAGNI）

- 不做对称翻转增强（`flip_idx=[]`）。
- 不做多目标场景（一张图只有一座塔）。
- 不做相机内参标定流程（用户提供 `camera.yaml`）。
- 不做模型量化 / TensorRT 加速（先要正确性，再谈速度）。
- 不做时序滤波（卡尔曼/IMM）；视频流逐帧独立 PnP。后续若抖动严重再加。
