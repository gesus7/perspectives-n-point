# 输电铁塔双塔段姿态估计 (YOLO-pose + PnP)

基于直升机俯视航拍视角，对对称输电铁塔的**两个塔段**进行端到端 6DoF 姿态估计：

- **TowerBase（类别 0）** —— 固定在地面的底部塔段。
- **TowerTop（类别 1）** —— 用绳索吊在直升机下方、悬浮在空中的顶部塔段。

两个塔段由同一个 YOLO26-pose 模型检测（两类别，共用 12 个关键点槽位），各自独立地用
OpenCV RANSAC-PnP 对其自身的 3D 关键点求解位姿。之后即可计算两者的相对位姿（顶部相对
于底部的 6DoF 位姿）。

```
合成双塔段数据 (Blender)  →  YOLO26-pose 关键点检测  →  分类别 OpenCV RANSAC-PnP  →  6DoF 姿态 (R, t)
```

由于目标具有几何对称性，常规的关键点训练方法会失效。本项目的核心是**分类别的对称性感知
训练损失（per-class symmetry-aware loss）**，它使网络能够收敛，同时为每个类别保持固定的
物理关键点 ID，从而确保推理时 PnP 依然可解。详见下文 [对称性问题](#对称性问题)。

---

## 流水线阶段

| 阶段 | 执行命令 |
|------|---------|
| 1. 渲染数据集 | `python data_synth/render_dataset_batch.py --num-train 5000 --num-val 500` |
| 2. 训练 | `python train/train.py --config config.yaml` |
| 3. 评估 | `python eval_dataset.py --dataset dataset --split val --weights <训练好的.pt>` |
| 4. 图像推理 | `python infer/infer_image.py --input <path> --output infer_out` |
| 5. 视频/摄像头推理 | `python infer/infer_video.py --input <video\|0>` |

## 环境配置

```bash
pip install -r requirements.txt
# 第 1 阶段（数据渲染）需要将 Blender 3.6+ / 4.x 添加到系统环境变量 PATH 中。
#   例如：export PATH="/e/Apps/Blender:$PATH"
# 推荐使用 yolo conda 环境 (Ultralytics 8.4.x, torch + CUDA)。
```

---

## 对称性问题

对称物体存在多种物理上等效的关键点标注方式。两种常规方法均无法解决此问题：

- **固定 ID 标签**（关键点 1 始终对应同一个物理角点）：在不同视角下，视觉上相同的角点会
  获得矛盾的标签，导致 `pose_loss` 无法收敛。
- **位置规范化标签**（如“最左侧角点 = ID 1”）：训练可以收敛，但破坏了 2D↔3D 的对应关系，
  导致 **PnP 无法求解姿态**。

### 解决方案：对称性感知最小损失

标签保持**固定的物理 ID**（对 PnP 安全）。训练时，计算预测值与物体**所有真旋转
（proper-rotation）对称状态**的关键点损失，并仅对**最小值**反向传播。这使网络可以自由
收敛到当前视角下最易预测的对称标签，且其输出始终对应一个 PnP 可求解的真实刚体姿态。

```
Loss = min(s ∈ 各对称状态) { keypoint_loss(pred, GT[permutation_s]) }
```

**仅有真旋转（行列式为 +1）是有效假设。** 真实相机无法产生反射（行列式为 -1，如纯镜像），
包含它们会导致网络输出 PnP 无法求解的标签。有效的对称状态由 3D 关键点自动发现。

### 分类别对称性（两个塔段）

数据集有**两个类别**，共用一个 YOLO-pose 头（kpt_shape `[12, 3]`）。每个类别几何不同，
因此各有自己的对称排列。损失会从 batch 中恢复每个前景 anchor 的类别，然后只在**该类别**
的排列上求最小值 —— TowerBase 实例绝不会用 TowerTop 的对称性来评分，反之亦然。

具体实现：

- `utils/symmetry.py` —— `compute_symmetry_perms(keypoints_3d)` 自动发现某个物体的真旋转
  对称性；`load_symmetry_perms_per_class({类别: json路径})` 返回分类别的字典。
- `train/symmetry_loss.py` —— `patch_symmetry_aware_pose_loss(perms_by_class)` 用猴子补丁
  修补 Ultralytics 的**两种**关键点损失实现（`v8PoseLoss` 和 `PoseLoss26`，后者用于端到端
  YOLO26-pose），并包裹 `loss()` 方法以暂存 `batch["cls"]`，使被修补的关键点损失能把每个
  实例路由到其类别的排列。补丁是**幂等的**（重复调用安全）。
- `train/train.py` 读取 `model.keypoints_3d`（底部）与 `model.keypoints_3d_top`（顶部），
  构建分类别字典，并在 `model.train()` 前自动应用补丁。调试时可用 `--no-symmetry` 禁用。

> **指标说明：** 由于模型可能输出任何有效的对称标签，常规的“预测点 i 对齐真实点 i”指标
> （包括 Ultralytics 内置的 pose mAP）即使姿态完美也会报告极大误差。`eval_dataset.py` 与
> 损失函数采用相同逻辑、**分类别**求最小值，从而报告**真实**几何精度。
>
> 在我们训练好的模型（towerv63）上：对称性**开启**时 RMSE 为 19 px（底部）/ 45 px（顶部），
> OKS 达 0.98；对称性**关闭**时 RMSE 飙升至约 250 px —— 这正是上述伪影。

---

## 合成数据渲染

`data_synth/render_dataset_batch.py` 会生成批处理的 Blender 进程，渲染随机化的航拍视图，
并写入两类别的 YOLO-pose 标签（每个物体 12 个关键点 + 可见性标志，每张图 0–2 个实例）。

**双塔段场景组合**（逐帧，按 `render.scene_weights` 加权）：

| 场景类型 | 权重 | 说明 |
|---------|------|------|
| base_only | 0.25 | 仅地面固定的底部塔段。相机在 20–80 m 处环绕底部。 |
| top_only | 0.25 | 仅悬浮的顶部塔段。相机在 ≤20 m 处环绕顶部。 |
| both | 0.50 | 底部在地面，顶部悬挂在相机→底部的视线射线上、距相机 ≤20 m（直升机吊着顶部，越过相机指向地面的底部）。 |

- 每 50 帧生成一张**纯背景**图像（无物体，空标签），用于难负样本训练。
- 顶部放置时会预留物体包围盒半径作为余量，确保**整个**顶部（最远的关键点）都在
  `top_camera_distance` 上限（20 m）以内。
- 顶部塔段被摆正（原生 +Y → 世界 +Z），并施加随机偏航角与轻微的绳索摆动
  （`render.top_tilt_jitter_deg`）。

**域随机化（Domain randomization）**（均对 PnP 安全 —— 不扰动相机内参与物体几何）：

- 相机：距离、俯仰角、偏航角、横滚角随机抖动（读取自 `config.yaml`）。
- 光照：每帧重新随机生成太阳方向、能量及冷/暖色调。
- 环境天空填充：强度与轻微色调随机化（`hdri_strength` 范围），柔化硬阴影，确保阴影中的
  关键点依然可见。
- 地面：随机化背景纹理、UV 映射（缩放/平面旋转/偏移）和粗糙度，确保完全相同的纹理不会
  出现两次。

**背景调度逻辑：**

- **无放回抽取**：每张背景在重复前都被使用一次，保证覆盖均匀。
- **随机种子**：默认每次运行都从操作系统熵池获取新种子，确保**每次渲染的背景顺序都不同**。
  可通过 `--bg-seed N` 复现特定顺序。

```bash
# 完整渲染（若省略标志，则从 config.yaml 读取 num_train/num_val）
python data_synth/render_dataset_batch.py --num-train 5000 --num-val 500

# 小型冒烟测试渲染（带可复现的背景顺序）
python data_synth/render_dataset_batch.py --num-train 20 --num-val 5 --bg-seed 42
```

输出：`dataset/images/{train,val}/*.png`、`dataset/labels/{train,val}/*.txt`
（多实例，每个可见物体一行）及 `dataset/data.yaml`（`names: {0: TowerBase, 1: TowerTop}`）。

---

## 评估

`eval_dataset.py` 在标记数据集上推理，并报告**分类别**的检测率、边界框 IoU、关键点 OKS 和
关键点 RMSE（像素）。它**默认支持对称性感知**，并使用**分类别**的对称排列，使每个物体只与
自己的有效标注比较。匹配限定为同类别预测（TowerBase 真值只匹配 TowerBase 预测）。

```bash
# 在渲染的 val 划分上做对称性感知评估
python eval_dataset.py --dataset dataset --split val \
    --weights runs/pose/perspectives-n-point/towerv63/weights/best.pt \
    --iou-thresh 0.5

# --no-symmetry  使用原始的 i 对 i 匹配（用于观察未处理对称性时的指标假象）
```

我们训练好的模型在 500 张 val 图上的指标：

| 类别 | GT | 检出 | 检测率 | IoU | OKS | RMSE(px) |
|------|----|------|--------|-----|-----|----------|
| TowerBase | 386 | 376 | 97.4% | 0.904 | 0.984 | 19.2 |
| TowerTop | 324 | 247 | 76.2% | 0.806 | 0.972 | 45.0 |
| 总计 | 710 | 623 | 87.7% | 0.863 | 0.979 | 29.4 |

---

## 推理

推理是**分类别的**：每个检测都被路由到其所属类别的 3D 关键点来求解 PnP。标注图像用不同
颜色绘制两个类别（底部 = 红色关键点，顶部 = 橙色），并为每个物体标注类别名 + 6DoF 姿态
文本，JSON 输出中每个检测带有 `class_id` + `class_name`。

```bash
# 单张图像或整个图像文件夹
python infer/infer_image.py --input some.jpg  --output infer_out
python infer/infer_image.py --input some_dir/ --output infer_out
#   -> 输出: infer_out/<name>_annot.jpg (标注图) + <name>_pose.json (R, t, 重投影误差, 类别)

# 视频文件或网络摄像头 (按 q 退出)
python infer/infer_video.py --input some.mp4 --output out.mp4
python infer/infer_video.py --input 0

# 使用特定权重检查点
python infer/infer_image.py --input some.jpg --weights path/to/best.pt
```

姿态解算（`infer/pose_solver.py`）：RANSAC + `SOLVEPNP_SQPNP`，再对内点做 LM
（Levenberg-Marquardt）优化。置信度低于 `infer.conf_thresh` 的关键点不参与 PnP；至少需要
4 个高置信度关键点。

### 相对位姿与飞行指令

当画面中恰好检测到 **一个 TowerBase 和一个 TowerTop** 时，推理流水线会额外计算顶部塔段
相对于底部塔段的 6DoF 位姿，并给出**直升机飞行指令**——一段方向提示，告诉操作员该往哪个
方向移动，使悬挂的顶部塔段对准地面固定的底部塔段。

```
T_base_top = T_cam_base⁻¹ × T_cam_top    （顶部塔段在底部塔段坐标系中的位姿）
move_cam   = t_base − t_top              （直升机在相机坐标系中需移动的位移）
```

位移向量在相机坐标系（直升机操作员俯视视角）中解释：

| 轴 | + 方向 | − 方向 | 含义 |
|------|--------|--------|------|
| X | 向右 | 向左 | 画面横向 |
| Y | 向后 | 向前 | 画面纵向 |
| Z | 向下 | 向上 | 深度方向（下降/爬升） |

极小分量（<0.5m 死区）会被忽略。指令会作为横幅绘制在标注图像上（`draw_text` 通过 PIL +
TrueType 中文字体渲染），并写入 `_pose.json` 的 `relative_pose` 字段。

输出示例（img_000002，同时检测到两个塔段）：
```
请向下41.6m、向右15.0m、向后0.6m
move down 41.6m, right 15.0m, back 0.6m
```

实现：`infer/relative_pose.py`（`compute_relative_pose` + `flight_instruction`）。

---

## 项目结构

```text
config.yaml              所有可调参数（相机/模型/渲染/训练/推理）
camera.yaml              相机内参 K、畸变参数、图像尺寸
mesh/Base.obj            铁塔底部塔段模型 (单位：米)
mesh/TowerTop.obj        铁塔顶部塔段模型 (单位：米)
keypoints_3d_original.json  TowerBase 的 12 个关键点 (物体局部坐标系)
keypoints_3d_top.json   TowerTop 的 12 个关键点 (物体局部坐标系)

data_synth/
  blender_render.py        在 Blender 内部运行：渲染并标记单个批次（两类别）
  render_dataset_batch.py  生成批处理 Blender 进程（渲染主入口）
  render_dataset.py        单进程渲染器（适用于小任务）
train/
  train.py                 训练主入口；自动应用分类别对称性感知损失
  symmetry_loss.py         修补 Ultralytics 关键点损失 (v8PoseLoss + PoseLoss26)
utils/
  symmetry.py              自动发现物体的真旋转关键点排列
  camera.py / config.py / keypoints.py    各类数据加载器
infer/
  pipeline.py              分类别的 YOLO 推理 + 分类别 PnP
  pose_solver.py           RANSAC-SQPNP + LM 姿态解算
  relative_pose.py         顶部相对底部的 6DoF 位姿 + 直升机飞行指令
  visualize.py             绘制关键点、3D 边界框、坐标轴及分类别姿态文本
  infer_image.py / infer_video.py         命令行推理入口
eval_dataset.py           分类别的对称性感知精度评估
scripts/                  完整性验证脚本（例如：通过真实标签解算 PnP）
tests/                    pytest 单元测试 + Blender 集成测试（15 个通过）
```

## 核心配置参数 (`config.yaml`)

| 参数 | 含义 |
| --- | --- |
| `model.obj` / `model.obj_top` | 底部 / 顶部塔段 OBJ 路径 |
| `model.keypoints_3d` / `keypoints_3d_top` | 每个类别的 3D 关键点 JSON；也是对称性发现的数据源 |
| `model.classes` | 类别 ID → 名称映射（`{0: TowerBase, 1: TowerTop}`） |
| `render.num_train` / `num_val` | 计划渲染的图像数量 |
| `render.scene_weights` | `[base_only, top_only, both]` 逐帧采样权重 |
| `render.camera_distance` | base_only 场景下相机距铁塔的距离范围（米） |
| `render.top_camera_distance` | 相机→顶部的距离范围（米）；硬上限，确保顶部 ≤20 m |
| `render.top_tilt_jitter_deg` | 悬挂顶部塔段的绳索摆动倾角 |
| `render.pitch_deg` | 俯仰角范围；90° = 垂直向下 |
| `render.sun_strength` / `hdri_strength` | 太阳能量 / 环境光填充的随机范围 |
| `train.epochs` / `imgsz` / `batch` | 训练超参数 |
| `train.pose` | 姿态损失权重系数 |
| `infer.weights` | 默认模型检查点 |
| `infer.conf_thresh` | 关键点置信度阈值（低于则不输入 PnP） |
| `infer.det_conf_thresh` | 检测框置信度阈值 |
| `infer.ransac_reproj_err` / `ransac_iters` | RANSAC-PnP 相关设置 |

**显存不足（如 RTX 3060 6GB）?** 请调低 `train.batch` (16 → 8 → 4) 或 `train.imgsz`
(1088 → 960)。如果无法自动下载 `yolo26n-pose.pt`，请将 `train.weights` 改为本地已有的权重
（如 `yolo11n-pose.pt`）。

## 测试

```bash
pytest -v          # PnP 单元测试 + Blender 集成测试（需要 Blender 在 PATH 中）
```
