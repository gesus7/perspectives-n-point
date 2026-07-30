"""Symmetry-aware keypoint loss for training YOLO-pose on symmetric objects.

Problem: a symmetric object has multiple physically-identical keypoint labelings.
Forcing the network to predict one fixed labeling gives contradictory gradients
across viewpoints (the pose loss won't converge). Canonicalizing labels by image
position makes training converge but destroys the fixed 2D<->3D correspondence
that PnP needs at inference.

Fix (symmetry-aware min-loss): keep FIXED physical keypoint ids in the labels so
PnP stays solvable, but at training time compute the keypoint loss against every
*proper* rigid symmetry of the object and backprop only the minimum. The network
is then free to converge to whichever symmetric labeling is easiest for a given
view, and its output always corresponds to a real rigid pose.

IMPORTANT: only PROPER rotations (det +1) are valid hypotheses. Reflections
(det -1, e.g. pure x- or y-mirror) cannot be produced by any real camera view,
so including them would let the network output a labeling PnP cannot solve.

PER-CLASS: the dataset has two classes (0=TowerBase, 1=TowerTop) sharing one
keypoint head. Each class has its OWN geometry and therefore its OWN symmetry
permutations; an instance must be scored only against its class's perms. The loss
recovers each foreground anchor's class (the same way Ultralytics selects target
keypoints) and minimizes over that class's permutations.

This module patches BOTH keypoint-loss implementations in Ultralytics 8.4.x:
  - v8PoseLoss.calculate_keypoints_loss        (returns 2-tuple)
  - PoseLoss26.calculate_keypoints_loss        (returns 3-tuple, has RLE)
PoseLoss26 is what end2end YOLO26-pose models use (wrapped in E2ELoss), so both
must be patched. It also wraps each class's loss() to stash batch["cls"] so the
patched keypoint loss can route instances to their class's perms. Call
patch_symmetry_aware_pose_loss(perms_by_class) once before train().
用于在对称物体上训练 YOLO-pose 的对称感知关键点损失。

问题：对称物体具有多个物理上相同的关键点标注方式。强迫网络预测一个固定的标注方式会导致不同视角下的梯度相互矛盾（姿态损失无法收敛）。
通过图像位置对标注进行规范化可以使训练收敛，但会破坏推理时 PnP 所需的固定 2D<->3D 对应关系。

解决方案（对称感知最小损失）：在标注中保持**固定的**物理关键点 ID，以便 PnP 仍然可解，但在训练时，
针对物体的每一种**真**刚体对称变换计算关键点损失，并仅反向传播最小值。这样网络就可以自由地收敛到对于给定视角最易学的对称标注方式，
而其输出始终对应一个真实的刚体姿态。

重要：只有真旋转（det +1）才是有效的假设。反射变换（det -1，例如纯 x 镜像或纯 y 镜像）无法由任何真实相机视角产生，
因此如果包含它们，网络可能输出 PnP 无法求解的标注方式。

按类别处理：数据集中有两个类别（0=塔基，1=塔顶），共享同一个关键点头。每个类别具有**各自的**几何结构，
因此具有**各自的**对称置换；一个实例必须仅针对其所属类别的置换进行评分。
损失函数会恢复每个前景锚框的类别（采用与 Ultralytics 选择目标关键点相同的方式），并在该类别的置换上取最小值。

本模块会同时修补 Ultralytics 8.4.x 中的两个关键点损失实现：
  - v8PoseLoss.calculate_keypoints_loss        （返回 2 元组）
  - PoseLoss26.calculate_keypoints_loss        （返回 3 元组，含 RLE）
端到端 YOLO26-pose 模型（包装在 E2ELoss 中）使用的是 PoseLoss26，因此两个都必须修补。
本模块还会包装每个类别的 loss() 函数，在其中保存 batch["cls"]，从而使修补后的关键点损失能够将各个实例路由到其所属类别的置换。
在调用 train() 之前，请调用一次 patch_symmetry_aware_pose_loss(perms_by_class)。
"""
import torch

from ultralytics.utils.loss import v8PoseLoss, PoseLoss26
from ultralytics.utils.ops import xyxy2xywh


def _keypoint_loss_per_instance(keypoint_loss, pred_kpt, gt_kpt, kpt_mask, area):
    """Replicate KeypointLoss.forward math but return per-instance loss (no mean).

    Returns tensor of shape (n_instances,). Mirrors Ultralytics' cocoeval form:
        e = d / ((2*sigma)^2 * (area+eps) * 2)
        loss = factor * sum_k (1 - exp(-e)) * mask
    """
    d = (pred_kpt[..., 0] - gt_kpt[..., 0]).pow(2) + (pred_kpt[..., 1] - gt_kpt[..., 1]).pow(2)
    kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
    e = d / ((2 * keypoint_loss.sigmas).pow(2) * (area + 1e-9) * 2)
    return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).sum(dim=1)  # (n_instances,)


def _perms_for_class(self, class_id, device):
    """Return (and cache on self) the perm LongTensor for a class id, on device.

    Falls back to the identity permutation if a class id has no registered perms
    (should not happen with a correct config)."""
    cache = self._sym_perms_cache
    key = int(class_id)
    t = cache.get(key)
    if t is None or t.device != device:
        perms_list = self._sym_perms_by_class.get(
            key, [list(range(self.kpt_shape[0]))]
        )
        t = torch.tensor(perms_list, dtype=torch.long, device=device)
        cache[key] = t
    return t


def _min_over_perclass_symmetries(self, pred_kpt, gt_kpt, kpt_mask, area, cls_fg):
    """Per-instance keypoint loss minimized over EACH instance's own class perms,
    then averaged over all foreground instances (matches the original mean reduction).

    pred_kpt/gt_kpt: (n_fg, K, dim), kpt_mask: (n_fg, K), area: (n_fg, 1),
    cls_fg: (n_fg,) integer class id per foreground instance.
    """
    n_fg = gt_kpt.shape[0]
    # Per-instance loss is computed in float32 (keypoint_loss.sigmas is float32),
    # so accumulate in float32 even when AMP runs the model in half precision.
    out = torch.zeros(n_fg, dtype=torch.float32, device=gt_kpt.device)
    # Process one class at a time so each subset uses only its own permutations.
    for c in torch.unique(cls_fg):
        sel = cls_fg == c
        perms = _perms_for_class(self, c.item(), gt_kpt.device)
        pk, gk, km, ar = pred_kpt[sel], gt_kpt[sel], kpt_mask[sel], area[sel]
        best = None
        for s in range(perms.shape[0]):
            p = perms[s]
            li = _keypoint_loss_per_instance(self.keypoint_loss, pk, gk[:, p, :], km[:, p], ar)
            best = li if best is None else torch.minimum(best, li)
        out[sel] = best
    return out.mean()


def _select_target_classes(self, batch_cls, batch_idx, target_gt_idx, masks):
    """Build the per-anchor class id, mirroring v8PoseLoss._select_target_keypoints.

    Scatters per-instance class ids into (BS, max_objs) by (image, within-image
    position), then gathers by target_gt_idx -> (BS, N_anchors). The caller indexes
    [masks] to obtain the class of every foreground anchor.
    """
    batch_idx = batch_idx.flatten()
    batch_size = len(masks)
    device = masks.device

    cls = batch_cls.to(device).view(-1).float()
    max_objs = torch.unique(batch_idx, return_counts=True)[1].max()
    batched_cls = torch.zeros((batch_size, max_objs), device=device)

    batch_idx_long = batch_idx.long()
    offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
    offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long))
    offsets = offsets.cumsum(0)
    within_idx = torch.arange(len(batch_idx), device=device) - offsets[batch_idx_long]
    batched_cls[batch_idx_long, within_idx] = cls

    selected = batched_cls.gather(1, target_gt_idx)  # (BS, N_anchors)
    return selected.long()


def make_v8_fn():
    """Symmetry-aware replacement for v8PoseLoss.calculate_keypoints_loss (2-tuple)."""

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)
        kpts_loss = 0
        kpts_obj_loss = 0
        if masks.any():
            target_bboxes = target_bboxes / stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            cls_fg = _select_target_classes(self, self._sym_cls, batch_idx, target_gt_idx, masks)[masks]
            kpts_loss = _min_over_perclass_symmetries(self, pred_kpt, gt_kpt, kpt_mask, area, cls_fg)
            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
        return kpts_loss, kpts_obj_loss

    return calculate_keypoints_loss


def make_pose26_fn():
    """Symmetry-aware replacement for PoseLoss26.calculate_keypoints_loss (3-tuple, RLE).

    The RLE loss term is preserved as-is (it operates per-keypoint on the model's
    flow-based confidence and is not part of the symmetric coordinate ambiguity).
    """

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)
        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0
        if masks.any():
            target_bboxes = target_bboxes / stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            cls_fg = _select_target_classes(self, self._sym_cls, batch_idx, target_gt_idx, masks)[masks]
            kpts_loss = _min_over_perclass_symmetries(self, pred_kpt, gt_kpt, kpt_mask, area, cls_fg)
            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask).clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
        return kpts_loss, kpts_obj_loss, rle_loss

    return calculate_keypoints_loss


def _wrap_loss(orig_loss):
    """Wrap a PoseLoss.loss() so it stashes batch['cls'] on the instance before
    delegating. The patched calculate_keypoints_loss reads self._sym_cls to route
    each instance to its class's symmetry perms."""

    def loss(self, preds, batch):
        self._sym_cls = batch["cls"]
        return orig_loss(self, preds, batch)

    loss._sym_wrapped = True
    return loss


def _install_perms(cls_obj, perms_by_class):
    """Attach the per-class perm dict + cache to a loss class (shared by instances)."""
    cls_obj._sym_perms_by_class = {int(k): list(v) for k, v in perms_by_class.items()}
    cls_obj._sym_perms_cache = {}


def patch_symmetry_aware_pose_loss(perms_by_class):
    """Patch both pose-loss classes to use the per-class symmetry-aware keypoint loss.

    perms_by_class: {class_id: list of full keypoint-index permutations (identity
    first)}, each describing a proper rigid symmetry of THAT class's keypoints.
    """
    for cls_obj, make_fn in ((v8PoseLoss, make_v8_fn), (PoseLoss26, make_pose26_fn)):
        _install_perms(cls_obj, perms_by_class)
        cls_obj.calculate_keypoints_loss = make_fn()
        # Wrap loss() once (idempotent) so batch['cls'] is available downstream.
        if not getattr(cls_obj.loss, "_sym_wrapped", False):
            cls_obj.loss = _wrap_loss(cls_obj.loss)

    counts = {c: len(p) for c, p in perms_by_class.items()}
    print(f"[symmetry-loss] patched v8PoseLoss + PoseLoss26 with per-class "
          f"proper-symmetry hypotheses: {counts}")
