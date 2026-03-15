#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整医学影像分析 Pipeline 验证
TorchXRayVision(分类) → Grad-CAM(粗定位) → PSPNet(解剖分割) → MedSAM(精细分割) → 文本报告

运行: python pipeline_verify.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchxrayvision as xrv
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 修复中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(SCRIPT_DIR, "tests", "pneumonia_test.jpg")
MEDSAM_CHECKPOINT = os.path.join(SCRIPT_DIR, "model", "medsam_vit_b.pth")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_output", "pipeline")
MEDSAM_IMG_SIZE = 1024
POSITIVE_THRESHOLD = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 解剖结构中英文名映射
ANATOMY_CN = {
    'Left Clavicle': '左锁骨', 'Right Clavicle': '右锁骨',
    'Left Scapula': '左肩胛骨', 'Right Scapula': '右肩胛骨',
    'Left Lung': '左肺', 'Right Lung': '右肺',
    'Left Hilus Pulmonis': '左肺门', 'Right Hilus Pulmonis': '右肺门',
    'Heart': '心脏', 'Aorta': '主动脉',
    'Facies Diaphragmatica': '膈面', 'Mediastinum': '纵隔',
    'Weasand': '气管', 'Spine': '脊柱'
}

# 病理中英文名映射
PATHOLOGY_CN = {
    'Atelectasis': '肺不张', 'Consolidation': '实变',
    'Infiltration': '浸润', 'Pneumothorax': '气胸',
    'Edema': '水肿', 'Emphysema': '肺气肿',
    'Fibrosis': '纤维化', 'Effusion': '胸腔积液',
    'Pneumonia': '肺炎', 'Pleural_Thickening': '胸膜增厚',
    'Cardiomegaly': '心脏扩大', 'Nodule': '结节',
    'Mass': '肿块', 'Hernia': '疝气',
    'Lung Lesion': '肺部病变', 'Fracture': '骨折',
    'Lung Opacity': '肺部阴影', 'Enlarged Cardiomediastinum': '心纵隔增大'
}


def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


# ============================================================
# Step 1: 加载图像
# ============================================================
def load_image():
    banner("Step 1/6: 加载图像")
    print(f"  图像路径: {IMG_PATH}")

    # 加载为 torchxrayvision 格式 (归一化到 [-1024, 1024])
    img_xrv = xrv.utils.load_image(IMG_PATH)
    print(f"  XRV 格式: shape={img_xrv.shape}, range=[{img_xrv.min():.0f}, {img_xrv.max():.0f}]")

    # 加载为 OpenCV 格式 (用于可视化和 MedSAM)
    img_bgr = cv2.imdecode(np.fromfile(IMG_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"  原始尺寸: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    return img_xrv, img_bgr, img_rgb


# ============================================================
# Step 2: DenseNet121 病理分类
# ============================================================
def classify(img_xrv):
    banner("Step 2/6: DenseNet121 病理分类")

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img_224 = transform(img_xrv)

    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(DEVICE)
    model.eval()

    img_tensor = torch.from_numpy(img_224).unsqueeze(0).to(DEVICE)

    # 需要梯度 (为 Grad-CAM 准备)
    img_tensor.requires_grad_(True)

    preds = model(img_tensor)
    results = dict(zip(model.pathologies, preds[0].detach().cpu().numpy()))

    # 输出结果
    positives = {k: v for k, v in results.items() if v > POSITIVE_THRESHOLD}
    print(f"  检测到 {len(positives)} 项疑似阳性 (阈值>{POSITIVE_THRESHOLD}):")
    for name, prob in sorted(positives.items(), key=lambda x: -x[1]):
        cn = PATHOLOGY_CN.get(name, name)
        print(f"    ▸ {name} ({cn}): {prob:.1%}")

    return model, img_tensor, img_224, results, positives


# ============================================================
# Step 3: Grad-CAM 热力图定位
# ============================================================
def _patch_inplace_relu():
    """
    临时将 torch.relu_ (in-place) 替换为 torch.relu (non-in-place)
    DenseNet 的 _DenseLayer.bn_function 在内部调用 torch.relu_，
    这会破坏 autograd 计算图，导致 Grad-CAM 反向传播失败。
    返回一个恢复函数。
    """
    original_relu_ = torch.relu_
    torch.relu_ = lambda input: torch.relu(input)
    # 同时 patch F.relu 的 inplace 路径
    original_f_relu = F.relu
    def safe_relu(input, inplace=False):
        return original_f_relu(input, inplace=False)
    F.relu = safe_relu

    def restore():
        torch.relu_ = original_relu_
        F.relu = original_f_relu
    return restore


def _replace_relu_inplace(model):
    """将模型中所有 in-place ReLU 替换为非 in-place 版本"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU) and module.inplace:
            module.inplace = False


def grad_cam(model, img_tensor, img_224, positives):
    banner("Step 3/6: Grad-CAM 热力图定位")

    if not positives:
        print("  无阳性病理，跳过 Grad-CAM")
        return {}

    # 修复 DenseNet 的 in-place ReLU 问题 (module 级别 + 函数级别)
    _replace_relu_inplace(model)
    restore_relu = _patch_inplace_relu()

    try:
        return _grad_cam_impl(model, img_tensor, img_224, positives)
    finally:
        restore_relu()


def _grad_cam_impl(model, img_tensor, img_224, positives):

    # 获取最后一个卷积层
    target_layer = model.features[-1]
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    cam_results = {}

    for pathology_name in positives:
        idx = list(model.pathologies).index(pathology_name)

        # 前向传播
        model.zero_grad()
        img_input = img_tensor.detach().clone().requires_grad_(True)
        output = model(img_input)

        # 反向传播目标类
        target_score = output[0, idx]
        target_score.backward(retain_graph=False)

        # 计算 Grad-CAM
        act = activations['value']
        grad = gradients['value']
        weights = grad.mean(dim=[2, 3], keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # 归一化
        if cam.max() > 0:
            cam = cam / cam.max()

        # 提取高亮区域 bounding box (使用更高阈值以获得更精确的定位)
        # 尝试逐步降低阈值直到找到合理大小的 bbox
        bbox_224 = None
        for thresh in [0.6, 0.5, 0.4, 0.3]:
            binary = (cam > thresh).astype(np.uint8)
            coords = np.where(binary > 0)
            if len(coords[0]) > 0:
                y1, y2 = coords[0].min(), coords[0].max()
                x1, x2 = coords[1].min(), coords[1].max()
                box_area = (x2 - x1) * (y2 - y1)
                img_area = 224 * 224
                # 如果 bbox 面积 < 60% 图像面积，认为是合理的定位
                if box_area < img_area * 0.6:
                    bbox_224 = [int(x1), int(y1), int(x2), int(y2)]
                    break
        if bbox_224 is None:
            # 所有阈值都无法得到合理 bbox，使用质心周围区域
            cy, cx = np.unravel_index(cam.argmax(), cam.shape)
            half = 56  # 224 / 4
            bbox_224 = [
                int(max(0, cx - half)), int(max(0, cy - half)),
                int(min(224, cx + half)), int(min(224, cy + half))
            ]

        cam_results[pathology_name] = {
            'heatmap': cam,
            'bbox_224': bbox_224
        }

        cn = PATHOLOGY_CN.get(pathology_name, pathology_name)
        print(f"  ▸ {pathology_name} ({cn}): Grad-CAM bbox = {bbox_224}")

    fh.remove()
    bh.remove()

    # 保存 Grad-CAM 可视化
    n_results = len(cam_results)
    fig, axes = plt.subplots(1, n_results + 1, figsize=(5 * (n_results + 1), 5))
    if n_results == 0:
        return cam_results

    axes_list = axes if isinstance(axes, np.ndarray) else [axes]
    axes_list[0].imshow(img_224[0], cmap='gray')
    axes_list[0].set_title("Original", fontsize=12)
    axes_list[0].axis('off')

    for i, (name, data) in enumerate(cam_results.items()):
        ax = axes_list[i + 1]
        ax.imshow(img_224[0], cmap='gray', alpha=0.7)
        ax.imshow(data['heatmap'], cmap='jet', alpha=0.4)
        bbox = data['bbox_224']
        from matplotlib.patches import Rectangle
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                         linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        cn = PATHOLOGY_CN.get(name, name)
        ax.set_title(f"{name}\n({cn})", fontsize=10)
        ax.axis('off')

    plt.suptitle("Grad-CAM Localization", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gradcam_results.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grad-CAM 可视化已保存: {path}")

    return cam_results


# ============================================================
# Step 4: PSPNet 解剖分割 + 区域匹配
# ============================================================
def anatomical_segmentation(img_xrv, cam_results):
    banner("Step 4/6: PSPNet 解剖分割 + 区域匹配")

    transform_seg = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(512)
    ])
    img_512 = transform_seg(img_xrv)

    seg_model = xrv.baseline_models.chestx_det.PSPNet()
    seg_model = seg_model.to(DEVICE)
    seg_model.eval()

    with torch.no_grad():
        seg_output = seg_model(torch.from_numpy(img_512).unsqueeze(0).to(DEVICE))

    seg_masks = {}
    for i, target in enumerate(seg_model.targets):
        mask = seg_output[0, i].cpu().numpy()
        seg_masks[target] = (mask > 0.5).astype(np.uint8)

    # 将 Grad-CAM bbox 从 224 映射到 512 坐标系,  计算与解剖结构的重叠
    location_results = {}
    scale = 512 / 224

    for pathology_name, cam_data in cam_results.items():
        bbox_224 = cam_data['bbox_224']
        bbox_512 = [int(b * scale) for b in bbox_224]

        # 在 512 空间中创建 Grad-CAM 关注区域掩膜
        cam_resized = cv2.resize(cam_data['heatmap'], (512, 512))
        cam_mask = (cam_resized > 0.3).astype(np.uint8)

        # 计算与每个解剖结构的重叠率
        overlaps = {}
        for anat_name, anat_mask in seg_masks.items():
            intersection = np.logical_and(cam_mask, anat_mask).sum()
            cam_area = cam_mask.sum()
            if cam_area > 0:
                overlap_ratio = intersection / cam_area
                if overlap_ratio > 0.05:  # 至少 5% 重叠
                    overlaps[anat_name] = float(overlap_ratio)

        # 按重叠率排序
        sorted_overlaps = sorted(overlaps.items(), key=lambda x: -x[1])
        primary_location = sorted_overlaps[0] if sorted_overlaps else ("Unknown", 0.0)

        location_results[pathology_name] = {
            'bbox_512': bbox_512,
            'primary_anatomy': primary_location[0],
            'primary_overlap': primary_location[1],
            'all_overlaps': dict(sorted_overlaps[:3])
        }

        cn_path = PATHOLOGY_CN.get(pathology_name, pathology_name)
        cn_anat = ANATOMY_CN.get(primary_location[0], primary_location[0])
        print(f"  ▸ {pathology_name} ({cn_path}): 主要位于 {primary_location[0]} ({cn_anat}), 重叠率 {primary_location[1]:.1%}")

    return seg_masks, location_results, img_512


# ============================================================
# Step 5: MedSAM 精细分割
# ============================================================
def medsam_segmentation(img_rgb, cam_results, location_results):
    banner("Step 5/6: MedSAM 精细分割")

    if not os.path.exists(MEDSAM_CHECKPOINT):
        print(f"  [跳过] 未找到 MedSAM 权重: {MEDSAM_CHECKPOINT}")
        return {}

    from segment_anything import sam_model_registry

    print("  加载 MedSAM 模型...")
    # 先用 CPU 映射加载权重，避免 CUDA 不可用时报错
    medsam_model = sam_model_registry["vit_b"]()
    state_dict = torch.load(MEDSAM_CHECKPOINT, map_location=DEVICE, weights_only=True)
    medsam_model.load_state_dict(state_dict)
    medsam_model = medsam_model.to(DEVICE)
    medsam_model.eval()
    print("  MedSAM 模型加载完成")

    H, W = img_rgb.shape[:2]

    # 预处理: resize 到 1024x1024
    img_1024 = cv2.resize(img_rgb, (MEDSAM_IMG_SIZE, MEDSAM_IMG_SIZE))
    img_1024_norm = (img_1024.astype(np.float64) - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
    img_tensor = torch.tensor(img_1024_norm).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # 计算 image embedding (只需一次)
    print("  计算图像 embedding...")
    with torch.no_grad():
        img_embed = medsam_model.image_encoder(img_tensor)
    print("  Image embedding 完成")

    medsam_results = {}

    for pathology_name, cam_data in cam_results.items():
        # 从 Grad-CAM bbox (224空间) 映射到原始图像空间
        bbox_224 = cam_data['bbox_224']

        # 检查 bbox 是否有意义 (面积小于图像的 70%)
        box_area_224 = (bbox_224[2] - bbox_224[0]) * (bbox_224[3] - bbox_224[1])
        if box_area_224 > 224 * 224 * 0.7:
            cn = PATHOLOGY_CN.get(pathology_name, pathology_name)
            print(f"  ▸ {pathology_name} ({cn}): 跳过 (Grad-CAM 范围过大，无法精确定位)")
            continue

        scale_x = W / 224
        scale_y = H / 224
        bbox_orig = [
            int(bbox_224[0] * scale_x),
            int(bbox_224[1] * scale_y),
            int(bbox_224[2] * scale_x),
            int(bbox_224[3] * scale_y)
        ]

        # 适当扩大 bbox (增加 15% margin)
        bw = bbox_orig[2] - bbox_orig[0]
        bh = bbox_orig[3] - bbox_orig[1]
        margin_x = int(bw * 0.15)
        margin_y = int(bh * 0.15)
        bbox_padded = [
            max(0, bbox_orig[0] - margin_x),
            max(0, bbox_orig[1] - margin_y),
            min(W, bbox_orig[2] + margin_x),
            min(H, bbox_orig[3] + margin_y)
        ]

        # 映射到 1024 空间
        bbox_1024 = np.array([
            bbox_padded[0] / W * MEDSAM_IMG_SIZE,
            bbox_padded[1] / H * MEDSAM_IMG_SIZE,
            bbox_padded[2] / W * MEDSAM_IMG_SIZE,
            bbox_padded[3] / H * MEDSAM_IMG_SIZE
        ])

        # MedSAM 推理
        with torch.no_grad():
            box_torch = torch.as_tensor(bbox_1024[None, None, ...], dtype=torch.float, device=DEVICE)
            sparse_emb, dense_emb = medsam_model.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )
            low_res_logits, _ = medsam_model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False
            )
            low_res_pred = torch.sigmoid(low_res_logits)
            mask = F.interpolate(low_res_pred, size=(H, W), mode='bilinear', align_corners=False)
            mask = mask.squeeze().cpu().numpy()
            mask_binary = (mask > 0.5).astype(np.uint8)

        # 计算分割区域的统计信息
        fg_pixels = np.count_nonzero(mask_binary)
        total_pixels = mask_binary.size
        area_ratio = fg_pixels / total_pixels

        # 找到精确的 bounding box
        contours, _ = cv2.findContours(mask_binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.concatenate(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            precise_bbox = [x, y, x + w, y + h]
        else:
            precise_bbox = bbox_padded

        medsam_results[pathology_name] = {
            'mask': mask_binary,
            'bbox_precise': precise_bbox,
            'area_ratio': float(area_ratio),
            'fg_pixels': int(fg_pixels),
            'bbox_input': bbox_padded
        }

        cn = PATHOLOGY_CN.get(pathology_name, pathology_name)
        print(f"  ▸ {pathology_name} ({cn}):")
        print(f"    输入框: {bbox_padded}")
        print(f"    精确框: {precise_bbox}")
        print(f"    分割面积: {area_ratio:.2%} (前景 {fg_pixels} 像素)")

    return medsam_results


# ============================================================
# Step 6: 生成文本报告
# ============================================================
def generate_report(results, positives, cam_results, location_results, medsam_results, img_shape):
    banner("Step 6/6: 生成文本报告")

    H, W = img_shape[:2]

    report = {
        "image_size": f"{W}x{H}",
        "model": "DenseNet121 (densenet121-res224-all)",
        "positive_threshold": POSITIVE_THRESHOLD,
        "total_pathologies_checked": len(results),
        "positive_count": len(positives),
        "summary": f"在胸部X光片中检测到 {len(positives)} 项疑似阳性病理",
        "findings": [],
        "negative_findings": []
    }

    # 阳性结果
    for name, prob in sorted(positives.items(), key=lambda x: -x[1]):
        finding = {
            "pathology_en": name,
            "pathology_cn": PATHOLOGY_CN.get(name, name),
            "probability": round(float(prob), 4),
            "probability_pct": f"{float(prob):.1%}"
        }

        # 添加定位信息
        if name in location_results:
            loc = location_results[name]
            anat = loc['primary_anatomy']
            finding["location"] = {
                "primary_anatomy_en": anat,
                "primary_anatomy_cn": ANATOMY_CN.get(anat, anat),
                "overlap_ratio": round(loc['primary_overlap'], 3),
                "related_structures": {
                    ANATOMY_CN.get(k, k): round(v, 3)
                    for k, v in loc['all_overlaps'].items()
                }
            }

        # 添加 Grad-CAM bbox
        if name in cam_results:
            bbox_224 = cam_results[name]['bbox_224']
            finding["gradcam_bbox_224"] = bbox_224

        # 添加 MedSAM 精细分割信息
        if name in medsam_results:
            ms = medsam_results[name]
            finding["medsam_segmentation"] = {
                "precise_bbox": ms['bbox_precise'],
                "area_ratio": round(ms['area_ratio'], 4),
                "area_pct": f"{ms['area_ratio']:.2%}",
                "foreground_pixels": ms['fg_pixels']
            }
            # 生成自然语言描述
            anat_cn = finding.get("location", {}).get("primary_anatomy_cn", "未知区域")
            bbox = ms['bbox_precise']
            cx = (bbox[0] + bbox[2]) / 2 / W
            cy = (bbox[1] + bbox[3]) / 2 / H

            pos_h = "上方" if cy < 0.4 else ("下方" if cy > 0.6 else "中部")
            pos_w = "偏左" if cx < 0.4 else ("偏右" if cx > 0.6 else "居中")

            finding["description"] = (
                f"{finding['pathology_cn']}，位于{anat_cn}区域（{pos_h}{pos_w}），"
                f"病灶面积约占图像 {ms['area_ratio']:.2%}，"
                f"精确边界框 [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
            )
        elif name in location_results:
            anat_cn = finding.get("location", {}).get("primary_anatomy_cn", "未知区域")
            finding["description"] = f"{finding['pathology_cn']}，主要位于{anat_cn}区域"

        report["findings"].append(finding)

    # 阴性结果 (简要)
    for name, prob in sorted(results.items(), key=lambda x: -x[1]):
        if name not in positives:
            report["negative_findings"].append({
                "pathology_en": name,
                "pathology_cn": PATHOLOGY_CN.get(name, name),
                "probability": round(float(prob), 4)
            })

    # 输出 JSON 报告
    report_json = json.dumps(report, ensure_ascii=False, indent=2)

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, "diagnosis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_json)
    print(f"  JSON 报告已保存: {report_path}")

    # 生成可读文本报告
    text_report = generate_readable_report(report)
    text_path = os.path.join(OUTPUT_DIR, "diagnosis_report.txt")
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text_report)
    print(f"  文本报告已保存: {text_path}")

    # 打印报告
    print("\n" + "─" * 60)
    print(text_report)
    print("─" * 60)

    return report, report_json


def generate_readable_report(report):
    """生成可读的中文文本报告 (适合直接给 LLM)"""
    lines = []
    lines.append("=" * 50)
    lines.append("     胸部 X 光 AI 辅助诊断报告")
    lines.append("=" * 50)
    lines.append(f"图像尺寸: {report['image_size']}")
    lines.append(f"分析模型: {report['model']}")
    lines.append(f"阳性阈值: {report['positive_threshold']}")
    lines.append(f"检查项目: {report['total_pathologies_checked']} 项")
    lines.append("")
    lines.append(f"▶ {report['summary']}")
    lines.append("")

    if report['findings']:
        lines.append("━" * 50)
        lines.append("【疑似阳性发现】")
        lines.append("━" * 50)
        for i, f in enumerate(report['findings'], 1):
            lines.append(f"\n  {i}. {f['pathology_cn']} ({f['pathology_en']})")
            lines.append(f"     置信度: {f['probability_pct']}")
            if 'location' in f:
                loc = f['location']
                lines.append(f"     解剖位置: {loc['primary_anatomy_cn']} ({loc['primary_anatomy_en']})")
            if 'medsam_segmentation' in f:
                seg = f['medsam_segmentation']
                lines.append(f"     精确边界框: {seg['precise_bbox']}")
                lines.append(f"     病灶面积占比: {seg['area_pct']}")
            if 'description' in f:
                lines.append(f"     描述: {f['description']}")

    lines.append("")
    lines.append("━" * 50)
    lines.append("【阴性项目】")
    lines.append("━" * 50)
    neg_items = [f"{nf['pathology_cn']}({nf['probability']:.1%})" for nf in report['negative_findings']]
    lines.append("  " + ", ".join(neg_items))

    lines.append("\n" + "=" * 50)
    lines.append("  注: 本报告由 AI 模型自动生成，仅供辅助参考")
    lines.append("  最终诊断请以医师判断为准")
    lines.append("=" * 50)

    return "\n".join(lines)


# ============================================================
# 可视化汇总
# ============================================================
def save_summary_visualization(img_224, img_bgr, cam_results, medsam_results, positives):
    """保存 pipeline 汇总可视化"""
    if not positives:
        return

    # 选取概率最高的病理来展示
    top_pathology = max(positives, key=positives.get)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # 1. 原始图像
    axes[0].imshow(img_224[0], cmap='gray')
    axes[0].set_title("1. Input X-Ray", fontsize=13)
    axes[0].axis('off')

    # 2. Grad-CAM
    if top_pathology in cam_results:
        axes[1].imshow(img_224[0], cmap='gray', alpha=0.7)
        axes[1].imshow(cam_results[top_pathology]['heatmap'], cmap='jet', alpha=0.4)
        cn = PATHOLOGY_CN.get(top_pathology, top_pathology)
        axes[1].set_title(f"2. Grad-CAM: {cn}", fontsize=13)
        axes[1].axis('off')

    # 3. MedSAM 分割
    if top_pathology in medsam_results:
        ms = medsam_results[top_pathology]
        axes[2].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        mask_vis = np.zeros((*ms['mask'].shape, 4))
        mask_vis[ms['mask'] > 0] = [1, 0, 0, 0.4]
        axes[2].imshow(mask_vis)
        bbox = ms['bbox_precise']
        from matplotlib.patches import Rectangle
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                         linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
        axes[2].add_patch(rect)
        axes[2].set_title(f"3. MedSAM Segmentation", fontsize=13)
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, "MedSAM\nNot Available", ha='center', va='center', fontsize=14)
        axes[2].axis('off')

    # 4. 分类结果柱状图 (只显示 top 8)
    sorted_results = sorted(positives.items(), key=lambda x: -x[1])
    names = [PATHOLOGY_CN.get(n, n) for n, _ in sorted_results[:8]]
    probs = [p for _, p in sorted_results[:8]]
    colors = ['#e74c3c' if p > 0.5 else '#3498db' for p in probs]
    axes[3].barh(range(len(names)), probs, color=colors)
    axes[3].set_yticks(range(len(names)))
    axes[3].set_yticklabels(names, fontsize=10)
    axes[3].set_xlim(0, 1)
    axes[3].set_title("4. Classification", fontsize=13)
    axes[3].axvline(x=0.5, color='red', linestyle='--', alpha=0.3)
    axes[3].invert_yaxis()

    plt.suptitle("Complete Medical Imaging Pipeline", fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pipeline_summary.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Pipeline 汇总图已保存: {path}")


# ============================================================
# 主流程
# ============================================================
def main():
    banner("完整医学影像分析 Pipeline 验证")
    print(f"  设备: {DEVICE}")
    print(f"  MedSAM 权重: {'✓ 已找到' if os.path.exists(MEDSAM_CHECKPOINT) else '✗ 未找到'}")

    # Step 1
    img_xrv, img_bgr, img_rgb = load_image()

    # Step 2
    model, img_tensor, img_224, results, positives = classify(img_xrv)

    # Step 3
    cam_results = grad_cam(model, img_tensor, img_224, positives)

    # Step 4
    seg_masks, location_results, img_512 = anatomical_segmentation(img_xrv, cam_results)

    # Step 5
    medsam_results = medsam_segmentation(img_rgb, cam_results, location_results)

    # Step 6
    report, report_json = generate_report(
        results, positives, cam_results, location_results, medsam_results, img_bgr.shape
    )

    # 可视化汇总
    save_summary_visualization(img_224, img_bgr, cam_results, medsam_results, positives)

    banner("✅ Pipeline 验证完成!")
    print(f"  所有输出保存在: {OUTPUT_DIR}")

    return report


if __name__ == "__main__":
    main()
