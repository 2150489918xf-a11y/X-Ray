#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🌟 Target & Snipe Pipeline v2
目标锁定与精准狙击 — 4 阶医学影像分析流水线

第一阶: TorchXRayVision 定性 (是什么病？)
第二阶: PSPNet 肺部掩膜 (降噪滤网)
第三阶: Grad-CAM × 肺掩膜 → OpenCV bbox (靶心提取)
第四阶: 智能分流 (弥漫性→热力图 / 实体性→MedSAM)

运行: python pipeline_v2.py
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
IMG_NAME = os.path.splitext(os.path.basename(IMG_PATH))[0]  # e.g. "pneumonia_test"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_output", IMG_NAME)
MEDSAM_IMG_SIZE = 1024
POSITIVE_THRESHOLD = 0.5  # 严格阈值: >50% 才算阳性

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 疾病分类表 (第四阶分流依据)
# ============================================================
# 弥漫性疾病: 边界模糊，不适合 MedSAM 画精确轮廓
DIFFUSE_DISEASES = {
    'Pneumonia', 'Infiltration', 'Effusion', 'Edema',
    'Consolidation', 'Emphysema', 'Fibrosis', 'Atelectasis'
}
# 实体性病灶: 边界清晰，适合 MedSAM 精确分割
SOLID_LESIONS = {
    'Nodule', 'Mass', 'Lung Lesion', 'Hernia'
}
# 其他解剖异常: 用热力图 + 解剖匹配
ANATOMICAL_ABNORM = {
    'Cardiomegaly', 'Enlarged Cardiomediastinum', 'Pleural_Thickening',
    'Pneumothorax', 'Fracture', 'Lung Opacity'
}

# 中英文名映射
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

ANATOMY_CN = {
    'Left Lung': '左肺', 'Right Lung': '右肺',
    'Heart': '心脏', 'Aorta': '主动脉',
    'Facies Diaphragmatica': '膈面', 'Mediastinum': '纵隔',
    'Spine': '脊柱'
}


def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


# ============================================================
# 第一阶: 大视野定性 — 到底是啥病？(支持并发检测)
# ============================================================
CONCURRENT_GAP = 0.10   # Top2 与 Top1 差距 <10% 时视为并发
SUSPECT_THRESHOLD = 0.30 # 30-50% 之间且显著高于第二名 → 标记为可疑


def _classify_disease_type(name):
    """判断疾病属于哪种类型"""
    if name in DIFFUSE_DISEASES:
        return "diffuse"
    elif name in SOLID_LESIONS:
        return "solid"
    else:
        return "anatomical"


def stage1_classify(img_path):
    """
    输入原始图片，运行 TorchXRayVision DenseNet121。
    支持并发检测: 如果多个疾病概率接近且都 >阈值，全部作为目标。
    """
    banner("第一阶: 大视野定性 — TorchXRayVision 分类")

    # 加载图像
    print(f"  图像路径: {img_path}")
    img_xrv = xrv.utils.load_image(img_path)
    img_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"  原始尺寸: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    # 预处理
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img_224 = transform(img_xrv)

    # 加载模型
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(DEVICE)
    model.eval()

    img_tensor = torch.from_numpy(img_224).unsqueeze(0).to(DEVICE)

    # 推理
    with torch.no_grad():
        preds = model(img_tensor)
    all_results = dict(zip(model.pathologies, preds[0].cpu().numpy()))

    # 打印所有结果
    print(f"\n  18 项检查结果 (阈值 > {POSITIVE_THRESHOLD}):")
    sorted_results = sorted(all_results.items(), key=lambda x: -x[1])
    for name, prob in sorted_results:
        cn = PATHOLOGY_CN.get(name, name)
        marker = "✅" if prob > POSITIVE_THRESHOLD else ("🟡" if prob > SUSPECT_THRESHOLD else "  ")
        print(f"    {marker} {cn:8s} ({name:30s}): {prob:.1%}")

    # === 并发检测逻辑 ===
    top_name, top_prob = sorted_results[0]

    # 情况 1: 最高概率都没超过可疑阈值 → 未见异常
    if top_prob <= SUSPECT_THRESHOLD:
        print(f"\n  🟢 结论: 未见明显异常 (最高概率 {top_name}: {top_prob:.1%} < {SUSPECT_THRESHOLD})")
        return None, None, None, None, None, all_results

    # 收集所有目标 (>阈值 且 与Top1差距<10%)
    targets = []
    for name, prob in sorted_results:
        if prob > POSITIVE_THRESHOLD and (top_prob - prob) <= CONCURRENT_GAP:
            disease_type = _classify_disease_type(name)
            targets.append({
                'name': name,
                'name_cn': PATHOLOGY_CN.get(name, name),
                'prob': float(prob),
                'disease_type': disease_type
            })

    # 情况 2: 没有超过严格阈值，但 Top1 在可疑区间 (30-50%)
    #         且它比第二名高出 15%+ → 标记为"可疑"
    if not targets and top_prob > SUSPECT_THRESHOLD:
        second_prob = sorted_results[1][1] if len(sorted_results) > 1 else 0
        if (top_prob - second_prob) > 0.15:
            disease_type = _classify_disease_type(top_name)
            targets.append({
                'name': top_name,
                'name_cn': PATHOLOGY_CN.get(top_name, top_name),
                'prob': float(top_prob),
                'disease_type': disease_type,
                'suspect': True  # 标记为可疑而非确定
            })
            print(f"\n  🟡 可疑目标: {targets[0]['name_cn']} ({top_prob:.1%})")
            print(f"     概率未达阈值但显著高于第二名 (差距 {top_prob - second_prob:.1%})")

    if not targets:
        print(f"\n  🟢 结论: 未见明显异常 (最高概率 {top_name}: {top_prob:.1%})")
        return None, None, None, None, None, all_results

    # 打印锁定结果
    if len(targets) == 1:
        t = targets[0]
        status = "🟡 可疑" if t.get('suspect') else "🎯 锁定"
        print(f"\n  {status}: {t['name_cn']} ({t['name']})  置信度: {t['prob']:.1%}")
        print(f"  📋 疾病类型: {t['disease_type']}")
    else:
        print(f"\n  🎯 检测到 {len(targets)} 个并发目标 (概率差距 <{CONCURRENT_GAP:.0%}):")
        for i, t in enumerate(targets, 1):
            print(f"    [{i}] {t['name_cn']} ({t['name']}): {t['prob']:.1%}  [{t['disease_type']}]")

    return model, img_tensor, img_224, img_xrv, img_rgb, all_results, targets


# ============================================================
# 第二阶: 解剖学遮罩 — PSPNet 肺部掩膜
# ============================================================
def stage2_lung_mask(img_xrv):
    """
    运行 PSPNet，但只提取左右肺掩膜作为内部降噪滤网。
    输出: 224x224 的二值 lung_mask (肺=1, 其他=0)
    """
    banner("第二阶: 解剖学遮罩 — PSPNet 肺部掩膜")

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

    # 只提取左肺和右肺
    targets = list(seg_model.targets)
    lung_mask_512 = np.zeros((512, 512), dtype=np.uint8)

    for lung_name in ['Left Lung', 'Right Lung']:
        if lung_name in targets:
            idx = targets.index(lung_name)
            mask = seg_output[0, idx].cpu().numpy()
            lung_mask_512 = np.maximum(lung_mask_512, (mask > 0.5).astype(np.uint8))
            area_pct = (mask > 0.5).sum() / (512 * 512)
            cn = ANATOMY_CN.get(lung_name, lung_name)
            print(f"  ✅ {cn} ({lung_name}) 掩膜提取完成, 面积占比: {area_pct:.1%}")

    # 缩放到 224x224 (与 Grad-CAM 空间一致)
    lung_mask_224 = cv2.resize(lung_mask_512, (224, 224), interpolation=cv2.INTER_NEAREST)

    total_lung_pct = lung_mask_224.sum() / (224 * 224)
    print(f"  📐 肺部掩膜总面积: {total_lung_pct:.1%} (224x224 空间)")

    # 提取解剖定位信息 (用于后续报告)
    anatomy_masks_512 = {}
    for i, target in enumerate(targets):
        mask = seg_output[0, i].cpu().numpy()
        anatomy_masks_512[target] = (mask > 0.5).astype(np.uint8)

    return lung_mask_224, lung_mask_512, anatomy_masks_512


# ============================================================
# 第三阶: 靶心提取 — 病种自适应遮罩 + 多目标BBox
# ============================================================
def _patch_inplace_relu():
    """临时将 torch.relu_ 替换为 torch.relu (Grad-CAM 所需)"""
    original_relu_ = torch.relu_
    torch.relu_ = lambda input: torch.relu(input)
    original_f_relu = F.relu
    def safe_relu(input, inplace=False):
        return original_f_relu(input, inplace=False)
    F.relu = safe_relu
    def restore():
        torch.relu_ = original_relu_
        F.relu = original_f_relu
    return restore


# 掩膜路由: 根据疾病类型决定用什么掩膜
MASK_ROUTE_LUNG = {'Pneumonia', 'Consolidation', 'Mass', 'Nodule', 'Infiltration',
                   'Lung Lesion', 'Lung Opacity', 'Emphysema', 'Fibrosis'}
MASK_ROUTE_CARDIAC = {'Cardiomegaly', 'Enlarged Cardiomediastinum'}
MASK_ROUTE_PLEURAL = {'Effusion', 'Pneumothorax', 'Pleural_Thickening', 'Atelectasis'}


def stage3_target_extraction(model, img_tensor, img_224, lung_mask_224,
                             target_name, anatomy_masks_512=None):
    """
    仅针对第一阶锁定的目标疾病生成 Grad-CAM。
    动态掩膜路由 + 多目标 BBox 提取。
    返回: cam_raw, clean_cam, bboxes_224 (列表!)
    """
    banner(f"第三阶: 靶心提取 — {PATHOLOGY_CN.get(target_name, target_name)}")

    target_cn = PATHOLOGY_CN.get(target_name, target_name)
    target_idx = list(model.pathologies).index(target_name)
    print(f"  🎯 仅对 [{target_cn}] 生成 Grad-CAM (类别索引: {target_idx})")

    # 修复 in-place ReLU
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU) and module.inplace:
            module.inplace = False
    restore_relu = _patch_inplace_relu()

    try:
        target_layer = model.features[-1]
        activations, gradients = {}, {}
        def fwd_hook(m, i, o): activations['v'] = o.detach()
        def bwd_hook(m, gi, go): gradients['v'] = go[0].detach()
        fh = target_layer.register_forward_hook(fwd_hook)
        bh = target_layer.register_full_backward_hook(bwd_hook)

        model.zero_grad()
        img_input = img_tensor.detach().clone().requires_grad_(True)
        output = model(img_input)
        output[0, target_idx].backward()

        act = activations['v']
        grad = gradients['v']
        weights = grad.mean(dim=[2, 3], keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        fh.remove()
        bh.remove()
    finally:
        restore_relu()

    print(f"  ✅ Grad-CAM 原始热力图生成完成")

    # === 动作 B: 病种自适应遮罩 (Dynamic Anatomic Routing) ===
    if target_name in MASK_ROUTE_LUNG:
        # 纯肺内病变: 严格肺掩膜
        clean_cam = cam * lung_mask_224.astype(np.float32)
        mask_desc = "严格肺掩膜"
    elif target_name in MASK_ROUTE_CARDIAC:
        # 心血管病变: 不用肺掩膜！
        clean_cam = cam.copy()
        mask_desc = "无掩膜 (心血管病变)"
    elif target_name in MASK_ROUTE_PLEURAL:
        # 胸膜腔病变: 膨胀肺掩膜, 保留边缘
        dilated = cv2.dilate(lung_mask_224, np.ones((25, 25), np.uint8), iterations=1)
        clean_cam = cam * dilated.astype(np.float32)
        mask_desc = "膨胀肺掩膜 (保留胸膜边缘)"
    else:
        clean_cam = cam * lung_mask_224.astype(np.float32)
        mask_desc = "默认肺掩膜"

    if clean_cam.max() > 0:
        clean_cam = clean_cam / clean_cam.max()

    noise_pct = (1 - (clean_cam.sum() / max(cam.sum(), 1e-8))) * 100
    print(f"  ✅ 掩膜策略: {mask_desc}, 去噪: {noise_pct:.1f}%")

    # === 动作 C: 多目标 BBox 提取 ===
    bboxes_224 = []
    used_thresh = 0
    for thresh in [0.5, 0.4, 0.3, 0.2]:
        binary = (clean_cam > thresh).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_area = cv2.contourArea(max(contours, key=cv2.contourArea))
            found = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # 保留 >最大轮廓20% 且 >100像素 的所有区域
                if area > max_area * 0.20 and area > 100:
                    x, y, w, h = cv2.boundingRect(cnt)
                    box_area = w * h
                    if box_area < 224 * 224 * 0.6:  # 排除过大框
                        found.append([x, y, x + w, y + h])
            if found:
                bboxes_224 = found
                used_thresh = thresh
                break

    if not bboxes_224:
        cy, cx = np.unravel_index(clean_cam.argmax(), clean_cam.shape)
        half = 40
        bboxes_224 = [[
            int(max(0, cx - half)), int(max(0, cy - half)),
            int(min(224, cx + half)), int(min(224, cy + half))
        ]]
        print(f"  ⚠️ 使用质心回退 bbox")
    else:
        print(f"  ✅ 提取到 {len(bboxes_224)} 个 bbox (阈值={used_thresh})")

    for i, bb in enumerate(bboxes_224):
        print(f"    📍 bbox[{i}] (224空间): {bb}")

    return cam, clean_cam, bboxes_224


# ============================================================
# 第四阶: 智能分流
# ============================================================
def stage4_route_diffuse(img_rgb, img_224, clean_cam, bboxes_224, target_name, anatomy_masks_512):
    """
    路线 A: 弥漫性疾病 → 只展示热力图叠加, 不使用 MedSAM
    支持多 bbox (双肺多发)
    """
    banner("第四阶: 弥漫性疾病 → 热力图展示")
    target_cn = PATHOLOGY_CN.get(target_name, target_name)
    print(f"  📋 {target_cn} 属于弥漫性疾病，边界本身模糊")
    print(f"  🚫 跳过 MedSAM (强行画线显得外行)")

    H, W = img_rgb.shape[:2]
    cam_orig = cv2.resize(clean_cam, (W, H), interpolation=cv2.INTER_LINEAR)

    # 映射所有 bbox 到原始分辨率
    sx, sy = W / 224, H / 224
    bboxes_orig = []
    for bb in bboxes_224:
        bboxes_orig.append([int(bb[0]*sx), int(bb[1]*sy),
                            int(bb[2]*sx), int(bb[3]*sy)])
    print(f"  📍 {len(bboxes_orig)} 个病灶区域")

    # 解剖定位
    cam_512 = cv2.resize(clean_cam, (512, 512))
    cam_mask_512 = (cam_512 > 0.3).astype(np.uint8)
    best_anat, best_overlap = "Unknown", 0
    for anat_name, anat_mask in anatomy_masks_512.items():
        inter = np.logical_and(cam_mask_512, anat_mask).sum()
        area = cam_mask_512.sum()
        if area > 0:
            ratio = inter / area
            if ratio > best_overlap:
                best_overlap = ratio
                best_anat = anat_name

    anat_cn = ANATOMY_CN.get(best_anat, best_anat)
    # 判断是否双肺: 检查 bbox 跨越中线
    bilateral = False
    if len(bboxes_orig) > 1:
        lefts = [bb for bb in bboxes_orig if (bb[0]+bb[2])/2 < W/2]
        rights = [bb for bb in bboxes_orig if (bb[0]+bb[2])/2 >= W/2]
        if lefts and rights:
            bilateral = True
            anat_cn = "双肺"
            print(f"  📍 解剖定位: 双肺多发 (左{len(lefts)}处, 右{len(rights)}处)")
        else:
            print(f"  📍 解剖定位: {anat_cn}, {len(bboxes_orig)}处病灶")
    else:
        print(f"  📍 解剖定位: {anat_cn} ({best_anat})")

    return {
        'route': 'diffuse',
        'heatmap_orig': cam_orig,
        'bboxes_orig': bboxes_orig,
        'anatomy': best_anat,
        'anatomy_cn': anat_cn,
        'overlap': best_overlap,
        'bilateral': bilateral,
        'lesion_count': len(bboxes_orig)
    }


def stage4_route_solid(img_rgb, clean_cam, bboxes_224, target_name, anatomy_masks_512):
    """
    路线 B: 实体性病灶 → 对每个 bbox 分别使用 MedSAM 精确分割
    """
    banner("第四阶: 实体性病灶 → MedSAM 精确分割")
    target_cn = PATHOLOGY_CN.get(target_name, target_name)
    print(f"  📋 {target_cn} 属于实体性病灶，{len(bboxes_224)} 个框")

    H, W = img_rgb.shape[:2]
    sx, sy = W / 224, H / 224

    # 映射所有 bbox + 10% padding
    bboxes_padded = []
    for bb in bboxes_224:
        bx = [int(bb[0]*sx), int(bb[1]*sy), int(bb[2]*sx), int(bb[3]*sy)]
        bw, bh = bx[2]-bx[0], bx[3]-bx[1]
        mx, my = int(bw*0.10), int(bh*0.10)  # 10% padding
        padded = [max(0,bx[0]-mx), max(0,bx[1]-my),
                  min(W,bx[2]+mx), min(H,bx[3]+my)]
        bboxes_padded.append(padded)

    if not os.path.exists(MEDSAM_CHECKPOINT):
        print(f"  ⚠️ MedSAM 权重未找到, 回退到热力图模式")
        cam_orig = cv2.resize(clean_cam, (W, H))
        return {
            'route': 'solid_fallback',
            'heatmap_orig': cam_orig,
            'bboxes_orig': bboxes_padded,
            'anatomy': 'Unknown', 'anatomy_cn': '未知',
            'overlap': 0, 'lesion_count': len(bboxes_padded)
        }

    from segment_anything import sam_model_registry

    print(f"  加载 MedSAM...")
    medsam = sam_model_registry["vit_b"]()
    sd = torch.load(MEDSAM_CHECKPOINT, map_location=DEVICE, weights_only=True)
    medsam.load_state_dict(sd)
    medsam = medsam.to(DEVICE).eval()

    # image embedding (只算一次)
    img_1024 = cv2.resize(img_rgb, (MEDSAM_IMG_SIZE, MEDSAM_IMG_SIZE))
    img_norm = (img_1024.astype(np.float64) - img_1024.min()) / np.clip(img_1024.max()-img_1024.min(), 1e-8, None)
    img_t = torch.tensor(img_norm).float().permute(2,0,1).unsqueeze(0).to(DEVICE)

    print(f"  计算 image embedding...")
    with torch.no_grad():
        img_embed = medsam.image_encoder(img_t)

    # 对每个 bbox 分别分割
    combined_mask = np.zeros((H, W), dtype=np.uint8)
    precise_bboxes = []

    for i, bp in enumerate(bboxes_padded):
        b1024 = np.array([bp[0]/W*1024, bp[1]/H*1024,
                          bp[2]/W*1024, bp[3]/H*1024])
        with torch.no_grad():
            box_t = torch.as_tensor(b1024[None,None,...], dtype=torch.float, device=DEVICE)
            sp, de = medsam.prompt_encoder(points=None, boxes=box_t, masks=None)
            logits, _ = medsam.mask_decoder(
                image_embeddings=img_embed,
                image_pe=medsam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp, dense_prompt_embeddings=de,
                multimask_output=False)
            mask = torch.sigmoid(logits)
            mask = F.interpolate(mask, size=(H,W), mode='bilinear', align_corners=False)
            mb = (mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        combined_mask = np.maximum(combined_mask, mb)

        # 精确 bbox
        contours, _ = cv2.findContours(mb*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            pts = np.concatenate(contours)
            x,y,w,h = cv2.boundingRect(pts)
            precise_bboxes.append([x, y, x+w, y+h])
        else:
            precise_bboxes.append(bp)

        print(f"    bbox[{i}] 分割完成")

    fg = np.count_nonzero(combined_mask)
    area_ratio = fg / combined_mask.size
    print(f"  ✅ MedSAM 共分割 {len(bboxes_padded)} 个区域, 总面积: {area_ratio:.2%}")

    # 解剖定位
    cam_512 = cv2.resize(clean_cam, (512, 512))
    cam_mask_512 = (cam_512 > 0.3).astype(np.uint8)
    best_anat, best_overlap = "Unknown", 0
    for anat_name, anat_mask in anatomy_masks_512.items():
        inter = np.logical_and(cam_mask_512, anat_mask).sum()
        area = cam_mask_512.sum()
        if area > 0:
            ratio = inter / area
            if ratio > best_overlap:
                best_overlap = ratio
                best_anat = anat_name

    return {
        'route': 'solid',
        'mask': combined_mask,
        'bbox_precise': precise_bboxes,
        'bboxes_orig': bboxes_padded,
        'area_ratio': float(area_ratio),
        'fg_pixels': int(fg),
        'anatomy': best_anat,
        'anatomy_cn': ANATOMY_CN.get(best_anat, best_anat),
        'overlap': best_overlap,
        'lesion_count': len(bboxes_padded)
    }


# ============================================================
# NMS 空间语义合并 — IoU 去重
# ============================================================
IOU_MERGE_THRESHOLD = 0.6  # IoU > 60% 视为同一物理病灶

def _calc_iou(a, b):
    """计算两个 bbox [x1,y1,x2,y2] 的 IoU"""
    xi1at = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0, xi2-xi1at) * max(0, yi2-yi1)
    aa = (a[2]-a[0]) * (a[3]-a[1])
    ab = (b[2]-b[0]) * (b[3]-b[1])
    union = aa + ab - inter
    return inter / max(union, 1e-8)


def nms_merge_findings(all_target_results, img_shape):
    """
    将多个目标按空间 IoU 合并为物理发现 (findings)。
    IoU > 60% 的目标合并: 概率最高的为 primary，其余为 associated。
    """
    banner("NMS 空间语义合并")
    H, W = img_shape[:2]
    sx, sy = W / 224, H / 224
    n = len(all_target_results)

    # 将所有目标的主 bbox 转到同一空间 (原图空间)
    orig_bboxes = []
    for tr in all_target_results:
        bbs = tr['bboxes_224']
        # 取最大的 bbox 作为代表
        if bbs:
            biggest = max(bbs, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            orig_bboxes.append([int(biggest[0]*sx), int(biggest[1]*sy),
                                int(biggest[2]*sx), int(biggest[3]*sy)])
        else:
            orig_bboxes.append([0, 0, W, H])

    # 贪心合并: 按概率降序
    indices = list(range(n))
    indices.sort(key=lambda i: all_target_results[i]['target']['prob'], reverse=True)

    merged = [False] * n
    findings = []

    for i in indices:
        if merged[i]:
            continue
        primary = all_target_results[i]
        associated = []

        for j in indices:
            if j == i or merged[j]:
                continue
            iou = _calc_iou(orig_bboxes[i], orig_bboxes[j])
            if iou > IOU_MERGE_THRESHOLD:
                associated.append(all_target_results[j])
                merged[j] = True
                print(f"  🔗 合并: {all_target_results[j]['target']['name_cn']} "
                      f"→ {primary['target']['name_cn']} (IoU={iou:.2f})")

        merged[i] = True
        findings.append({
            'primary': primary,
            'associated': associated
        })

    print(f"  ✅ {n} 个原始目标 → {len(findings)} 个物理发现")
    for i, f in enumerate(findings, 1):
        p = f['primary']['target']
        assoc_str = ", ".join(a['target']['name_cn'] for a in f['associated'])
        print(f"    [{i}] {p['name_cn']} ({p['prob']:.1%})"
              f"{' + ' + assoc_str if assoc_str else ''}")

    return findings


# ============================================================
# Master Canvas — 唯一总览图
# ============================================================
FINDING_COLORS = ['#00ff00', '#00bfff', '#ff6600', '#ff00ff', '#ffff00']

def generate_master_canvas(img_rgb, findings, lung_mask_224, all_results):
    """
    Master Canvas — 多面板布局:
      上排: [原始 X-Ray]  [分类柱状图]
      下排: 每个发现一个面板 (各自独立清晰)
    """
    from matplotlib.patches import Rectangle
    banner("生成 Master Canvas")

    H, W = img_rgb.shape[:2]
    n_findings = len(findings)

    # 布局: 上排 2 格, 下排 n_findings 格
    n_cols = max(n_findings, 2)
    fig = plt.figure(figsize=(5 * n_cols, 12))
    gs = fig.add_gridspec(2, n_cols, height_ratios=[1.2, 1], hspace=0.25, wspace=0.15)

    # ── 上排左: 原始 X-Ray ──
    ax_orig = fig.add_subplot(gs[0, :n_cols//2])
    ax_orig.imshow(img_rgb, cmap='gray')
    ax_orig.set_title("原始 X-Ray", fontsize=14)
    ax_orig.axis('off')

    # ── 上排右: 分类柱状图 ──
    ax_bar = fig.add_subplot(gs[0, n_cols//2:])
    filtered = {k: v for k, v in all_results.items() if v > 0.2}
    sorted_items = sorted(filtered.items(), key=lambda x: x[1])
    names = [PATHOLOGY_CN.get(n, n) for n, _ in sorted_items]
    probs = [v for _, v in sorted_items]
    primary_names = {f['primary']['target']['name'] for f in findings}
    colors = ['#e74c3c' if n in primary_names else '#bdc3c7'
              for n, _ in sorted_items]
    ax_bar.barh(names, probs, color=colors)
    ax_bar.axvline(x=POSITIVE_THRESHOLD, color='red', linestyle='--', alpha=0.7,
                   label=f'阈值={POSITIVE_THRESHOLD}')
    ax_bar.set_xlim(0, 1)
    ax_bar.set_title(f"分类概率 ({n_findings} 个发现)", fontsize=14)
    ax_bar.legend(fontsize=9)

    # ── 下排: 每个发现一个面板 ──
    for fi, finding in enumerate(findings):
        ax = fig.add_subplot(gs[1, fi])
        color_hex = FINDING_COLORS[fi % len(FINDING_COLORS)]

        p = finding['primary']
        s4 = p['stage4']
        target_cn = p['target']['name_cn']
        prob = p['target']['prob']

        if s4['route'] == 'diffuse' or s4['route'] == 'solid_fallback':
            # 弥漫性: 热力图叠加在原图上
            hm = s4['heatmap_orig']
            ax.imshow(img_rgb, cmap='gray', alpha=0.5)
            ax.imshow(hm, cmap='jet', alpha=0.5, extent=[0, W, H, 0])
            # 画 bbox
            for bb in s4['bboxes_orig']:
                rect = Rectangle((bb[0],bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                                  lw=2, edgecolor=color_hex, facecolor='none', linestyle='--')
                ax.add_patch(rect)
            dtype_label = "弥漫性"
        elif s4['route'] == 'solid':
            # 实体性: MedSAM 遮罩
            overlay = img_rgb.copy()
            mask_vis = s4['mask']
            cr, cg, cb = int(color_hex[1:3],16), int(color_hex[3:5],16), int(color_hex[5:7],16)
            overlay[mask_vis > 0] = (overlay[mask_vis > 0] * 0.5 + np.array([cr,cg,cb]) * 0.5).astype(np.uint8)
            ax.imshow(overlay)
            for bb in s4.get('bbox_precise', []):
                rect = Rectangle((bb[0],bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                                  lw=2, edgecolor=color_hex, facecolor='none')
                ax.add_patch(rect)
            dtype_label = "MedSAM"
        else:
            ax.imshow(img_rgb, cmap='gray')
            dtype_label = "回退"

        # 标题
        title = f"发现{fi+1}: {target_cn}\n{prob:.1%} [{dtype_label}]"
        location = s4.get('anatomy_cn', '')
        if location:
            title += f" | {location}"
        if finding['associated']:
            assoc = ",".join(a['target']['name_cn'] for a in finding['associated'])
            title += f"\n+{assoc}"
        ax.set_title(title, fontsize=10, color=color_hex, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f"Target & Snipe v2 — {n_findings}个物理发现 (NMS合并)",
                 fontsize=15, fontweight='bold')
    path = os.path.join(OUTPUT_DIR, "master_canvas.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Master Canvas: {path}")
    return path


# ============================================================
# 精简 JSON 报告 (按物理发现分组)
# ============================================================
def generate_report(findings, all_results, img_shape):
    """生成精简报告: 按物理病灶分组, 给 LLM 用"""
    banner("生成诊断报告")
    H, W = img_shape[:2]

    report = {
        "pipeline": "Target & Snipe v2 (NMS)",
        "image_size": f"{W}x{H}",
        "model": "DenseNet121 (densenet121-res224-all)",
        "finding_count": len(findings),
    }

    for fi, finding in enumerate(findings, 1):
        p = finding['primary']
        s4 = p['stage4']
        t = p['target']

        f_data = {
            "primary_disease": f"{t['name']} ({t['prob']:.1%})",
            "disease_type": t['disease_type'],
            "location": s4.get('anatomy_cn', '未知'),
            "lesion_count": s4.get('lesion_count', 1),
        }

        if finding['associated']:
            f_data["associated_features"] = [
                f"{a['target']['name']} ({a['target']['prob']:.1%})"
                for a in finding['associated']
            ]

        if s4['route'] == 'diffuse':
            f_data["visual_action"] = "弥漫性热力图叠加"
            f_data["bilateral"] = s4.get('bilateral', False)
        elif s4['route'] == 'solid':
            f_data["visual_action"] = f"MedSAM精确分割, 面积{s4['area_ratio']:.2%}"
        else:
            f_data["visual_action"] = "热力图叠加(回退)"

        report[f"finding_{fi}"] = f_data

    # 保存 JSON
    path_json = os.path.join(OUTPUT_DIR, "report_v2.json")
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 文本报告
    all_primary_names = set()
    for f in findings:
        all_primary_names.add(f['primary']['target']['name'])
        for a in f['associated']:
            all_primary_names.add(a['target']['name'])

    lines = [
        "=" * 50,
        "   胸部 X 光 AI 辅助诊断报告 (v2-NMS)",
        "=" * 50,
        f"图像: {W}x{H} | 模型: DenseNet121",
        f"物理发现: {len(findings)} 个",
        "",
    ]

    for fi, finding in enumerate(findings, 1):
        p = finding['primary']
        t = p['target']
        s4 = p['stage4']
        route_name = "热力图" if s4['route'] != 'solid' else "MedSAM分割"
        location = s4.get('anatomy_cn', '未知')
        n_lesions = s4.get('lesion_count', 1)

        lines.append(f"━━━ 发现 {fi}: {t['name_cn']} ({t['prob']:.1%}) ━━━")
        lines.append(f"  位置: {location} | 路线: {route_name} | 区域数: {n_lesions}")

        if finding['associated']:
            assoc = ", ".join(f"{a['target']['name_cn']}({a['target']['prob']:.1%})"
                              for a in finding['associated'])
            lines.append(f"  伴随特征: {assoc}")
        lines.append("")

    lines.extend([
        "━" * 50,
        "【其他检查项目 (>20%)】",
        "━" * 50,
    ])
    for name, prob in sorted(all_results.items(), key=lambda x: -x[1]):
        if name not in all_primary_names and prob > 0.2:
            cn = PATHOLOGY_CN.get(name, name)
            lines.append(f"  {cn} ({name}): {prob:.1%}")
    lines.extend([
        "",
        "=" * 50,
        "  注: 本报告由 AI 自动生成，仅供辅助参考",
        "=" * 50
    ])

    text = "\r\n".join(lines)
    path_txt = os.path.join(OUTPUT_DIR, "report_v2.txt")
    with open(path_txt, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"  📄 JSON: {path_json}")
    print(f"  📄 TXT:  {path_txt}")
    print(f"\n{'─'*60}")
    print(text)
    print(f"{'─'*60}")

    return report


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("🌟 Target & Snipe Pipeline v2 (NMS)")
    print(f"  设备: {DEVICE}")

    # === 第一阶: 定性 ===
    result = stage1_classify(IMG_PATH)
    if result[0] is None:
        all_results = result[-1]
        print("\n✅ 分析完成: 未见明显异常")
        sys.exit(0)

    model, img_tensor, img_224, img_xrv, img_rgb, all_results, targets = result

    # === 第二阶: 肺掩膜 ===
    lung_mask_224, lung_mask_512, anatomy_masks_512 = stage2_lung_mask(img_xrv)

    # === 第三阶 + 第四阶: 逐目标处理 ===
    all_target_results = []

    for i, target in enumerate(targets):
        target_name = target['name']
        disease_type = target['disease_type']
        is_suspect = target.get('suspect', False)

        if len(targets) > 1:
            banner(f"处理目标 [{i+1}/{len(targets)}]: {target['name_cn']}")

        cam_raw, clean_cam, bboxes_224 = stage3_target_extraction(
            model, img_tensor, img_224, lung_mask_224, target_name)

        if disease_type == "solid":
            stage4_result = stage4_route_solid(
                img_rgb, clean_cam, bboxes_224, target_name, anatomy_masks_512)
        else:
            stage4_result = stage4_route_diffuse(
                img_rgb, img_224, clean_cam, bboxes_224, target_name, anatomy_masks_512)

        stage4_result['is_suspect'] = is_suspect

        all_target_results.append({
            'target': target,
            'cam_raw': cam_raw,
            'clean_cam': clean_cam,
            'bboxes_224': bboxes_224,
            'stage4': stage4_result
        })

    # === NMS 合并 ===
    findings = nms_merge_findings(all_target_results, img_rgb.shape)

    # === Master Canvas ===
    generate_master_canvas(img_rgb, findings, lung_mask_224, all_results)

    # === 报告 ===
    report = generate_report(findings, all_results, img_rgb.shape)

    print(f"\n{'='*60}")
    print(f"  ✅ Target & Snipe Pipeline v2 (NMS) 完成!")
    print(f"  原始目标: {len(targets)} → 合并发现: {len(findings)}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")
