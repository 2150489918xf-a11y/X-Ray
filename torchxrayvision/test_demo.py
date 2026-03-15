#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TorchXRayVision 模型测试脚本
- 加载预训练 DenseNet121 模型对胸部 X 光片进行病理分类
- 加载 PSPNet 模型进行解剖结构分割
- 保存可视化结果
"""

import os
import sys
import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# ========== 配置 ==========
IMG_PATH = os.path.join("tests", "00000001_000.png")
OUTPUT_DIR = "test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("TorchXRayVision 模型测试")
print("=" * 60)

# ========== 1. 加载并预处理图像 ==========
print(f"\n[1/4] 加载图像: {IMG_PATH}")
img = xrv.utils.load_image(IMG_PATH)
print(f"  原始图像形状: {img.shape}, 值范围: [{img.min():.1f}, {img.max():.1f}]")

transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])
img_224 = transform(img)
print(f"  预处理后形状: {img_224.shape}")

# ========== 2. 病理分类 ==========
print("\n[2/4] 加载 DenseNet121 分类模型 (densenet121-res224-all)...")
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

with torch.no_grad():
    img_tensor = torch.from_numpy(img_224).unsqueeze(0)
    preds = model(img_tensor).cpu().numpy()[0]

results = dict(zip(model.pathologies, preds))

# 按概率排序
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("\n  ┌─────────────────────────────────┬────────────┐")
print("  │ 病理 (Pathology)                │ 概率       │")
print("  ├─────────────────────────────────┼────────────┤")
for name, prob in sorted_results:
    bar = "█" * int(prob * 20)
    print(f"  │ {name:<32}│ {prob:.4f} {bar}")
print("  └─────────────────────────────────┴────────────┘")

# ========== 3. 分类结果可视化 ==========
print("\n[3/4] 生成分类结果可视化...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左: 原始图像
axes[0].imshow(img_224[0], cmap='gray')
axes[0].set_title("Input Chest X-Ray", fontsize=14)
axes[0].axis('off')

# 右: 分类结果柱状图
pathology_names = [name for name, _ in sorted_results]
probabilities = [prob for _, prob in sorted_results]
colors = ['#e74c3c' if p > 0.5 else '#3498db' if p > 0.3 else '#95a5a6' for p in probabilities]

bars = axes[1].barh(range(len(pathology_names)), probabilities, color=colors)
axes[1].set_yticks(range(len(pathology_names)))
axes[1].set_yticklabels(pathology_names, fontsize=9)
axes[1].set_xlabel("Probability", fontsize=12)
axes[1].set_title("Pathology Classification Results", fontsize=14)
axes[1].set_xlim(0, 1)
axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold=0.5')
axes[1].legend()
axes[1].invert_yaxis()

plt.tight_layout()
cls_path = os.path.join(OUTPUT_DIR, "classification_results.png")
plt.savefig(cls_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  分类结果已保存: {cls_path}")

# ========== 4. 解剖分割 ==========
print("\n[4/4] 加载 PSPNet 解剖分割模型...")
try:
    seg_model = xrv.baseline_models.chestx_det.PSPNet()
    seg_model.eval()

    # 分割模型需要 512x512 输入
    transform_seg = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(512)
    ])
    img_512 = transform_seg(img)

    with torch.no_grad():
        img_seg_tensor = torch.from_numpy(img_512).unsqueeze(0)
        seg_output = seg_model(img_seg_tensor)

    print(f"  分割输出形状: {seg_output.shape}")
    print(f"  分割目标: {seg_model.targets}")

    # 可视化分割结果
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # 第一个子图显示原图
    axes[0].imshow(img_512[0], cmap='gray')
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis('off')

    # 选取关键解剖结构可视化
    key_targets = ['Left Lung', 'Right Lung', 'Heart', 'Aorta',
                   'Mediastinum', 'Spine', 'Left Clavicle']
    for i, target_name in enumerate(key_targets):
        if target_name in seg_model.targets:
            idx = seg_model.targets.index(target_name)
            mask = seg_output[0, idx].cpu().numpy()
            axes[i + 1].imshow(img_512[0], cmap='gray', alpha=0.7)
            axes[i + 1].imshow(mask, cmap='jet', alpha=0.4)
            axes[i + 1].set_title(target_name, fontsize=11)
            axes[i + 1].axis('off')

    plt.suptitle("Anatomical Segmentation Results (PSPNet)", fontsize=16)
    plt.tight_layout()
    seg_path = os.path.join(OUTPUT_DIR, "segmentation_results.png")
    plt.savefig(seg_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  分割结果已保存: {seg_path}")

except Exception as e:
    print(f"  分割模型加载失败 (可能需要额外下载): {e}")

# ========== 完成 ==========
print("\n" + "=" * 60)
print("✅ 测试完成！结果保存在:", OUTPUT_DIR)
print("=" * 60)
