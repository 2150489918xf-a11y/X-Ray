"""
批量评估脚本 — 用 ground_truth.json 测试 Pipeline v2
输出: 每张图的检测结果 vs 标准答案, 统计分类准确率
"""
import os, sys, json
import numpy as np
import cv2
import torch
import torchvision
import torchxrayvision as xrv
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 导入 pipeline 函数 ──
from pipeline_v2 import (
    stage1_classify, stage2_lung_mask, stage3_target_extraction,
    stage4_route_diffuse, stage4_route_solid,
    nms_merge_findings, generate_master_canvas, generate_report,
    POSITIVE_THRESHOLD, PATHOLOGY_CN, banner,
    OUTPUT_DIR
)

TESTS_DIR = os.path.join(os.path.dirname(__file__), "tests")
GT_PATH = os.path.join(TESTS_DIR, "ground_truth.json")
EVAL_OUTPUT = os.path.join(os.path.dirname(__file__), "test_output", "evaluation")


def run_evaluation():
    """对每张测试图运行 pipeline, 与 ground truth 对比"""
    os.makedirs(EVAL_OUTPUT, exist_ok=True)

    with open(GT_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    cases = gt_data['cases']
    all_eval = []

    print("=" * 60)
    print("  Pipeline v2 批量评估")
    print("=" * 60)

    for ci, case in enumerate(cases, 1):
        img_file = case['image']
        img_path = os.path.join(TESTS_DIR, img_file)

        if not os.path.exists(img_path):
            print(f"\n⚠️ 跳过 {img_file} (文件不存在)")
            continue

        print(f"\n{'━'*60}")
        print(f"  案例 {ci}/{len(cases)}: {img_file}")
        print(f"  预期: {case['expected_diseases'] or '无异常'}")
        print(f"  说明: {case['description']}")
        print(f"{'━'*60}")

        # 运行第一阶
        # 需要临时修改 IMG_PATH
        import pipeline_v2
        old_img = pipeline_v2.IMG_PATH
        old_output = pipeline_v2.OUTPUT_DIR
        pipeline_v2.IMG_PATH = img_path

        case_name = os.path.splitext(img_file)[0]
        case_output = os.path.join(EVAL_OUTPUT, case_name)
        os.makedirs(case_output, exist_ok=True)
        pipeline_v2.OUTPUT_DIR = case_output

        result = stage1_classify(img_path)

        if result[0] is None:
            # 未检出任何异常
            all_results = result[-1]
            detected = []
            print(f"  结果: 未见明显异常")

            eval_item = {
                'image': img_file,
                'expected': case['expected_diseases'],
                'expected_absent': case['expected_absent'],
                'detected': [],
                'all_probs': {k: float(round(v, 4)) for k, v in all_results.items()},
                'hits': [],
                'misses': case['expected_diseases'][:],
                'false_positives': [],
                'correct': len(case['expected_diseases']) == 0
            }
        else:
            model, img_tensor, img_224, img_xrv, img_rgb, all_results, targets = result
            detected = [t['name'] for t in targets]

            # 继续跑完整流水线
            lung_mask_224, lung_mask_512, anatomy_masks_512 = stage2_lung_mask(img_xrv)

            all_target_results = []
            for i, target in enumerate(targets):
                cam_raw, clean_cam, bboxes_224 = stage3_target_extraction(
                    model, img_tensor, img_224, lung_mask_224, target['name'])

                if target['disease_type'] == "solid":
                    s4 = stage4_route_solid(
                        img_rgb, clean_cam, bboxes_224, target['name'], anatomy_masks_512)
                else:
                    s4 = stage4_route_diffuse(
                        img_rgb, img_224, clean_cam, bboxes_224, target['name'], anatomy_masks_512)

                s4['is_suspect'] = target.get('suspect', False)
                all_target_results.append({
                    'target': target, 'cam_raw': cam_raw,
                    'clean_cam': clean_cam, 'bboxes_224': bboxes_224, 'stage4': s4
                })

            findings = nms_merge_findings(all_target_results, img_rgb.shape)
            generate_master_canvas(img_rgb, findings, lung_mask_224, all_results)
            generate_report(findings, all_results, img_rgb.shape)

            # 评估
            hits = [d for d in case['expected_diseases'] if d in detected]
            misses = [d for d in case['expected_diseases'] if d not in detected]
            false_positives = [d for d in detected if d in case['expected_absent']]

            eval_item = {
                'image': img_file,
                'expected': case['expected_diseases'],
                'expected_absent': case['expected_absent'],
                'detected': detected,
                'all_probs': {k: float(round(v, 4)) for k, v in all_results.items()},
                'hits': hits,
                'misses': misses,
                'false_positives': false_positives,
                'correct': len(misses) == 0 and len(false_positives) == 0
            }

        all_eval.append(eval_item)

        # 打印结果
        print(f"\n  📊 评估结果:")
        print(f"    检测到: {eval_item['detected'] or '无'}")
        print(f"    命中 ✅: {eval_item['hits'] or '无'}")
        print(f"    漏检 ❌: {eval_item['misses'] or '无'}")
        print(f"    误检 ⚠️: {eval_item['false_positives'] or '无'}")
        print(f"    整体: {'✅ 正确' if eval_item['correct'] else '❌ 有误'}")

        # 恢复
        pipeline_v2.IMG_PATH = old_img
        pipeline_v2.OUTPUT_DIR = old_output

    # ── 汇总 ──
    print(f"\n\n{'='*60}")
    print(f"  📋 评估汇总")
    print(f"{'='*60}")

    n_total = len(all_eval)
    n_correct = sum(1 for e in all_eval if e['correct'])
    all_hits = sum(len(e['hits']) for e in all_eval)
    all_expected = sum(len(e['expected']) for e in all_eval)
    all_misses = sum(len(e['misses']) for e in all_eval)
    all_fp = sum(len(e['false_positives']) for e in all_eval)

    sensitivity = all_hits / max(all_expected, 1)
    accuracy = n_correct / max(n_total, 1)

    print(f"  案例总数: {n_total}")
    print(f"  完全正确: {n_correct}/{n_total} ({accuracy:.0%})")
    print(f"  疾病命中: {all_hits}/{all_expected} (灵敏度={sensitivity:.0%})")
    print(f"  总漏检数: {all_misses}")
    print(f"  总误检数: {all_fp}")

    summary = {
        'total_cases': n_total,
        'correct_cases': n_correct,
        'accuracy': round(accuracy, 4),
        'sensitivity': round(sensitivity, 4),
        'total_hits': all_hits,
        'total_expected': all_expected,
        'total_misses': all_misses,
        'total_false_positives': all_fp,
        'cases': all_eval
    }

    summary_path = os.path.join(EVAL_OUTPUT, "evaluation_report.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 评估报告: {summary_path}")

    # 生成汇总图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 每个案例结果
    case_names = [e['image'].split('.')[0][:15] for e in all_eval]
    case_colors = ['#2ecc71' if e['correct'] else '#e74c3c' for e in all_eval]
    axes[0].barh(case_names, [1]*n_total, color=case_colors)
    axes[0].set_title(f"案例结果 ({n_correct}/{n_total} 正确)", fontsize=13)
    axes[0].set_xlim(0, 1.2)
    for i, e in enumerate(all_eval):
        label = "✅" if e['correct'] else f"❌ 漏{len(e['misses'])} 误{len(e['false_positives'])}"
        axes[0].text(1.05, i, label, va='center', fontsize=10)

    # 右: 指标
    metrics = ['准确率', '灵敏度']
    values = [accuracy, sensitivity]
    bar_colors = ['#3498db', '#e67e22']
    axes[1].bar(metrics, values, color=bar_colors, width=0.5)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("评估指标", fontsize=13)
    for i, v in enumerate(values):
        axes[1].text(i, v + 0.03, f"{v:.0%}", ha='center', fontsize=14, fontweight='bold')

    plt.suptitle("Pipeline v2 批量评估结果", fontsize=15, fontweight='bold')
    plt.tight_layout()
    chart_path = os.path.join(EVAL_OUTPUT, "evaluation_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 评估图表: {chart_path}")


if __name__ == "__main__":
    run_evaluation()
