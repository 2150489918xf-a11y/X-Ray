"""
下载更多带标签的测试图片
来源: ieee8023/covid-chestxray-dataset (torchxrayvision 同作者, 开源)
+ NIH Clinical Center 公开样本
"""
import os, urllib.request, json, csv, io

TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
os.makedirs(TESTS_DIR, exist_ok=True)

# ── 精选测试集: 手动挑选覆盖不同病种的图片 ──
# 来源1: ieee8023/covid-chestxray-dataset (GitHub 公开)
GITHUB_BASE = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/"

# 来源2: NIH ChestX-ray14 公开样本 (镜像)
NIH_BASE = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/"

IMAGES_TO_DOWNLOAD = [
    # COVID-19 确诊 (PA view, X-ray)
    {
        "filename": "covid_case1_vietnam.jpeg",
        "url": GITHUB_BASE + "auntminnie-a-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg",
        "finding": "COVID-19 Pneumonia",
        "expected_diseases": ["Pneumonia", "Infiltration"],
        "expected_absent": [],
        "description": "65岁男性, COVID-19确诊, 左上肺浸润影 (Vietnam Cho Ray Hospital)",
        "source": "NEJM 10.1056/NEJMc2001272"
    },
    {
        "filename": "covid_case2_vietnam_progression.jpeg",
        "url": GITHUB_BASE + "auntminnie-b-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg",
        "finding": "COVID-19 Pneumonia (progression)",
        "expected_diseases": ["Pneumonia", "Consolidation", "Infiltration"],
        "expected_absent": [],
        "description": "同一患者3天后, 浸润扩大+实变",
        "source": "NEJM 10.1056/NEJMc2001272"
    },
    # ARDS (严重呼吸窘迫)
    {
        "filename": "ards_case.jpeg",
        "url": GITHUB_BASE + "auntminnie-d-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg",
        "finding": "ARDS",
        "expected_diseases": ["Pneumonia", "Consolidation"],
        "expected_absent": [],
        "description": "COVID-19进展为ARDS, 双肺弥漫性白肺",
        "source": "NEJM 10.1056/NEJMc2001272"
    },
    # Streptococcus 肺炎
    {
        "filename": "streptococcus_pneumonia.jpg",
        "url": GITHUB_BASE + "streptococcus-pneumoniae-702-background.jpg",
        "finding": "Streptococcus Pneumonia",
        "expected_diseases": ["Pneumonia", "Consolidation"],
        "expected_absent": [],
        "description": "链球菌肺炎, 典型大叶性实变",
        "source": "Radiopaedia"
    },
    # Pneumocystis (免疫低下患者)
    {
        "filename": "pneumocystis_pneumonia.png",
        "url": GITHUB_BASE + "pneumocystis-carinii-702-background.png",
        "finding": "Pneumocystis Pneumonia",
        "expected_diseases": ["Pneumonia", "Infiltration"],
        "expected_absent": [],
        "description": "卡氏肺孢子菌肺炎, 弥漫性间质浸润",
        "source": "Radiopaedia"
    },
    # SARS
    {
        "filename": "sars_case.jpeg",
        "url": GITHUB_BASE + "sars-10.1148rg.242035193-g04mr34g04a.jpeg",
        "finding": "SARS",
        "expected_diseases": ["Pneumonia", "Infiltration"],
        "expected_absent": [],
        "description": "SARS患者, 双肺毛玻璃影",
        "source": "Radiology 10.1148/rg.242035193"
    },
    # MERS
    {
        "filename": "mers_case.png",
        "url": GITHUB_BASE + "nejmc2409399_f1-PA.png",
        "finding": "MERS",
        "expected_diseases": ["Pneumonia", "Consolidation"],
        "expected_absent": [],
        "description": "MERS患者, 右下肺实变",
        "source": "NEJM"
    },
    # 正常对照 (Radiopaedia)
    {
        "filename": "normal_radiopaedia.jpg",
        "url": GITHUB_BASE + "normal-362-background.jpg",
        "finding": "Normal",
        "expected_diseases": [],
        "expected_absent": ["Pneumonia", "Cardiomegaly", "Mass"],
        "description": "正常胸片, 不应检出异常",
        "source": "Radiopaedia"
    },
]


def download_images():
    """下载所有图片"""
    print(f"📥 开始下载 {len(IMAGES_TO_DOWNLOAD)} 张测试图片...")
    print(f"   目标目录: {TESTS_DIR}\n")

    downloaded = []
    failed = []

    for i, img in enumerate(IMAGES_TO_DOWNLOAD, 1):
        dst = os.path.join(TESTS_DIR, img['filename'])
        if os.path.exists(dst):
            print(f"  [{i}/{len(IMAGES_TO_DOWNLOAD)}] ✅ 已存在: {img['filename']}")
            downloaded.append(img)
            continue

        print(f"  [{i}/{len(IMAGES_TO_DOWNLOAD)}] 下载: {img['filename']}")
        print(f"       {img['description']}")
        try:
            urllib.request.urlretrieve(img['url'], dst)
            size_kb = os.path.getsize(dst) / 1024
            print(f"       ✅ 成功 ({size_kb:.0f} KB)")
            downloaded.append(img)
        except Exception as e:
            print(f"       ❌ 失败: {e}")
            failed.append(img)

    # 更新 ground_truth.json
    gt_path = os.path.join(TESTS_DIR, "ground_truth.json")
    if os.path.exists(gt_path):
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt = json.load(f)
    else:
        gt = {"description": "测试集 ground truth", "source_notes": {}, "cases": []}

    existing_images = {c['image'] for c in gt['cases']}

    for img in downloaded:
        if img['filename'] not in existing_images:
            gt['cases'].append({
                "image": img['filename'],
                "expected_diseases": img['expected_diseases'],
                "expected_absent": img['expected_absent'],
                "description": img['description'],
                "source": img['source']
            })
            gt['source_notes'][img['filename']] = f"{img['finding']} - {img['source']}"

    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"  下载完成: {len(downloaded)} 成功, {len(failed)} 失败")
    print(f"  ground_truth.json 已更新: {len(gt['cases'])} 个案例")
    print(f"{'='*50}")

    if failed:
        print(f"\n  失败列表:")
        for f_item in failed:
            print(f"    - {f_item['filename']}: {f_item['url']}")


if __name__ == "__main__":
    download_images()
