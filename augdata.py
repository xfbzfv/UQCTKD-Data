import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import albumentations as A

# ===================== 【请根据你的情况修改这里】 =====================
ORIGINAL_ROOT = r"D:\yjs\12-1\yolo-main\splitdata"  # 你的原始总目录（包含train/val/test）
AUG_ROOT = r"D:\yjs\12-1\yolo-main\splitdata2-aug"  # 增强后总目录（自动创建）
AUG_NUM_PER_IMAGE = 5  # 每张图生成的增强样本数
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']  # 东北大学数据集6类，可修改
COPY_RAW_DATA = True  # 是否保留原始图片/标签（建议True，避免丢失原始数据）
# ======================================================================

# 东北大学钢材缺陷数据集默认是6类，若你的类别数不同，修改上方CLASSES即可
NC = len(CLASSES)


# ===================== 1. 定义适配YOLO的增强管道（同步增强图片+标签） =====================
def get_yolo_aug_pipeline():
    return A.Compose([
        # 几何变换（不破坏缺陷特征）
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
            p=0.5, border_mode=cv2.BORDER_REPLICATE
        ),
        # 像素变换（模拟工业场景）
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussNoise(p=0.2),
        A.MotionBlur(p=0.1, blur_limit=3),
    ], bbox_params=A.BboxParams(
        format='yolo',  # 适配YOLO的bbox格式（归一化x_center,y_center,w,h）
        label_fields=['class_labels'],  # 关联类别标签
        min_area=10,  # 过滤增强后过小的bbox（避免无效标签）
        min_visibility=0.5  # 过滤遮挡过多的bbox
    ))


# ===================== 2. 复制原始test集（测试集不增强） =====================
def copy_test_set():
    src_test = os.path.join(ORIGINAL_ROOT, "test")
    dst_test = os.path.join(AUG_ROOT, "test")
    if os.path.exists(src_test):
        shutil.copytree(src_test, dst_test, dirs_exist_ok=True)
        print(f"已复制原始test集到：{dst_test}")
    else:
        print("警告：未找到原始test集，跳过复制")


# ===================== 3. 增强train/val集（同步处理图片+标签） =====================
def augment_train_val(split_name):
    """处理train/val的增强：split_name = "train" 或 "val" """
    # 原始路径
    src_split_dir = os.path.join(ORIGINAL_ROOT, split_name)
    src_img_dir = os.path.join(src_split_dir, "images")
    src_lab_dir = os.path.join(src_split_dir, "labels")
    if not os.path.exists(src_img_dir) or not os.path.exists(src_lab_dir):
        print(f"跳过{split_name}：未找到images/labels文件夹")
        return

    # 增强后路径
    dst_split_dir = os.path.join(AUG_ROOT, split_name)
    dst_img_dir = os.path.join(dst_split_dir, "images")
    dst_lab_dir = os.path.join(dst_split_dir, "labels")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lab_dir, exist_ok=True)

    # 1. 先复制原始图片/标签（可选）
    if COPY_RAW_DATA:
        shutil.copytree(src_img_dir, dst_img_dir, dirs_exist_ok=True)
        shutil.copytree(src_lab_dir, dst_lab_dir, dirs_exist_ok=True)
        print(f"已复制{split_name}原始数据到增强目录")

    # 2. 遍历图片进行增强
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(img_extensions)]
    pbar = tqdm(img_files, desc=f"增强{split_name}集")

    for img_file in pbar:
        # 读取图片
        img_path = os.path.join(src_img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过损坏图片：{img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        # 读取对应标签（YOLO格式：class_id x_center y_center width height）
        lab_file = os.path.splitext(img_file)[0] + ".txt"
        lab_path = os.path.join(src_lab_dir, lab_file)
        if not os.path.exists(lab_path):
            print(f"跳过无标签图片：{img_file}（未找到{lab_file}）")
            continue
        
        with open(lab_path, "r", encoding="utf-8") as f:
            lab_lines = f.readlines()
        bboxes = []
        class_labels = []
        for line in lab_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"标签格式错误：{lab_path}（行：{line}）")
                continue
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:5]))
            bboxes.append(bbox)
            class_labels.append(class_id)

        # 生成增强样本
        aug_pipeline = get_yolo_aug_pipeline()
        for aug_idx in range(AUG_NUM_PER_IMAGE):
            # 应用增强
            augmented = aug_pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_classes = augmented["class_labels"]

            # 转换为CV2格式并保存图片
            aug_img = cv2.cvtColor((aug_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            aug_img_name = f"{os.path.splitext(img_file)[0]}_aug_{aug_idx+1}{os.path.splitext(img_file)[1]}"
            aug_img_path = os.path.join(dst_img_dir, aug_img_name)
            cv2.imwrite(aug_img_path, aug_img)

            # 保存增强后的标签（保持YOLO格式）
            aug_lab_name = f"{os.path.splitext(lab_file)[0]}_aug_{aug_idx+1}.txt"
            aug_lab_path = os.path.join(dst_lab_dir, aug_lab_name)
            with open(aug_lab_path, "w", encoding="utf-8") as f:
                for cls, bbox in zip(aug_classes, aug_bboxes):
                    # 确保bbox在0-1范围内（增强可能轻微越界，裁剪）
                    bbox = [max(0, min(1, x)) for x in bbox]
                    f.write(f"{cls} {' '.join(map(str, bbox))}\n")

    pbar.close()
    print(f"{split_name}集增强完成：{len(img_files)}张原始图 → 新增{len(img_files)*AUG_NUM_PER_IMAGE}张增强图")


# ===================== 4. 生成data.yaml和classes.txt =====================
def generate_config_files():
    # 生成classes.txt（每行一个类别）
    classes_path = os.path.join(AUG_ROOT, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        for cls in CLASSES:
            f.write(f"{cls}\n")
    print(f"已生成类别文件：{classes_path}")

    # 生成data.yaml（YOLO训练配置）
    yaml_content = f"""train: ./train/images
val: ./val/images
test: ./test/images

nc: {NC}
names: {CLASSES}
"""
    yaml_path = os.path.join(AUG_ROOT, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"已生成配置文件：{yaml_path}")


# ===================== 执行主流程 =====================
if __name__ == "__main__":
    # 检查原始目录是否存在
    if not os.path.exists(ORIGINAL_ROOT):
        raise FileNotFoundError(f"原始总目录不存在：{ORIGINAL_ROOT}")
    
    # 创建增强后总目录
    os.makedirs(AUG_ROOT, exist_ok=True)

    # 复制test集
    copy_test_set()

    # 增强train和val集
    augment_train_val("train")
    augment_train_val("val")

    # 生成配置文件
    generate_config_files()

    print(f"\n所有操作完成！增强后数据集目录：{AUG_ROOT}")