import torch
import json
import shutil
import yaml
import argparse
import subprocess
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2 # 导入 OpenCV 用于图像处理
import numpy as np # 导入 NumPy 用于数组操作

# --- 注意：convert_coco_to_yolo 和 create_dataset_yaml 函数保持不变 ---
# 由于您在 main 中注释掉了调用，这里暂时保留函数定义但不使用
def convert_coco_to_yolo(coco_json_path, images_dir, output_labels_dir, class_mapping):
    """
    将 COCO JSON 格式的标注转换为 YOLO TXT 格式。
    :param coco_json_path: COCO JSON 文件路径
    :param images_dir: 对应的图片目录
    :param output_labels_dir: 输出的 YOLO TXT 标签文件目录
    :param class_mapping: COCO 类别 ID 到 YOLO 类别 ID 的映射字典
    """
    os.makedirs(output_labels_dir, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 创建一个 image_id 到 file_name 和 size 的映射
    image_info = {img['id']: {'file_name': img['file_name'], 'width': img['width'], 'height': img['height']} for img in coco_data['images']}

    # 初始化一个字典来存储每个 image_id 的 annotations
    annotations_dict = {img_id: [] for img_id in image_info.keys()}
    for ann in coco_data['annotations']:
        annotations_dict[ann['image_id']].append(ann)

    for image_id, anns in annotations_dict.items():
        img_info = image_info[image_id]
        img_width = img_info['width']
        img_height = img_info['height']
        txt_filename = os.path.splitext(img_info['file_name'])[0] + '.txt'
        txt_path = os.path.join(output_labels_dir, txt_filename)

        with open(txt_path, 'w') as f:
            for ann in anns:
                category_id = ann['category_id']
                if category_id not in class_mapping:
                    print(f"警告: 发现未映射的 category_id {category_id} 在 image_id {image_id} 中，已跳过。")
                    continue
                yolo_class_id = class_mapping[category_id]

                # COCO bbox format: [x_min, y_min, width, height]
                bbox = ann['bbox']
                x_min, y_min, width, height = bbox

                # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
                x_center = (x_min + width / 2) / img_width
                y_center = (y_min + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height

                f.write(f"{yolo_class_id} {x_center} {y_center} {norm_width} {norm_height}\n")


def create_dataset_yaml(dataset_path, output_yaml_path, class_names):
    """
    创建 YOLOv5 训练所需的 dataset.yaml 配置文件。
    :param dataset_path: 数据集根目录 (包含 train, valid, test)
    :param output_yaml_path: 输出的 .yaml 文件路径
    :param class_names: 类别名称列表
    """
    dataset_path = Path(dataset_path).resolve()
    yaml_content = {
        'path': str(dataset_path),  # dataset root dir
        'train': 'train/images',  # 相对于 path 的路径
        'val': 'valid/images',    # 相对于 path 的路径
        'test': 'test/images',    # 相对于 path 的路径

        # Classes
        'nc': len(class_names),  # number of classes
        'names': class_names     # class names
    }

    with open(output_yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    print(f"数据集配置文件已创建: {output_yaml_path}")


def run_command(command, cwd=None):
    """执行 shell 命令并实时打印输出"""
    print(f"执行命令: {' '.join(command)}")
    try:
        # 修改为实时输出，以便在 VS Code 终端看到训练过程
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in process.stdout:
            print(line, end='') # 实时打印输出
        process.wait() # 等待进程结束
        if process.returncode != 0:
             raise subprocess.CalledProcessError(process.returncode, command)
        # 原来的 run 方法也可以，但不实时显示
        # result = subprocess.run(command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {' '.join(e.cmd)}")
        print(e.output)
        raise

def plot_results(csv_path, plot_path):
    """
    从 results.csv 读取数据并绘制图表
    :param csv_path: results.csv 文件路径
    :param plot_path: 保存图表的路径
    """
    try:
        # 读取 CSV 文件
        # YOLOv5 的 CSV 文件可能以 '#' 开头的注释行，pandas 通常能处理，但最好明确
        # 如果第一行是注释，pandas.read_csv 会自动跳过（如果 sep 正确）
        # 但为了保险，可以指定 header=0 或者手动处理
        # 检查第一行是否是注释
        with open(csv_path, 'r') as f:
            first_line = f.readline()
        header_row = 0
        if first_line.startswith('#'):
            header_row = 1 # 实际标题在第二行

        data = pd.read_csv(csv_path, skiprows=header_row)
        # print(data.head()) # 可以打印前几行检查列名

        # 清理列名（去除可能的前缀空格）
        data.columns = data.columns.str.strip()

        # 假设列名是固定的 (根据 YOLOv5 v6.0+)
        # train/box_loss, train/obj_loss, train/cls_loss, metrics/precision, metrics/recall, metrics/mAP_0.5, metrics/mAP_0.5:0.95, val/box_loss, val/obj_loss, val/cls_loss, x/lr0, x/lr1, x/lr2
        epochs = data['epoch']
        precision = data['metrics/precision']
        recall = data['metrics/recall']
        mAP_0_5 = data['metrics/mAP_0.5']
        mAP_0_5_0_95 = data['metrics/mAP_0.5:0.95']

        # 创建图表
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, precision, label='Precision')
        plt.title('Precision vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(epochs, recall, label='Recall', color='orange')
        plt.title('Recall vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(epochs, mAP_0_5, label='mAP@0.5', color='green')
        plt.title('mAP@0.5 vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('mAP@0.5')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(epochs, mAP_0_5_0_95, label='mAP@0.5:0.95', color='red')
        plt.title('mAP@0.5:0.95 vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('mAP@0.5:0.95')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"训练指标图表已保存至: {plot_path}")
        plt.show() # 在 VS Code 中会弹出图表窗口

    except FileNotFoundError:
        print(f"警告: 未找到 results.csv 文件 ({csv_path})，无法绘制图表。")
    except KeyError as e:
        print(f"错误: results.csv 中缺少预期的列 {e}，请检查 YOLOv5 版本或 CSV 格式。")
    except Exception as e:
        print(f"绘制图表时发生错误: {e}")

def preprocess_image_opencv(image_path, output_path):
    """
    使用 OpenCV 对单张图片进行预处理。
    步骤：直方图均衡化 -> 高斯滤波 -> 锐化
    :param image_path: 原始图片路径
    :param output_path: 处理后图片保存路径
    """
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"警告: 无法读取图片 {image_path}")
        return

    # 1. 直方图均衡化 (增强对比度)
    # 转换到 YUV 色彩空间，对 Y 通道进行均衡化
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # 2. 高斯滤波 (轻微降噪)
    kernel_size = (5, 5)
    sigma = 0 # 让 OpenCV 自动计算
    img_filtered = cv2.GaussianBlur(img_eq, kernel_size, sigma)

    # 3. 锐化 (可选，可能会增加噪声)
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    img_sharpened = cv2.filter2D(img_filtered, -1, kernel_sharpen)

    # 保存处理后的图片 (这里保存的是最终锐化后的图片)
    cv2.imwrite(str(output_path), img_sharpened)
    # print(f"已处理并保存图片: {output_path}")

def preprocess_dataset_opencv(dataset_split_dir, output_split_dir):
    """
    对整个数据集划分 (train/valid/test) 的图片进行预处理。
    :param dataset_split_dir: 包含 images/ 和 labels/ 的原始数据集划分目录
    :param output_split_dir: 保存处理后 images/ 和 labels/ 的输出目录
    """
    images_dir = dataset_split_dir / "images"
    labels_dir = dataset_split_dir / "labels"
    output_images_dir = output_split_dir / "images"
    output_labels_dir = output_split_dir / "labels"

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # 复制 labels
    if labels_dir.exists():
        for label_file in labels_dir.iterdir():
            if label_file.is_file() and label_file.suffix.lower() == '.txt':
                shutil.copy(label_file, output_labels_dir / label_file.name)
        print(f"Labels 已复制到 {output_labels_dir}")

    # 处理并复制 images
    if images_dir.exists():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for image_file in images_dir.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                output_image_path = output_images_dir / image_file.name
                preprocess_image_opencv(image_file, output_image_path)
        print(f"Images 已处理并保存到 {output_images_dir}")

def main(args):
    project_root = Path(__file__).parent.resolve()
    yolov5_dir = project_root / "yolov5"
    dataset_dir = project_root / "dataset"
    models_dir = project_root / "models"
    output_dir = project_root / "runs" # YOLOv5 默认输出目录

    
    '''
    # --- 1. 转换标注格式 ---
    print("--- 步骤 1: 转换 COCO 标注到 YOLO 格式 ---")
    splits = ['train', 'valid', 'test']
    # 假设所有 splits 的类别 ID 映射是一致的，我们从 train 数据中获取
    train_coco_json = dataset_dir / "train" / "_annotations.coco.json"
    with open(train_coco_json, 'r') as f:
        train_coco_data = json.load(f)
    # 假设类别 "bees" 对应 YOLO 类别 ID 0
    class_mapping = {1: 0}  # COCO category_id -> YOLO class_id
    class_names = ["bees"]  # 你的类别名称列表

    for split in splits:
        coco_json_path = dataset_dir / split / "_annotations.coco.json"
        images_dir = dataset_dir / split
        output_labels_dir = dataset_dir / split / "labels"
        output_images_dir = dataset_dir / split / "images"  # 新增: images 子目录

        if not coco_json_path.exists():
            print(f"警告: 找不到 {coco_json_path}，跳过 {split} 集转换。")
            continue

        print(f"正在转换 {split} 集...")

    # 创建 labels 和 images 目录
        os.makedirs(output_labels_dir, exist_ok=True)
        os.makedirs(output_images_dir, exist_ok=True)

    # 将该 split 下的所有图片文件复制到 images/ 子目录
    # 获取所有图片文件（假设是 .jpg, .jpeg, .png）
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for file_path in images_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # 复制到 images/ 目录
                dest = output_images_dir / file_path.name
                if not dest.exists():
                    shutil.copy(file_path, dest)

    # 转换标注
        convert_coco_to_yolo(coco_json_path, images_dir, output_labels_dir, class_mapping)

    print("✅ 图片已复制到 images/ 子目录，标注已转换为 YOLO 格式。")

    '''
    
    dataset_yaml_path = project_root / "bees_dataset.yaml"
    
    '''
    
    # --- 2. 创建数据集配置文件 ---
    print("\n--- 步骤 2: 创建数据集配置文件 ---")

    create_dataset_yaml(dataset_dir, dataset_yaml_path, class_names)
    
    '''


    # --- 3. 训练模型 ---
    latest_train_run_dir = None # 用于存储最新的训练 run 目录
    if args.train:
        print("\n--- 步骤 3: 开始训练模型 (含图像预处理) ---")
        
        '''

        # --- 新增: 图像预处理步骤 ---
        print("\n--- 步骤 3a: 对训练集和验证集进行图像预处理 ---")
        # 定义预处理后数据集的存放路径
        processed_dataset_dir = project_root / "processed_dataset"
        os.makedirs(processed_dataset_dir, exist_ok=True)
        
        splits_to_process = ['train', 'valid'] # 通常只对训练集和验证集做预处理
        for split in splits_to_process:
            original_split_dir = dataset_dir / split
            processed_split_dir = processed_dataset_dir / split
            if original_split_dir.exists():
                print(f"正在处理 {split} 集...")
                preprocess_dataset_opencv(original_split_dir, processed_split_dir)
        
        # 创建一个指向预处理后数据集的临时 dataset.yaml
        temp_dataset_yaml_path = project_root / "temp_processed_bees_dataset.yaml"
        # 注意：这里需要根据你的实际类别名称修改
        class_names = ["bees"] # 假设类别名称是 bees
        create_dataset_yaml(processed_dataset_dir, temp_dataset_yaml_path, class_names)
        print(f"已创建临时预处理数据集配置文件: {temp_dataset_yaml_path}")
        # --- 预处理步骤结束 ---

        '''

        #  temp_dataset_yaml_path = project_root / "temp_processed_bees_dataset.yaml"

        weights_path = models_dir / "yolov5m.pt"
        #  weights_path = models_dir / "custom_yolov5m.pt"  
        if not weights_path.exists():
            print(f"警告: 预训练权重 {weights_path} 不存在。YOLOv5 可能会自动下载。")
            # 你可以选择提供 yolov5m.pt 的下载链接或手动放置
            # 这里我们仍然尝试使用它，YOLOv5 脚本会处理不存在的情况
            weights_path = "yolov5m.pt" # 如果 custom_yolov5m.pt 不存在，则尝试默认的

        # 修改训练命令，使用预处理后的数据集配置文件
        train_cmd = [
            "python", "train.py",
            "--img", "416", # 根据你的图片尺寸设置
            "--batch", str(args.batch_size),
            "--epochs", str(args.epochs), # 使用传入的 epochs 参数
            "--data", str(dataset_yaml_path), # 使用原始数据集
            # "--data", str(temp_dataset_yaml_path), # 使用预处理后的数据集
            "--weights", str(weights_path),
            "--name", "bees_yolov5m_train_processed" ,# 训练 run 的名称，添加后缀以区分
            "--device", "0"  # 使用 GPU
        ]
        if args.resume:
            train_cmd.append("--resume")

        run_command(train_cmd, cwd=str(yolov5_dir))

        # --- 4. 绘制训练指标 ---
        # 训练完成后查找并绘制图表
        train_runs_dir = yolov5_dir / "runs" / "train"
        if train_runs_dir.exists():
            train_run_dirs = [d for d in train_runs_dir.iterdir() if d.is_dir()]
            if train_run_dirs:
                # 找到最新的 run 目录 (按修改时间)
                latest_train_run_dir = max(train_run_dirs, key=os.path.getmtime)
                results_csv_path = latest_train_run_dir / "results.csv"
                plot_save_path = latest_train_run_dir / "training_curves.png"
                if results_csv_path.exists():
                    plot_results(str(results_csv_path), str(plot_save_path))
                else:
                    print(f"警告: 在 {latest_train_run_dir} 中未找到 results.csv 文件。")
            else:
                print("未找到训练输出目录。")
        else:
             print("未找到训练输出目录。")



    # --- 5. 验证模型 ---
    if args.val:
        print("\n--- 步骤 5: 验证模型 ---")
        # 确定要验证的模型权重路径
        val_weights_path = None
        if args.val_weights:
            val_weights_path = Path(args.val_weights)
        elif args.train and latest_train_run_dir: # 如果执行了训练，通常验证刚刚训练完的模型
            # 使用在训练步骤中找到的 latest_train_run_dir
            val_weights_path = latest_train_run_dir / "weights" / "best.pt"
            if not val_weights_path.exists():
                val_weights_path = latest_train_run_dir / "weights" / "last.pt"
        else: # 如果没有刚训练完的，或者没有执行训练，则主动查找最新的
            # 优先查找 yolov5/runs/train 下最新的 bees_yolov5m_train* 目录
            train_runs_base_dir = yolov5_dir / "runs" / "train"
            if train_runs_base_dir.exists():
                # 使用 glob 查找所有匹配 bees_yolov5m_train* 的目录
                matching_train_dirs = list(train_runs_base_dir.glob("bees_yolov5m_train*"))
                if matching_train_dirs:
                    # 根据修改时间找出最新的目录
                    latest_matching_train_run = max(matching_train_dirs, key=lambda p: p.stat().st_mtime)
                    potential_weights = [
                        latest_matching_train_run / "weights" / "best.pt",
                        latest_matching_train_run / "weights" / "last.pt"
                    ]
                    # 按优先级查找存在的权重文件
                    for pw in potential_weights:
                        if pw.exists():
                            val_weights_path = pw
                            break
                    if val_weights_path is None:
                         print(f"警告: 在最新的匹配目录 {latest_matching_train_run} 中未找到 best.pt 或 last.pt。")
                else:
                    print("未找到匹配 'bees_yolov5m_train*' 模式的训练输出目录。")

        # 如果以上都没找到，则回退到 models/custom_yolov5m.pt
        if val_weights_path is None:
            val_weights_path = models_dir / "custom_yolov5m.pt"
            print(f"未找到特定训练权重，回退到默认权重路径: {val_weights_path}")

        # 最终检查权重文件是否存在
        if not val_weights_path.exists():
            print(f"错误: 要验证的模型权重 {val_weights_path} 不存在。")
            return

        print(f"使用模型权重进行验证: {val_weights_path}")

        # 注意：验证时通常使用原始的、未经预处理的测试集
        val_cmd = [
            "python", "val.py",
            "--data", str(dataset_yaml_path), # 验证时使用原始数据集配置
            "--weights", str(val_weights_path),
            "--img", "416", # 验证图片尺寸
            "--name", "bees_yolov5m_val_processed", # 验证 run 的名称，添加后缀以区分
            "--device", "0"
        ]
        run_command(val_cmd, cwd=str(yolov5_dir))


    print("\n--- 所有步骤完成 ---")
    print(f"输出结果可在 {output_dir} 中找到。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 YOLOv5 训练和验证流程")
    # 修改默认行为：如果不提供 --train 和 --val，则默认执行 --train
    parser.add_argument("--train", action="store_true", help="执行训练")
    parser.add_argument("--val", action="store_true", help="执行验证")
    # 设置默认值
    parser.add_argument("--batch_size", type=int, default=6, help="训练批次大小 (默认: 6)")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数 (默认: 30)")
    parser.add_argument("--resume", action="store_true", help="从最后检查点恢复训练")
    parser.add_argument("--val_weights", type=str, help="用于验证的模型权重路径 (例如: runs/train/exp/weights/best.pt)")
    args = parser.parse_args()

    # 如果既没有指定 --train 也没有指定 --val，则默认执行训练
    if not args.train and not args.val:
        print("未指定 --train 或 --val，将默认执行训练 (--train)。")
        args.train = True

    main(args)



