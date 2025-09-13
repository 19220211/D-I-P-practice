import os
import glob
from pathlib import Path
from datetime import datetime
import torch
import subprocess
import shutil
import json
import math
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename # 使用更安全的 secure_filename

app = Flask(__name__)
app.secret_key = 'a_very_secret_key_for_flask_sessions' # 用于 flash 消息

# --- 配置 ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
DETECTION_OUTPUT_FOLDER = BASE_DIR / 'detections'
DATASET_DIR = BASE_DIR / 'dataset'
YOLOV5_DIR = BASE_DIR / 'yolov5'
MODELS_DIR = BASE_DIR / 'models'
TRAIN_RUNS_DIR = YOLOV5_DIR / 'runs' / 'train'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_OUTPUT_FOLDER, exist_ok=True)

# 更新允许的扩展名，添加视频格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}

# --- 辅助函数 ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_statistics(annotations_data, image_count):
    """根据 COCO 格式的 annotations_data 和 image_count 计算统计数据"""
    stats = {}
    if not annotations_data or image_count == 0:
        # 如果没有数据，返回默认值
        return {
            'image_count': 0,
            'annotation_count': 0,
            'avg_annotations_per_image': 0,
            'bee_counts_per_image': [],
            'avg_bees_per_image': 0,
            'std_bees_per_image': 0,
            'min_bees_per_image': 0,
            'max_bees_per_image': 0,
            'images_with_no_bees': 0,
            'category_counts': {},
            'bbox_widths': [],
            'bbox_heights': [],
            'bbox_areas': [],
            'bbox_centers_x': [],
            'bbox_centers_y': [],
            'bbox_aspect_ratios': []
        }

    annotations = annotations_data.get('annotations', [])
    images = annotations_data.get('images', [])
    categories = annotations_data.get('categories', [])

    annotation_count = len(annotations)
    stats['annotation_count'] = annotation_count
    stats['avg_annotations_per_image'] = annotation_count / image_count if image_count > 0 else 0

    # 统计每张图像的蜜蜂数量
    image_id_to_bee_count = Counter(ann['image_id'] for ann in annotations)
    bee_counts_per_image = list(image_id_to_bee_count.values())
    # 补齐没有蜜蜂的图像 (计数为0)
    all_image_ids = {img['id'] for img in images}
    bee_counts_per_image.extend([0] * (image_count - len(bee_counts_per_image)))
    
    stats['bee_counts_per_image'] = bee_counts_per_image
    stats['avg_bees_per_image'] = sum(bee_counts_per_image) / len(bee_counts_per_image) if bee_counts_per_image else 0
    stats['std_bees_per_image'] = (sum((x - stats['avg_bees_per_image'])**2 for x in bee_counts_per_image) / len(bee_counts_per_image)) ** 0.5 if bee_counts_per_image else 0
    stats['min_bees_per_image'] = min(bee_counts_per_image) if bee_counts_per_image else 0
    stats['max_bees_per_image'] = max(bee_counts_per_image) if bee_counts_per_image else 0
    stats['images_with_no_bees'] = bee_counts_per_image.count(0)

    # 类别频率
    category_id_to_name = {cat['id']: cat['name'] for cat in categories}
    category_counts = Counter(ann['category_id'] for ann in annotations)
    stats['category_counts'] = {category_id_to_name.get(cat_id, f"Unknown_{cat_id}"): count for cat_id, count in category_counts.items()}

    # 目标尺寸和位置
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    bbox_centers_x = []
    bbox_centers_y = []
    bbox_aspect_ratios = []

    for ann in annotations:
        bbox = ann['bbox'] # [x, y, width, height]
        w, h = bbox[2], bbox[3]
        bbox_widths.append(w)
        bbox_heights.append(h)
        bbox_areas.append(w * h)
        bbox_centers_x.append(bbox[0] + w / 2)
        bbox_centers_y.append(bbox[1] + h / 2)
        if h > 0: # 避免除以零
            bbox_aspect_ratios.append(w / h)
        else:
            bbox_aspect_ratios.append(0) # 或者可以选择跳过

    stats['bbox_widths'] = bbox_widths
    stats['bbox_heights'] = bbox_heights
    stats['bbox_areas'] = bbox_areas
    stats['bbox_centers_x'] = bbox_centers_x
    stats['bbox_centers_y'] = bbox_centers_y
    stats['bbox_aspect_ratios'] = bbox_aspect_ratios

    return stats


def get_dataset_stats():
    """获取数据集详细统计信息"""
    all_stats = {}
    overall_stats = {
        'total_images': 0,
        'total_annotations': 0,
        'total_bee_counts': [],
        'all_bbox_widths': [],
        'all_bbox_heights': [],
        'all_bbox_areas': [],
        'all_bbox_centers_x': [],
        'all_bbox_centers_y': [],
        'all_bbox_aspect_ratios': [],
        'all_category_counts': Counter()
    }

    for split in ['train', 'valid', 'test']:
        stats = {'name': split}
        image_dir = DATASET_DIR / split / 'images'
        annotation_file = DATASET_DIR / split / '_annotations.coco.json'
        
        stats['image_count'] = len(list(image_dir.glob('*'))) if image_dir.exists() else 0
        overall_stats['total_images'] += stats['image_count']

        if annotation_file.exists():
            try:
                with open(annotation_file, 'r') as f:
                    annotations_data = json.load(f)
            except Exception as e:
                print(f"Error loading annotation file {annotation_file}: {e}")
                annotations_data = {}
        else:
            annotations_data = {}

        detailed_stats = calculate_statistics(annotations_data, stats['image_count'])
        stats.update(detailed_stats)
        
        # 累积总体统计
        overall_stats['total_annotations'] += stats['annotation_count']
        overall_stats['total_bee_counts'].extend(stats['bee_counts_per_image'])
        overall_stats['all_bbox_widths'].extend(stats['bbox_widths'])
        overall_stats['all_bbox_heights'].extend(stats['bbox_heights'])
        overall_stats['all_bbox_areas'].extend(stats['bbox_areas'])
        overall_stats['all_bbox_centers_x'].extend(stats['bbox_centers_x'])
        overall_stats['all_bbox_centers_y'].extend(stats['bbox_centers_y'])
        overall_stats['all_bbox_aspect_ratios'].extend(stats['bbox_aspect_ratios'])
        overall_stats['all_category_counts'].update(stats['category_counts'])

        all_stats[split] = stats

    # 计算总体统计数据
    all_stats['overall'] = {}
    all_stats['overall']['total_images'] = overall_stats['total_images']
    all_stats['overall']['total_annotations'] = overall_stats['total_annotations']
    
    if overall_stats['total_images'] > 0:
        all_stats['overall']['avg_annotations_per_image'] = overall_stats['total_annotations'] / overall_stats['total_images']
        all_stats['overall']['avg_bees_per_image'] = sum(overall_stats['total_bee_counts']) / len(overall_stats['total_bee_counts'])
        all_stats['overall']['std_bees_per_image'] = (sum((x - all_stats['overall']['avg_bees_per_image'])**2 for x in overall_stats['total_bee_counts']) / len(overall_stats['total_bee_counts'])) ** 0.5
    else:
        all_stats['overall']['avg_annotations_per_image'] = 0
        all_stats['overall']['avg_bees_per_image'] = 0
        all_stats['overall']['std_bees_per_image'] = 0
        
    all_stats['overall']['min_bees_per_image'] = min(overall_stats['total_bee_counts']) if overall_stats['total_bee_counts'] else 0
    all_stats['overall']['max_bees_per_image'] = max(overall_stats['total_bee_counts']) if overall_stats['total_bee_counts'] else 0
    all_stats['overall']['images_with_no_bees'] = overall_stats['total_bee_counts'].count(0)
    
    all_stats['overall']['category_counts'] = dict(overall_stats['all_category_counts'])

    # 为了简化前端处理，可以将列表数据转换为字符串或摘要（这里保留列表，前端处理）
    # 例如，可以计算直方图bin或热力图网格，但为了灵活性，这里直接传递原始数据
    # 前端可以根据这些数据绘制图表

    return all_stats

def find_latest_model():
    """查找最新训练的模型权重"""
    if not TRAIN_RUNS_DIR.exists():
        print("警告: yolov5/runs/train 目录不存在。")
        return None

    # 查找所有以 bees_yolov5m_train 开头的目录
    train_dirs = [d for d in TRAIN_RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith('bees_yolov5m_train')]
    
    if not train_dirs:
        print("警告: 未在 yolov5/runs/train 中找到 bees_yolov5m_train* 目录。")
        return None

    # 按修改时间排序，取最新的
    latest_train_dir = max(train_dirs, key=os.path.getmtime)
    best_pt = latest_train_dir / 'weights' / 'best.pt'
    last_pt = latest_train_dir / 'weights' / 'last.pt'

    if best_pt.exists():
        print(f"找到最新训练的 best.pt 模型: {best_pt}")
        return best_pt
    elif last_pt.exists():
        print(f"找到最新训练的 last.pt 模型: {last_pt}")
        return last_pt
    else:
        print(f"警告: 在 {latest_train_dir} 中未找到 best.pt 或 last.pt。")
        return None

def get_model_path():
    """确定要使用的模型路径"""
    model_path = find_latest_model()
    if model_path and model_path.exists():
        print(f"使用最新训练的模型: {model_path}")
        return str(model_path)
    else:
        fallback_model = MODELS_DIR / 'custom_yolov5m.pt'
        if fallback_model.exists():
            print(f"未找到最新训练模型，使用回退模型: {fallback_model}")
            return str(fallback_model)
        else:
            error_msg = f"未找到任何可用的模型权重文件。请确保 {fallback_model} 存在。"
            print(f"错误: {error_msg}")
            raise FileNotFoundError(error_msg)

# --- 路由 ---

@app.route('/')
def index():
    """首页：显示数据集统计"""
    stats = get_dataset_stats()
    return render_template('index.html', stats=stats)

@app.route('/gallery')
def gallery():
    """图像浏览页面"""
    split = request.args.get('split', 'train') # 默认显示训练集
    page = request.args.get('page', 1, type=int)
    per_page = 20 # 每页显示的图像数量

    if split not in ['train', 'valid', 'test']:
        split = 'train'

    image_dir = DATASET_DIR / split / 'images'
    if not image_dir.exists():
        image_files = []
    else:
        image_files = sorted(list(image_dir.glob('*')))
    
    total_images = len(image_files)
    total_pages = (total_images + per_page - 1) // per_page if total_images > 0 else 1
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    images_on_page = image_files[start_idx:end_idx]

    # 生成相对路径供模板使用
    image_paths = [f'dataset/{split}/images/{img.name}' for img in images_on_page]

    return render_template(
        'gallery.html',
        images=image_paths,
        current_split=split,
        current_page=page,
        total_pages=total_pages,
        total_images=total_images
    )

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    """图像检测页面"""
    if request.method == 'POST':
        # 处理文件上传
        uploaded_files = request.files.getlist("image_files")
        if not uploaded_files or all(f.filename == '' for f in uploaded_files):
            flash('请选择至少一个图像文件')
            return redirect(request.url)

        # 创建一个临时目录来存放本次上传的所有文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_upload_dir = UPLOAD_FOLDER / f"temp_batch_{timestamp}"
        os.makedirs(temp_upload_dir, exist_ok=True)
        saved_filepaths = [] # 用于后续清理或记录

        for file in uploaded_files:
            if file and allowed_file(file.filename) and file.filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}:
                filename = secure_filename(file.filename)
                # 处理重名文件 (在临时目录内)
                base, ext = os.path.splitext(filename)
                counter = 1
                original_filename = filename
                while (temp_upload_dir / filename).exists():
                    filename = f"{base}_{counter}{ext}"
                    counter += 1
                filepath = temp_upload_dir / filename
                file.save(filepath)
                saved_filepaths.append(str(filepath)) # 保存临时文件路径
                print(f"已保存上传文件到临时目录: {filepath}")
            elif file and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS - {'png', 'jpg', 'jpeg'}:
                 flash(f'文件 {file.filename} 是视频文件，请使用视频检测功能')
            elif file:
                 flash(f'文件 {file.filename} 类型不支持')

        if not saved_filepaths:
            # 清理空的临时目录
            shutil.rmtree(temp_upload_dir, ignore_errors=True)
            flash('没有有效的图像文件被上传')
            return redirect(request.url)

        # 确定模型路径
        try:
            model_path = get_model_path()
        except FileNotFoundError as e:
            # 清理临时目录
            shutil.rmtree(temp_upload_dir, ignore_errors=True)
            flash(str(e))
            return redirect(url_for('detect'))

        # 为本次检测创建唯一的输出目录名
        detection_run_name = f"detect_{timestamp}"
        detection_run_dir_from_yolo = YOLOV5_DIR / 'runs' / 'detect' / detection_run_name

        # --- 构建并执行 YOLOv5 detect.py 命令 ---
        # 关键修改：将临时目录作为 source 传递给 detect.py
        source_arg = str(temp_upload_dir)

        cmd = [
            'python', 'detect.py',
            '--weights', model_path,
            '--source', source_arg, # <--- 传递临时目录
            '--project', str(YOLOV5_DIR / 'runs' / 'detect'),
            '--name', detection_run_name,
            '--exist-ok',
            '--save-txt',
            '--save-conf',
            '--conf-thres', '0.25' # 可根据需要调整
        ]

        print(f"\n--- 开始执行检测 ---")
        print(f"工作目录: {YOLOV5_DIR}")
        print(f"执行命令: {' '.join(cmd)}")
        print(f"源文件目录: {source_arg}")
        print(f"预期输出目录: {detection_run_dir_from_yolo}")

        try:
            # 捕获 stdout 和 stderr
            result = subprocess.run(
                cmd,
                cwd=str(YOLOV5_DIR),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("--- 检测执行成功 ---")
            print("STDOUT:")
            print(result.stdout)
            if result.stderr:
                print("STDERR (可能包含警告):")
                print(result.stderr)

        except subprocess.CalledProcessError as e:
            error_msg = f'检测过程中发生错误 (退出码 {e.returncode}): {e.stderr}'
            print(f"--- 检测执行失败 ---")
            print(error_msg)
            print("STDOUT:")
            print(e.stdout)
            flash(error_msg)
            # 清理临时目录
            shutil.rmtree(temp_upload_dir, ignore_errors=True)
            # 即使出错，也尝试显示已处理的图像 (如果有的话)


        # --- 收集并处理检测结果 ---
        print(f"\n--- 开始收集结果 ---")
        results = []
        
        # 确保 YOLOv5 生成的目录存在
        if not detection_run_dir_from_yolo.exists():
             error_msg = f"检测完成，但未找到预期的输出目录: {detection_run_dir_from_yolo}"
             print(error_msg)
             flash(error_msg)
             # 清理临时目录
             shutil.rmtree(temp_upload_dir, ignore_errors=True)
             # 返回空结果或尝试处理
        else:
            print(f"在 YOLOv5 输出目录中查找文件: {detection_run_dir_from_yolo}")

        # 遍历原始上传的文件名列表来查找结果
        # 注意：我们使用原始文件名（在临时目录中的），因为 detect.py 会处理它们并保存同名结果
        for filepath_str in saved_filepaths: # 遍历临时文件路径
            filepath = Path(filepath_str)
            filename = filepath.name # 获取文件名，例如 image.jpg
            print(f"  处理文件: {filename}")
            
            # 1. 查找检测后的图像
            detected_img_path = detection_run_dir_from_yolo / filename
            
            # 2. 查找对应的标签文件
            labels_filename = filepath.stem + '.txt' # 获取不带扩展名的文件名
            labels_path = detection_run_dir_from_yolo / 'labels' / labels_filename

            result_item = {
                'original': f'uploads/temp_batch_{timestamp}/{filename}', # 指向临时目录中的原始文件
                'detected': None,
                'labels': []
            }

            # --- 复制检测图像到我们的管理目录 ---
            if detected_img_path.exists():
                print(f"    找到检测图像: {detected_img_path}")
                final_detected_dir = DETECTION_OUTPUT_FOLDER / detection_run_name
                os.makedirs(final_detected_dir, exist_ok=True)
                final_detected_path = final_detected_dir / filename
                
                # 复制文件
                try:
                    shutil.copy(detected_img_path, final_detected_path)
                    print(f"    已复制检测图像到: {final_detected_path}")
                    result_item['detected'] = f'detections/{detection_run_name}/{filename}'
                except Exception as copy_err:
                    print(f"    复制检测图像失败: {copy_err}")
                    flash(f"复制检测图像 {filename} 时出错。")
            else:
                print(f"    未找到检测图像: {detected_img_path}")


            # --- 解析标签文件 ---
            if labels_path.exists():
                print(f"    找到标签文件: {labels_path}")
                try:
                    with open(labels_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 6:
                                result_item['labels'].append({
                                    'class_id': parts[0],
                                    'confidence': f"{float(parts[5]):.2f}"
                                })
                    if not result_item['labels']:
                         print(f"    标签文件 {labels_filename} 为空。")
                except Exception as label_err:
                    print(f"    读取/解析标签文件失败: {label_err}")
                    flash(f"解析标签文件 {labels_filename} 时出错。")
            else:
                print(f"    未找到标签文件: {labels_path}")
                # 检查检测图像是否存在来判断是无检测还是错误
                if detected_img_path.exists():
                     print(f"    检测完成，但未检测到目标。")

            results.append(result_item)
        
        print("--- 结果收集完成 ---\n")
        
        # 清理临时上传目录
        shutil.rmtree(temp_upload_dir, ignore_errors=True)
        print(f"已清理临时上传目录: {temp_upload_dir}")

        # 将结果传递给模板
        return render_template('detect_results.html', results=results, timestamp=timestamp)

    # GET 请求，显示上传表单
    return render_template('detect.html')


# --- 新增视频检测功能 ---

@app.route('/detect_video', methods=['GET', 'POST'])
def detect_video():
    """视频检测页面"""
    if request.method == 'POST':
        # 处理视频文件上传
        video_file = request.files.get("video_file")
        if not video_file or video_file.filename == '':
            flash('请选择一个视频文件')
            return redirect(request.url)

        if not allowed_file(video_file.filename):
             flash('文件类型不支持')
             return redirect(request.url)

        # 检查是否为视频文件
        file_ext = video_file.filename.rsplit('.', 1)[1].lower()
        if file_ext not in {'mp4', 'avi', 'mov', 'mkv'}:
             flash('请选择一个视频文件 (支持 mp4, avi, mov, mkv)')
             return redirect(request.url)


        # 保存上传的视频到 uploads 目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = secure_filename(video_file.filename)
        # 处理重名文件
        base, ext = os.path.splitext(filename)
        counter = 1
        original_filename = filename
        while (UPLOAD_FOLDER / filename).exists():
            filename = f"{base}_{counter}{ext}"
            counter += 1
        video_filepath = UPLOAD_FOLDER / filename
        video_file.save(video_filepath)
        print(f"已保存上传视频: {video_filepath}")

        # 确定模型路径
        try:
            model_path = get_model_path()
        except FileNotFoundError as e:
            # 清理已上传的视频
            video_filepath.unlink(missing_ok=True)
            flash(str(e))
            return redirect(url_for('detect_video'))

        # 为本次视频检测创建唯一的输出目录名
        video_detection_run_name = f"video_detect_{timestamp}"
        video_detection_run_dir_from_yolo = YOLOV5_DIR / 'runs' / 'detect' / video_detection_run_name

        # --- 构建并执行 YOLOv5 detect.py 命令 (视频) ---
        cmd = [
            'python', 'detect.py',
            '--weights', model_path,
            '--source', str(video_filepath), # <--- 传递视频文件路径
            '--project', str(YOLOV5_DIR / 'runs' / 'detect'),
            '--name', video_detection_run_name,
            '--exist-ok',
            '--save-txt',
            '--save-conf',
            '--conf-thres', '0.25' # 可根据需要调整
        ]

        print(f"\n--- 开始执行视频检测 ---")
        print(f"工作目录: {YOLOV5_DIR}")
        print(f"执行命令: {' '.join(cmd)}")
        print(f"源视频文件: {video_filepath}")
        print(f"预期输出目录: {video_detection_run_dir_from_yolo}")

        try:
            # 捕获 stdout 和 stderr
            result = subprocess.run(
                cmd,
                cwd=str(YOLOV5_DIR),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("--- 视频检测执行成功 ---")
            print("STDOUT:")
            print(result.stdout)
            if result.stderr:
                print("STDERR (可能包含警告):")
                print(result.stderr)

        except subprocess.CalledProcessError as e:
            error_msg = f'视频检测过程中发生错误 (退出码 {e.returncode}): {e.stderr}'
            print(f"--- 视频检测执行失败 ---")
            print(error_msg)
            print("STDOUT:")
            print(e.stdout)
            flash(error_msg)
            # 清理上传的视频
            video_filepath.unlink(missing_ok=True)
            # 即使出错，也尝试显示已处理的视频 (如果有的话)


        # --- 处理视频检测结果 ---
        print(f"\n--- 开始处理视频检测结果 ---")
        detected_video_path = None
        labels_dir_path = None

        # YOLOv5 detect.py 会将处理后的视频保存在 {project}/{name}/ 下，文件名与原视频相同
        potential_detected_video = video_detection_run_dir_from_yolo / video_filepath.name
        if potential_detected_video.exists():
            print(f"    找到检测后的视频: {potential_detected_video}")
            # 将检测后的视频复制到我们的管理目录
            final_video_dir = DETECTION_OUTPUT_FOLDER / video_detection_run_name
            os.makedirs(final_video_dir, exist_ok=True)
            final_detected_video_path = final_video_dir / video_filepath.name
            try:
                shutil.copy(potential_detected_video, final_detected_video_path)
                print(f"    已复制检测视频到: {final_detected_video_path}")
                detected_video_path = f'detections/{video_detection_run_name}/{video_filepath.name}'
            except Exception as copy_err:
                print(f"    复制检测视频失败: {copy_err}")
                flash(f"复制检测视频时出错。")
        else:
            print(f"    未找到检测后的视频: {potential_detected_video}")
            flash("视频检测完成，但未生成输出视频。")

        # 查找标签文件目录
        potential_labels_dir = video_detection_run_dir_from_yolo / 'labels'
        if potential_labels_dir.exists():
            print(f"    找到标签文件目录: {potential_labels_dir}")
            labels_dir_path = str(potential_labels_dir) # 可用于后续处理或显示
        else:
             print(f"    未找到标签文件目录: {potential_labels_dir}")

        print("--- 视频检测结果处理完成 ---\n")

        # 清理上传的原始视频文件 (可选，如果想保留原始上传文件则注释掉)
        # video_filepath.unlink(missing_ok=True)
        # print(f"已删除上传的原始视频文件: {video_filepath}")

        # 将结果传递给模板
        return render_template(
            'video_detect_results.html',
            original_video=f'uploads/{filename}',
            detected_video=detected_video_path,
            labels_dir=labels_dir_path,
            timestamp=timestamp
        )

    # GET 请求，显示上传表单
    return render_template('detect_video.html')


# --- 静态文件服务 ---

@app.route('/dataset/<path:filename>')
def dataset_files(filename):
    """提供对 dataset 目录下文件的访问"""
    return send_from_directory(DATASET_DIR, filename)

# 注意：由于我们清理了临时上传目录，直接访问 /uploads/temp_batch_... 下的原始文件将失败。
# 如果需要在结果页面显示原始图像，需要在复制检测图像时也复制原始图像。
# 为了简化，这里只提供对 detections 目录的访问。
# 如果需要显示原始图，可以修改 detect 路由，在复制 detected 图像时也复制 original 图像到 detections 目录下的一个子文件夹。

@app.route('/uploads/<path:filename>')
def uploaded_files(filename):
    """提供对 uploads 目录下文件的访问 (主要用于访问非临时的原始上传文件)"""
    # 如果 filename 包含 temp_batch_，说明是临时文件，已被清理，应返回 404 或默认图片
    if 'temp_batch_' in filename:
        # 可以返回一个默认的“图片已处理”或“原始图已清理”的占位符图片
        # 或者简单地返回 404
        from flask import abort
        abort(404) # 或者返回一个默认图片
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/detections/<path:filename>')
def detection_files(filename):
    """提供对 detections 目录下文件的访问"""
    return send_from_directory(DETECTION_OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    device = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f"PyTorch is using {device}.")
    print("Starting Flask app...")
    app.run(debug=True) # 本地运行，debug=True 方便开发




