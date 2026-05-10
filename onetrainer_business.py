import os
import re
import json
import math
import shutil
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from PyQt6.QtCore import QThread, pyqtSignal, QRect
from PyQt6.QtGui import QImageReader, QImage

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ==========================================
# CONSTANTS & DATA STRUCTURES
# ==========================================

BUCKETS = [
    (4.0, 1.0), (3.5, 1.0), (3.0, 1.0), (2.5, 1.0), (2.0, 1.0), 
    (1.75, 1.0), (1.5, 1.0), (1.25, 1.0), (1.0, 1.0), (1.0, 1.25), 
    (1.0, 1.5), (1.0, 1.75), (1.0, 2.0), (1.0, 2.5), (1.0, 3.0), 
    (1.0, 3.5), (1.0, 4.0)
]

@dataclass
class ImageInfo:
    filepath: str
    filename: str
    width: int
    height: int
    mp: float
    
    yolo_box: Optional[Tuple[int, int, int, int]] = None       
    yolo_padded: Optional[Tuple[int, int, int, int]] = None    
    
    primary_bucket: Tuple[float, float] = (1.0, 1.0)
    alternate_bucket: Tuple[float, float] = (1.0, 1.0)
    
    center_crop_rect: Tuple[int, int, int, int] = (0, 0, 0, 0) 
    primary_smart_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    alternate_smart_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    
    primary_cutoff: float = 0.0
    alternate_cutoff: float = 0.0
    
    primary_crop_px: float = 0.0
    primary_crop_dim: str = ""
    alternate_crop_px: float = 0.0
    alternate_crop_dim: str = ""
    
    best_ratio_type: str = "primary"

# ==========================================
# PROCESSING LOGIC
# ==========================================

class PathResolver:
    @staticmethod
    def resolve(path: str) -> str:
        if os.name == 'nt' and path.startswith('/mnt/'):
            match = re.match(r'^/mnt/([a-zA-Z])/(.*)', path)
            if match: return f"{match.group(1).upper()}:/{match.group(2)}"
        return path

class CropMath:
    @staticmethod
    def calculate_cutoff(crop_rect: Tuple[int, int, int, int], yolo_box: Tuple[int, int, int, int]) -> float:
        if not yolo_box: return 0.0
        cx, cy, cw, ch = crop_rect
        yx1, yy1, yx2, yy2 = yolo_box
        
        ix1, iy1 = max(cx, yx1), max(cy, yy1)
        ix2, iy2 = min(cx + cw, yx2), min(cy + ch, yy2)
        
        inter_area = 0 if (ix2 < ix1 or iy2 < iy1) else (ix2 - ix1) * (iy2 - iy1)
        yolo_area = (yx2 - yx1) * (yy2 - yy1)
        return max(0.0, yolo_area - inter_area)

    @staticmethod
    def calculate_smart_crop(w: int, h: int, bucket: Tuple[float, float], yolo_padded: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        target_ar = bucket[0] / bucket[1]
        orig_ar = w / h
        
        if target_ar > orig_ar:
            crop_w, crop_h = w, int(w / target_ar)
        else:
            crop_w, crop_h = int(h * target_ar), h
            
        if not yolo_padded:
            return (int((w - crop_w) / 2), int((h - crop_h) / 2), crop_w, crop_h)
            
        yx1, yy1, yx2, yy2 = yolo_padded
        cx = int(((yx1 + yx2) / 2) - (crop_w / 2))
        cy = int(((yy1 + yy2) / 2) - (crop_h / 2))
        
        return (max(0, min(w - crop_w, cx)), max(0, min(h - crop_h, cy)), crop_w, crop_h)

    @staticmethod
    def get_band_metrics(w: int, h: int, bucket: Tuple[float, float], res: int) -> Tuple[float, str]:
        budget_mp = res * res
        bucket_ar = bucket[0] / bucket[1]
        scale = math.sqrt(budget_mp / (w * h))
        scaled_w, scaled_h = w * scale, h * scale
        
        if scaled_w / scaled_h > bucket_ar:
            return (scaled_w - (scaled_h * bucket_ar), "Left/Right")
        return (scaled_h - (scaled_w / bucket_ar), "Top/Bottom")

class BucketCalculator:
    @staticmethod
    def calculate(w: int, h: int, resolution: int, yolo_padded: Optional[Tuple[int, int, int, int]] = None) -> ImageInfo:
        ar = w / h
        sorted_buckets = sorted(BUCKETS, key=lambda b: abs((b[0]/b[1]) - ar))
        primary, alternate = sorted_buckets[0], sorted_buckets[1]
        
        center_crop = CropMath.calculate_smart_crop(w, h, primary, None)
        primary_smart = CropMath.calculate_smart_crop(w, h, primary, yolo_padded)
        alternate_smart = CropMath.calculate_smart_crop(w, h, alternate, yolo_padded)
        
        primary_cutoff = CropMath.calculate_cutoff(primary_smart, yolo_padded) if yolo_padded else 0.0
        alternate_cutoff = CropMath.calculate_cutoff(alternate_smart, yolo_padded) if yolo_padded else 0.0
        best_ratio = "primary" if primary_cutoff <= alternate_cutoff else "alternate"

        pri_px, pri_dim = CropMath.get_band_metrics(w, h, primary, resolution)
        alt_px, alt_dim = CropMath.get_band_metrics(w, h, alternate, resolution)
            
        return ImageInfo(
            filepath="", filename="", width=w, height=h, mp=(w*h)/1000000,
            yolo_box=None, yolo_padded=yolo_padded,
            primary_bucket=primary, alternate_bucket=alternate,
            center_crop_rect=center_crop, primary_smart_rect=primary_smart,
            alternate_smart_rect=alternate_smart,
            primary_cutoff=primary_cutoff, alternate_cutoff=alternate_cutoff,
            primary_crop_px=pri_px, primary_crop_dim=pri_dim,
            alternate_crop_px=alt_px, alternate_crop_dim=alt_dim,
            best_ratio_type=best_ratio
        )

class DatasetExporter:
    @staticmethod
    def export_crop(img_info: ImageInfo, crop_rect: Tuple[int, int, int, int], save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(img_info.filename)[0]
        base_dir = os.path.dirname(img_info.filepath)

        # 1. Save Main Image strictly as PNG
        img = QImage(img_info.filepath)
        cropped_img = img.copy(QRect(*crop_rect))
        main_out_path = os.path.join(save_dir, f"{base_name}.png")
        cropped_img.save(main_out_path, "PNG")

        # 2. Copy joint caption text file
        txt_path = os.path.join(base_dir, f"{base_name}.txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(save_dir, f"{base_name}.txt"))

        # 3. Handle Masklabel (relative crop calculation)
        possible_exts = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
        mask_path = None
        for ext in possible_exts:
            pot_mask = os.path.join(base_dir, f"{base_name}-masklabel{ext}")
            if os.path.exists(pot_mask):
                mask_path = pot_mask
                break

        if mask_path:
            mask_img = QImage(mask_path)
            mw, mh = mask_img.width(), mask_img.height()
            if mw > 0 and mh > 0:
                cx, cy, cw, ch = crop_rect
                
                # Calculate relative boundaries (0.0 to 1.0)
                rel_x = cx / img_info.width
                rel_y = cy / img_info.height
                rel_w = cw / img_info.width
                rel_h = ch / img_info.height

                # Map back to mask absolute dimensions
                mx = int(rel_x * mw)
                my = int(rel_y * mh)
                
                # Min clamping ensures we never run off the edge by 1 pixel due to rounding
                mcw = min(mw - mx, int(rel_w * mw))
                mch = min(mh - my, int(rel_h * mh))

                cropped_mask = mask_img.copy(QRect(mx, my, mcw, mch))
                mask_out_path = os.path.join(save_dir, f"{base_name}-masklabel.png")
                cropped_mask.save(mask_out_path, "PNG")

# ==========================================
# WORKER THREADS
# ==========================================

class ScannerWorker(QThread):
    progress_signal = pyqtSignal(str, int, int)
    image_found_signal = pyqtSignal(str, object)
    concept_started_signal = pyqtSignal(str, bool)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, config_path: str, formats: List[str], target_res: int, neg_filters: List[str], use_yolo: bool, yolo_pad: int):
        super().__init__()
        self.config_path = config_path
        self.formats = [f.strip().lower() for f in formats]
        self.target_res = target_res
        self.neg_filters = [nf.strip() for nf in neg_filters if nf.strip()]
        self.use_yolo = use_yolo
        self.yolo_pad = yolo_pad
        self.is_running = True
        self.yolo_model = None

    def run(self):
        try:
            if self.use_yolo and YOLO_AVAILABLE:
                self.yolo_model = YOLO('yolov8n.pt')
            elif self.use_yolo and not YOLO_AVAILABLE:
                self.error_signal.emit("YOLO enabled but 'ultralytics' not installed.")
                return

            with open(self.config_path, 'r', encoding='utf-8') as f: config = json.load(f)
                
            for concept in config.get('concepts', []):
                if not self.is_running: break
                name = concept.get('name', 'Unknown')
                path = PathResolver.resolve(concept.get('path', ''))
                self.concept_started_signal.emit(name, concept.get('enabled', False))
                if not os.path.exists(path): continue
                    
                files = [os.path.join(path, f) for f in os.listdir(path) if f.split('.')[-1].lower() in self.formats and not any(nf in f for nf in self.neg_filters)]
                total = len(files)
                
                for i, filepath in enumerate(files):
                    if not self.is_running: break
                    reader = QImageReader(filepath)
                    w, h = reader.size().width(), reader.size().height()
                    
                    if w > 0 and h > 0:
                        yolo_box, yolo_padded = None, None
                        
                        if self.use_yolo and self.yolo_model:
                            results = self.yolo_model(filepath, classes=[0], verbose=False)
                            boxes = results[0].boxes.xyxy.cpu().numpy()
                            if len(boxes) > 0:
                                bx1, by1 = int(np.min(boxes[:, 0])), int(np.min(boxes[:, 1]))
                                bx2, by2 = int(np.max(boxes[:, 2])), int(np.max(boxes[:, 3]))
                                yolo_box = (bx1, by1, bx2, by2)
                                
                                px1, py1 = max(0, bx1 - self.yolo_pad), max(0, by1 - self.yolo_pad)
                                px2, py2 = min(w, bx2 + self.yolo_pad), min(h, by2 + self.yolo_pad)
                                yolo_padded = (px1, py1, px2, py2)

                        img_info = BucketCalculator.calculate(w, h, self.target_res, yolo_padded)
                        img_info.filepath = filepath
                        img_info.filename = os.path.basename(filepath)
                        img_info.yolo_box = yolo_box
                        
                        self.image_found_signal.emit(name, img_info)
                        
                    self.progress_signal.emit(name, i + 1, total)
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self): self.is_running = False

class BatchCropWorker(QThread):
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal()
    
    def __init__(self, images: List[ImageInfo]):
        super().__init__()
        self.images = images

    def run(self):
        for i, img_info in enumerate(self.images):
            save_dir = os.path.join(os.path.dirname(img_info.filepath), 'cropped')
            rect = img_info.primary_smart_rect if img_info.best_ratio_type == "primary" else img_info.alternate_smart_rect
            
            DatasetExporter.export_crop(img_info, rect, save_dir)
            
            self.progress_signal.emit(i + 1, len(self.images))
        self.finished_signal.emit()