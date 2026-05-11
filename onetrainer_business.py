import os
import re
import json
import math
import shutil
import sqlite3
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
# DATABASE CACHE
# ==========================================

class ImageCacheDB:
    def __init__(self, db_path="onetrainer_cache.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS image_cache (
                filepath TEXT PRIMARY KEY,
                file_size INTEGER,
                mtime REAL,
                width INTEGER,
                height INTEGER,
                yolo_run INTEGER,
                yolo_x1 INTEGER,
                yolo_y1 INTEGER,
                yolo_x2 INTEGER,
                yolo_y2 INTEGER
            )
        ''')
        self.conn.commit()

    def get(self, filepath: str, mtime: float, size: int, requires_yolo: bool):
        cursor = self.conn.cursor()
        cursor.execute('SELECT file_size, mtime, width, height, yolo_run, yolo_x1, yolo_y1, yolo_x2, yolo_y2 FROM image_cache WHERE filepath = ?', (filepath,))
        row = cursor.fetchone()
        
        if row:
            db_size, db_mtime, w, h, yolo_run, yx1, yy1, yx2, yy2 = row
            if db_size == size and abs(db_mtime - mtime) < 1.0:
                if requires_yolo and not yolo_run:
                    return None 
                yolo_box = (yx1, yy1, yx2, yy2) if yx1 is not None else None
                return w, h, yolo_box
        return None

    def put(self, filepath: str, size: int, mtime: float, w: int, h: int, yolo_box: Optional[Tuple], yolo_run: int):
        yx1, yy1, yx2, yy2 = yolo_box if yolo_box else (None, None, None, None)
        self.conn.execute('''
            INSERT OR REPLACE INTO image_cache
            (filepath, file_size, mtime, width, height, yolo_run, yolo_x1, yolo_y1, yolo_x2, yolo_y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (filepath, size, mtime, w, h, yolo_run, yx1, yy1, yx2, yy2))
        self.conn.commit()

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
    def calculate(w: int, h: int, resolution: int, yolo_box: Optional[Tuple[int, int, int, int]] = None, yolo_padded: Optional[Tuple[int, int, int, int]] = None) -> ImageInfo:
        ar = w / h
        sorted_buckets = sorted(BUCKETS, key=lambda b: abs((b[0]/b[1]) - ar))
        primary, alternate = sorted_buckets[0], sorted_buckets[1]
        
        center_crop = CropMath.calculate_smart_crop(w, h, primary, None)
        primary_smart = CropMath.calculate_smart_crop(w, h, primary, yolo_padded)
        alternate_smart = CropMath.calculate_smart_crop(w, h, alternate, yolo_padded)
        
        primary_cutoff = CropMath.calculate_cutoff(primary_smart, yolo_box) if yolo_box else 0.0
        alternate_cutoff = CropMath.calculate_cutoff(alternate_smart, yolo_box) if yolo_box else 0.0
        
        if primary_cutoff == 0:
            best_ratio = "primary"
        elif alternate_cutoff == 0:
            best_ratio = "alternate"
        elif alternate_cutoff < primary_cutoff:
            best_ratio = "alternate"
        else:
            best_ratio = "primary"

        pri_px, pri_dim = CropMath.get_band_metrics(w, h, primary, resolution)
        alt_px, alt_dim = CropMath.get_band_metrics(w, h, alternate, resolution)
            
        return ImageInfo(
            filepath="", filename="", width=w, height=h, mp=(w*h)/1000000,
            yolo_box=yolo_box, yolo_padded=yolo_padded,
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

        img = QImage(img_info.filepath)
        cropped_img = img.copy(QRect(*crop_rect))
        main_out_path = os.path.join(save_dir, f"{base_name}.png")
        cropped_img.save(main_out_path, "PNG")

        txt_path = os.path.join(base_dir, f"{base_name}.txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(save_dir, f"{base_name}.txt"))

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
                rel_x, rel_y = cx / img_info.width, cy / img_info.height
                rel_w, rel_h = cw / img_info.width, ch / img_info.height

                mx, my = int(rel_x * mw), int(rel_y * mh)
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

    def __init__(self, config_path: str, formats: List[str], target_res: int, neg_filters: List[str], use_yolo: bool, yolo_pad: int, force_rescan: bool = False, ui_states: dict = None):
        super().__init__()
        self.config_path = config_path
        self.formats = [f.strip().lower() for f in formats]
        self.target_res = target_res
        self.neg_filters = [nf.strip() for nf in neg_filters if nf.strip()]
        self.use_yolo = use_yolo
        self.yolo_pad = yolo_pad
        self.force_rescan = force_rescan
        self.ui_states = ui_states or {}
        self.is_running = True
        self.yolo_model = None

    def run(self):
        try:
            db = ImageCacheDB()
            
            if self.use_yolo and YOLO_AVAILABLE:
                self.yolo_model = YOLO('yolov8n.pt')
            elif self.use_yolo and not YOLO_AVAILABLE:
                self.error_signal.emit("YOLO enabled but 'ultralytics' not installed.")
                return

            with open(self.config_path, 'r', encoding='utf-8') as f: 
                config = json.load(f)
                
            for concept in config.get('concepts', []):
                if not self.is_running: break
                
                # Smart fallback for blank OneTrainer names
                raw_name = concept.get('name', '')
                path = PathResolver.resolve(concept.get('path', ''))
                name = str(raw_name).strip() if raw_name else ""
                if not name:
                    name = os.path.basename(os.path.normpath(path)) if path else "Unknown"
                
                # UI Selection overwrites JSON defaults
                if self.ui_states and name in self.ui_states:
                    is_enabled = self.ui_states[name]
                else:
                    is_enabled = concept.get('enabled', False)
                
                self.concept_started_signal.emit(name, is_enabled)
                
                if not is_enabled or not os.path.exists(path): 
                    continue
                    
                files = [os.path.join(path, f) for f in os.listdir(path) if f.split('.')[-1].lower() in self.formats and not any(nf in f for nf in self.neg_filters)]
                total = len(files)
                
                for i, filepath in enumerate(files):
                    if not self.is_running: break
                    
                    stat = os.stat(filepath)
                    mtime, size = stat.st_mtime, stat.st_size
                    
                    cached_data = None
                    if not self.force_rescan:
                        cached_data = db.get(filepath, mtime, size, self.use_yolo)
                    
                    w, h, yolo_box = None, None, None
                    if cached_data:
                        w, h, yolo_box = cached_data
                    else:
                        reader = QImageReader(filepath)
                        w, h = reader.size().width(), reader.size().height()
                        yolo_run = 0
                        
                        if w > 0 and h > 0:
                            if self.use_yolo and self.yolo_model:
                                yolo_run = 1
                                results = self.yolo_model(filepath, classes=[0], verbose=False)
                                boxes = results[0].boxes.xyxy.cpu().numpy()
                                if len(boxes) > 0:
                                    bx1, by1 = int(np.min(boxes[:, 0])), int(np.min(boxes[:, 1]))
                                    bx2, by2 = int(np.max(boxes[:, 2])), int(np.max(boxes[:, 3]))
                                    yolo_box = (bx1, by1, bx2, by2)
                            
                            db.put(filepath, size, mtime, w, h, yolo_box, yolo_run)

                    if w and h and w > 0 and h > 0:
                        yolo_padded = None
                        if yolo_box:
                            px1, py1 = max(0, yolo_box[0] - self.yolo_pad), max(0, yolo_box[1] - self.yolo_pad)
                            px2, py2 = min(w, yolo_box[2] + self.yolo_pad), min(h, yolo_box[3] + self.yolo_pad)
                            yolo_padded = (px1, py1, px2, py2)

                        img_info = BucketCalculator.calculate(w, h, self.target_res, yolo_box, yolo_padded)
                        img_info.filepath = filepath
                        img_info.filename = os.path.basename(filepath)
                        
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