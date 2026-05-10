import sys
import os
from typing import Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QFormLayout, QLineEdit, QPushButton, QSplitter, QTreeWidget, 
    QTreeWidgetItem, QFileDialog, QProgressBar, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from onetrainer_business import BUCKETS, ImageInfo, ScannerWorker, BatchCropWorker, BucketCalculator, YOLO_AVAILABLE
from onetrainer_preview import AdvancedPreviewDialog

class HistogramCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes.set_title("Bucket Distribution")
        self.fig.tight_layout()

    def update_plot(self, data: Dict[str, int]):
        self.axes.clear()
        buckets_str = [f"{b[0]}:{b[1]}" for b in BUCKETS]
        counts = [data.get(b, 0) for b in buckets_str]
        
        bars = self.axes.bar(buckets_str, counts, color='steelblue')
        self.axes.bar_label(bars, padding=3)
        self.axes.set_title("Bucket Distribution")
        self.axes.tick_params(axis='x', rotation=45)
        
        max_count = max(counts) if counts else 0
        if max_count == 0:
            self.axes.set_ylim(0, 1) 
        else:
            self.axes.set_ylim(0, max_count * 1.15)
            
        self.fig.tight_layout()
        self.draw()

class MainView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Henri's Advanced Batch Evaluator")
        self.resize(1500, 800)
        self.setAcceptDrops(True)
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.top_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.input_widget = QWidget()
        input_layout = QFormLayout(self.input_widget)
        
        self.txt_config = QLineEdit()
        self.btn_browse = QPushButton("Browse")
        config_row = QHBoxLayout()
        config_row.addWidget(self.txt_config)
        config_row.addWidget(self.btn_browse)
        
        self.txt_formats = QLineEdit("jpg, jpeg, webp, png")
        self.txt_neg_filters = QLineEdit("-masklabel")
        self.txt_resolutions = QLineEdit("1024")
        self.txt_batch_size = QLineEdit("6")
        
        self.chk_yolo = QCheckBox("Enable YOLOv8 Smart Cropping")
        self.chk_yolo.setChecked(YOLO_AVAILABLE)
        self.chk_yolo.setEnabled(YOLO_AVAILABLE)
        self.txt_yolo_pad = QLineEdit("20")
        
        self.btn_scan = QPushButton("Start Scan (Uses Cache)")
        self.btn_force_rescan = QPushButton("Force Rescan Selected")
        self.btn_force_rescan.setStyleSheet("background-color: #8f2b2b; color: white; font-weight: bold;")
        
        self.btn_toggle_sel = QPushButton("Toggle All Concepts") 
        self.btn_refresh = QPushButton("Recalculate All")
        self.btn_batch_crop = QPushButton("Batch Apply Smart Crop")
        self.btn_batch_crop.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold;")
        
        input_layout.addRow("Config File:", config_row)
        input_layout.addRow("Formats (CSV):", self.txt_formats)
        input_layout.addRow("Ignore Strings:", self.txt_neg_filters)
        input_layout.addRow("Resolutions (CSV):", self.txt_resolutions)
        input_layout.addRow("Planned Batch:", self.txt_batch_size)
        input_layout.addRow("", self.chk_yolo)
        input_layout.addRow("YOLO Pad (px):", self.txt_yolo_pad)
        
        btn_row_top = QHBoxLayout()
        btn_row_top.addWidget(self.btn_scan)
        btn_row_top.addWidget(self.btn_force_rescan)
        input_layout.addRow("", btn_row_top)
        
        btn_row_mid = QHBoxLayout()
        btn_row_mid.addWidget(self.btn_toggle_sel) 
        btn_row_mid.addWidget(self.btn_refresh)
        input_layout.addRow("", btn_row_mid)
        
        input_layout.addRow("", self.btn_batch_crop)
        
        self.histogram_widget = HistogramCanvas(self)
        self.top_splitter.addWidget(self.input_widget)
        self.top_splitter.addWidget(self.histogram_widget)
        self.top_splitter.setSizes([400, 1000])
        
        self.tree = QTreeWidget()
        self.tree.setColumnCount(11)
        self.tree.setHeaderLabels([
            "Concept", "File", "Dims", "Closest AR", "Alt AR", 
            "Crop (px)", "Crop Dim", "YOLO", "Selected Ratio", 
            "Orphan", "Progress"
        ])
        
        self.main_splitter.addWidget(self.top_splitter)
        self.main_splitter.addWidget(self.tree)
        self.main_splitter.setSizes([350, 450])
        main_layout.addWidget(self.main_splitter)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and event.mimeData().urls()[0].toLocalFile().endswith('.json'):
            event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        self.txt_config.setText(event.mimeData().urls()[0].toLocalFile())


class AppController:
    def __init__(self, view: MainView):
        self.view = view
        self.worker = None
        self.concept_nodes, self.concept_progress, self.image_data = {}, {}, []
        self.view.btn_browse.clicked.connect(self.browse_file)
        self.view.btn_scan.clicked.connect(self.toggle_scan)
        self.view.btn_force_rescan.clicked.connect(self.run_force_rescan)
        self.view.btn_toggle_sel.clicked.connect(self.toggle_selection) 
        self.view.btn_refresh.clicked.connect(self.update_analytics)
        self.view.btn_batch_crop.clicked.connect(self.run_batch_crop)
        self.view.tree.itemChanged.connect(self.on_item_changed)
        self.view.tree.itemDoubleClicked.connect(self.on_item_double_clicked)

    def browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(self.view, "Select JSON", "", "JSON (*.json)")
        if fname: self.view.txt_config.setText(fname)

    def toggle_scan(self):
        if self.worker and self.worker.is_running:
            self.worker.stop()
            self.view.btn_scan.setText("Start Scan (Uses Cache)")
        else:
            self.start_scan()
            
    def run_force_rescan(self):
        active_concepts = [
            self.view.tree.topLevelItem(i).text(0)
            for i in range(self.view.tree.topLevelItemCount())
            if self.view.tree.topLevelItem(i).checkState(0) == Qt.CheckState.Checked
        ]
        
        if not active_concepts:
            QMessageBox.warning(self.view, "No Selection", "Please check at least one concept in the tree view to force rescan.")
            return
            
        self.start_scan(force_concepts=active_concepts)

    def start_scan(self, force_concepts=None):
        cfg = self.view.txt_config.text()
        if not os.path.exists(cfg): return QMessageBox.warning(self.view, "Error", "Invalid config path.")

        res = int(self.view.txt_resolutions.text().split(',')[0].strip())
        pad = int(self.view.txt_yolo_pad.text())
        
        self.view.tree.clear()
        self.concept_nodes.clear()
        self.concept_progress.clear()
        self.image_data.clear()
        self.view.btn_scan.setText("Stop Scan")

        self.worker = ScannerWorker(
            cfg, self.view.txt_formats.text().split(','), res,
            self.view.txt_neg_filters.text().split(','),
            self.view.chk_yolo.isChecked(), pad,
            force_concepts=force_concepts
        )
        self.worker.concept_started_signal.connect(self.on_concept_started)
        self.worker.image_found_signal.connect(self.on_image_found)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_scan_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_concept_started(self, name: str, is_enabled: bool):
        item = QTreeWidgetItem(self.view.tree)
        item.setText(0, name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, Qt.CheckState.Checked if is_enabled else Qt.CheckState.Unchecked)
        
        # Disabled concepts skip scanning, but we still add a progress bar for UI consistency
        pbar = QProgressBar()
        pbar.setValue(0)
        self.view.tree.setItemWidget(item, 10, pbar)
        self.concept_nodes[name], self.concept_progress[name] = item, pbar

    def on_image_found(self, name: str, img: ImageInfo):
        parent = self.concept_nodes.get(name)
        child = QTreeWidgetItem(parent)
        self.update_row_visuals(child, img)
        self.image_data.append({'item': child, 'concept_item': parent, 'img_info': img})

    def update_row_visuals(self, child: QTreeWidgetItem, img: ImageInfo):
        child.setText(1, img.filename)
        child.setText(2, f"{img.width}x{img.height}")
        child.setText(3, f"{img.primary_bucket[0]}:{img.primary_bucket[1]}")
        child.setText(4, f"{img.alternate_bucket[0]}:{img.alternate_bucket[1]}")
        
        crop_px = img.primary_crop_px if img.best_ratio_type == "primary" else img.alternate_crop_px
        crop_dim = img.primary_crop_dim if img.best_ratio_type == "primary" else img.alternate_crop_dim
        
        child.setText(5, f"{crop_px:.1f}")
        child.setText(6, crop_dim)
        child.setText(7, "Yes" if img.yolo_padded else "No")
        child.setText(8, img.best_ratio_type.title())

    def on_progress(self, name: str, cur: int, tot: int):
        if pbar := self.concept_progress.get(name):
            pbar.setMaximum(tot)
            pbar.setValue(cur)

    def on_scan_finished(self):
        self.view.btn_scan.setText("Start Scan (Uses Cache)")
        self.worker = None
        self.update_analytics()

    def on_error(self, err: str): QMessageBox.critical(self.view, "Error", err)

    def on_item_changed(self, item: QTreeWidgetItem, col: int):
        if col == 0 and item.parent() is None: self.update_analytics()

    def toggle_selection(self):
        self.view.tree.blockSignals(True)
        target_state = Qt.CheckState.Checked
        if self.view.tree.topLevelItemCount() > 0:
            if self.view.tree.topLevelItem(0).checkState(0) == Qt.CheckState.Checked:
                target_state = Qt.CheckState.Unchecked

        for i in range(self.view.tree.topLevelItemCount()):
            self.view.tree.topLevelItem(i).setCheckState(0, target_state)
            
        self.view.tree.blockSignals(False)
        self.update_analytics()

    def on_item_double_clicked(self, item: QTreeWidgetItem, col: int):
        if item.parent() is None: return
        
        start_index = 0
        for i, data in enumerate(self.image_data):
            if data['item'] == item:
                start_index = i
                break
                
        dlg = AdvancedPreviewDialog(self.image_data, start_index, self.view)
        dlg.exec()
        
        for data in self.image_data:
            self.update_row_visuals(data['item'], data['img_info'])
        
        self.update_analytics()

    def update_analytics(self):
        if not self.image_data: return
        
        try:
            batch_size = int(self.view.txt_batch_size.text())
            res = int(self.view.txt_resolutions.text().split(',')[0].strip())
            pad = int(self.view.txt_yolo_pad.text())
        except ValueError:
            return
            
        bucket_counts = {f"{b[0]}:{b[1]}": 0 for b in BUCKETS}

        for data in self.image_data:
            img, child, concept = data['img_info'], data['item'], data['concept_item']
            
            # Recalculate dimensions dynamically
            new_info = BucketCalculator.calculate(img.width, img.height, res, img.yolo_padded)
            img.primary_bucket, img.alternate_bucket = new_info.primary_bucket, new_info.alternate_bucket
            img.best_ratio_type = new_info.best_ratio_type
            img.primary_crop_px, img.primary_crop_dim = new_info.primary_crop_px, new_info.primary_crop_dim
            img.alternate_crop_px, img.alternate_crop_dim = new_info.alternate_crop_px, new_info.alternate_crop_dim
            img.primary_smart_rect, img.alternate_smart_rect = new_info.primary_smart_rect, new_info.alternate_smart_rect
            
            self.update_row_visuals(child, img)

            if concept.checkState(0) == Qt.CheckState.Checked:
                target_bucket = img.primary_bucket if img.best_ratio_type == "primary" else img.alternate_bucket
                b_str = f"{target_bucket[0]}:{target_bucket[1]}"
                bucket_counts[b_str] = bucket_counts.get(b_str, 0) + 1

        self.view.histogram_widget.update_plot(bucket_counts)

        for data in self.image_data:
            img, child, concept = data['img_info'], data['item'], data['concept_item']
            if concept.checkState(0) == Qt.CheckState.Checked:
                target_bucket = img.primary_bucket if img.best_ratio_type == "primary" else img.alternate_bucket
                b_str = f"{target_bucket[0]}:{target_bucket[1]}"
                is_orphan = bucket_counts.get(b_str, 0) < batch_size
                child.setText(9, str(is_orphan))
                child.setForeground(9, Qt.GlobalColor.red if is_orphan else Qt.GlobalColor.black)
            else:
                child.setText(9, "-")
                child.setForeground(9, Qt.GlobalColor.gray)

    def run_batch_crop(self):
        active_imgs = [d['img_info'] for d in self.image_data if d['concept_item'].checkState(0) == Qt.CheckState.Checked]
        if not active_imgs: return
        self.view.btn_batch_crop.setEnabled(False)
        self.view.btn_batch_crop.setText(f"Processing... 0 / {len(active_imgs)}") # Show initial count
        
        self.batch_worker = BatchCropWorker(active_imgs)
        
        # --- ADD THIS LINE TO CONNECT THE SIGNAL ---
        self.batch_worker.progress_signal.connect(self.on_batch_progress) 
        
        self.batch_worker.finished_signal.connect(self.on_batch_finished)
        self.batch_worker.start()

    # --- ADD THIS NEW FUNCTION ---
    def on_batch_progress(self, current: int, total: int):
        self.view.btn_batch_crop.setText(f"Processing... {current} / {total}")

    def on_batch_finished(self):
        self.view.btn_batch_crop.setEnabled(True)
        self.view.btn_batch_crop.setText("Batch Apply Smart Crop")
        QMessageBox.information(self.view, "Complete", "Batch crop saved successfully.")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    view = MainView()
    controller = AppController(view)
    view.show()
    sys.exit(app.exec())