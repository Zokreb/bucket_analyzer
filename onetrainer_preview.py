import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
    QPushButton, QRadioButton, QButtonGroup, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QImage

# Import the business logic needed for saving
from onetrainer_business import ImageInfo, DatasetExporter

class AdvancedPreviewDialog(QDialog):
    def __init__(self, image_data: list, start_index: int, parent=None):
        super().__init__(parent)
        self.image_data = image_data
        self.current_index = start_index
        self.resize(1100, 800)
        
        self.init_ui()
        self.load_image(self.current_index)
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # Left Side: Image
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.img_label, stretch=3)
        
        # Right Side: Controls
        control_panel = QWidget()
        self.vbox = QVBoxLayout(control_panel)
        self.vbox.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Navigation
        nav_layout = QHBoxLayout()
        self.btn_prev_concept = QPushButton("<< Concept")
        self.btn_prev_img = QPushButton("< Image")
        self.btn_next_img = QPushButton("Image >")
        self.btn_next_concept = QPushButton("Concept >>")
        
        self.btn_prev_concept.clicked.connect(self.prev_concept)
        self.btn_prev_img.clicked.connect(self.prev_img)
        self.btn_next_img.clicked.connect(self.next_img)
        self.btn_next_concept.clicked.connect(self.next_concept)
        
        nav_layout.addWidget(self.btn_prev_concept)
        nav_layout.addWidget(self.btn_prev_img)
        nav_layout.addWidget(self.btn_next_img)
        nav_layout.addWidget(self.btn_next_concept)
        self.vbox.addLayout(nav_layout)
        
        line_nav = QFrame()
        line_nav.setFrameShape(QFrame.Shape.HLine)
        self.vbox.addWidget(line_nav)
        
        # Status
        self.vbox.addWidget(QLabel("<b>Bounding Box (YOLO):</b>"))
        self.lbl_status = QLabel()
        self.vbox.addWidget(self.lbl_status)
        
        line_status = QFrame()
        line_status.setFrameShape(QFrame.Shape.HLine)
        self.vbox.addWidget(line_status)
        
        # Radio Buttons
        self.bg = QButtonGroup(self)
        self.rad_primary = QRadioButton()
        self.lbl_pri_loss = QLabel()
        
        self.rad_alternate = QRadioButton()
        self.lbl_alt_loss = QLabel()
        
        self.bg.addButton(self.rad_primary, 1)
        self.bg.addButton(self.rad_alternate, 2)
        self.bg.buttonClicked.connect(self.on_radio_changed)
        
        self.vbox.addWidget(self.rad_primary)
        self.vbox.addWidget(self.lbl_pri_loss)
        self.vbox.addWidget(self.rad_alternate)
        self.vbox.addWidget(self.lbl_alt_loss)
        
        self.vbox.addStretch()
        self.btn_save = QPushButton("Save PNG Crop + Meta to Folder")
        self.btn_save.setStyleSheet("padding: 10px; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_crop)
        self.vbox.addWidget(self.btn_save)
        
        layout.addWidget(control_panel, stretch=1)

    def load_image(self, index: int):
        self.current_index = index
        self.img_info = self.image_data[index]['img_info']
        
        # Update Window & Labels
        concept_name = self.image_data[index]['concept_item'].text(0)
        self.setWindowTitle(f"Smart Crop Editor - [{concept_name}] {self.img_info.filename}")
        self.lbl_status.setText(f"Status: {'Detected' if self.img_info.yolo_padded else 'Not Detected'}")
        
        # Update Radio Text
        self.rad_primary.setText(f"Primary Bucket ({self.img_info.primary_bucket[0]}:{self.img_info.primary_bucket[1]})")
        self.lbl_pri_loss.setText(f"Crop Band: {self.img_info.primary_crop_px:.1f}px ({self.img_info.primary_crop_dim})")
        
        self.rad_alternate.setText(f"Alternate Bucket ({self.img_info.alternate_bucket[0]}:{self.img_info.alternate_bucket[1]})")
        self.lbl_alt_loss.setText(f"Crop Band: {self.img_info.alternate_crop_px:.1f}px ({self.img_info.alternate_crop_dim})")
        
        # Reset styles
        self.lbl_pri_loss.setStyleSheet("")
        self.lbl_alt_loss.setStyleSheet("")
        
        # Set active radio based on memory
        if self.img_info.best_ratio_type == "primary":
            self.rad_primary.setChecked(True)
            self.lbl_pri_loss.setStyleSheet("color: darkgreen; font-weight: bold;")
            self.current_mode = "primary"
        else:
            self.rad_alternate.setChecked(True)
            self.lbl_alt_loss.setStyleSheet("color: darkgreen; font-weight: bold;")
            self.current_mode = "alternate"
            
        self.draw_overlays()
        self.update_nav_buttons()

    def update_nav_buttons(self):
        self.btn_prev_img.setEnabled(self.current_index > 0)
        self.btn_next_img.setEnabled(self.current_index < len(self.image_data) - 1)
        
        # Check if there's a previous concept
        current_concept = self.image_data[self.current_index]['concept_item'].text(0)
        has_prev_concept = False
        for i in range(self.current_index - 1, -1, -1):
            c_item = self.image_data[i]['concept_item']
            if c_item.text(0) != current_concept and c_item.checkState(0) == Qt.CheckState.Checked:
                has_prev_concept = True
                break
        self.btn_prev_concept.setEnabled(has_prev_concept)
        
        # Check if there's a next concept
        has_next_concept = False
        for i in range(self.current_index + 1, len(self.image_data)):
            c_item = self.image_data[i]['concept_item']
            if c_item.text(0) != current_concept and c_item.checkState(0) == Qt.CheckState.Checked:
                has_next_concept = True
                break
        self.btn_next_concept.setEnabled(has_next_concept)

    # --- Navigation Logic ---
    def prev_img(self):
        if self.current_index > 0: self.load_image(self.current_index - 1)

    def next_img(self):
        if self.current_index < len(self.image_data) - 1: self.load_image(self.current_index + 1)

    def next_concept(self):
        current_concept = self.image_data[self.current_index]['concept_item'].text(0)
        for i in range(self.current_index + 1, len(self.image_data)):
            c_item = self.image_data[i]['concept_item']
            if c_item.text(0) != current_concept and c_item.checkState(0) == Qt.CheckState.Checked:
                self.load_image(i)
                return

    def prev_concept(self):
        current_concept = self.image_data[self.current_index]['concept_item'].text(0)
        found_target_concept = None
        
        # Walk backward to find the name of the previous enabled concept
        for i in range(self.current_index - 1, -1, -1):
            c_item = self.image_data[i]['concept_item']
            if c_item.text(0) != current_concept and c_item.checkState(0) == Qt.CheckState.Checked:
                found_target_concept = c_item.text(0)
                break
                
        if found_target_concept:
            # Walk backward again to find the FIRST image of that concept
            first_idx = i
            for j in range(i, -1, -1):
                if self.image_data[j]['concept_item'].text(0) == found_target_concept:
                    first_idx = j
                else:
                    break
            self.load_image(first_idx)

    def on_radio_changed(self, button):
        self.current_mode = "primary" if button == self.rad_primary else "alternate"
        self.img_info.best_ratio_type = self.current_mode # Persist choice to memory
        
        # Update styles
        self.lbl_pri_loss.setStyleSheet("color: darkgreen; font-weight: bold;" if self.current_mode == "primary" else "")
        self.lbl_alt_loss.setStyleSheet("color: darkgreen; font-weight: bold;" if self.current_mode == "alternate" else "")
        self.draw_overlays()
        
    def draw_overlays(self):
        pixmap = QPixmap(self.img_info.filepath)
        if pixmap.isNull(): return
        
        painter = QPainter(pixmap)
        w, h = pixmap.width(), pixmap.height()
        
        # Center Crop (Red)
        painter.setBrush(QColor(255, 0, 0, 80))
        painter.setPen(Qt.PenStyle.NoPen)
        cx, cy, cw, ch = self.img_info.center_crop_rect
        painter.drawRect(0, 0, w, cy) 
        painter.drawRect(0, cy + ch, w, h - (cy + ch)) 
        painter.drawRect(0, cy, cx, ch) 
        painter.drawRect(cx + cw, cy, w - (cx + cw), ch) 
        
        # YOLO Padded (Green)
        if self.img_info.yolo_padded:
            painter.setPen(QPen(QColor(0, 255, 0), max(3, int(w*0.005))))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            yx1, yy1, yx2, yy2 = self.img_info.yolo_padded
            painter.drawRect(yx1, yy1, yx2 - yx1, yy2 - yy1)
            
        # Smart Crop (Blue)
        rect = self.img_info.primary_smart_rect if self.current_mode == "primary" else self.img_info.alternate_smart_rect
        painter.setPen(QPen(QColor(0, 150, 255), max(4, int(w*0.008))))
        painter.drawRect(*rect)
        
        painter.end()
        self.img_label.setPixmap(pixmap.scaled(800, 800, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def save_crop(self):
        save_dir = os.path.join(os.path.dirname(self.img_info.filepath), 'cropped')
        rect = self.img_info.primary_smart_rect if self.current_mode == "primary" else self.img_info.alternate_smart_rect
        
        DatasetExporter.export_crop(self.img_info, rect, save_dir)
        
        base_name = os.path.splitext(self.img_info.filename)[0]
        success_msg = f"Exported successfully!\n\nDestination Path:\n{save_dir}\n\nBase File Name:\n{base_name}.png\n\n(Captions and masklabels included if present)"
        QMessageBox.information(self, "Saved", success_msg)