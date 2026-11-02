import sys
import os
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
from threading import Thread
from playsound import playsound
from time import time

from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QFormLayout
)

# -----------------------------
# Config & data
# -----------------------------
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
BEEP_FILE = "beep.mp3"

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# load or create attendance dataframe
if os.path.exists(ATTENDANCE_FILE):
    try:
        attendance_df = pd.read_csv(ATTENDANCE_FILE)
    except Exception:
        attendance_df = pd.DataFrame(columns=["Name", "Date", "Time"])
else:
    attendance_df = pd.DataFrame(columns=["Name", "Date", "Time"])


# -----------------------------
# Helpers
# -----------------------------
def load_known_faces():
    encodings = []
    names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            face_enc = face_recognition.face_encodings(image)
            if face_enc:
                encodings.append(face_enc[0])
                names.append(os.path.splitext(filename)[0])
    return encodings, names


def mark_attendance(name):
    global attendance_df
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    exists = ((attendance_df["Name"] == name) & (attendance_df["Date"] == date_str)).any()
    if not exists:
        attendance_df.loc[len(attendance_df)] = [name, date_str, time_str]
        try:
            Thread(target=lambda: playsound(BEEP_FILE), daemon=True).start()
        except Exception:
            pass
        attendance_df.to_csv(ATTENDANCE_FILE, index=False)


# -----------------------------
# Recognition thread (detect & mark)
# -----------------------------
class RecognitionThread(QThread):
    frame_updated = pyqtSignal(QImage)
    status_update = pyqtSignal(str)

    def __init__(self, parent=None, video_idx=0, tolerance=0.45):
        super().__init__(parent)
        self._running = False
        self.video_idx = video_idx
        self.tolerance = tolerance
        self.known_encodings, self.known_names = load_known_faces()

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.video_idx)
        if not cap.isOpened():
            self.status_update.emit("Unable to open camera")
            return
        self.status_update.emit("Recognition started")
        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.status_update.emit("Failed to read frame")
                break

            # work on small frame for speed
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            try:
                locations = face_recognition.face_locations(rgb_small)
                encodings = face_recognition.face_encodings(rgb_small, locations)
            except Exception:
                locations = []
                encodings = []

            # Draw on original (full) frame scaled coords
            for face_encoding, face_location in zip(encodings, locations):
                name = "Unknown"
                if self.known_encodings:
                    distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                    best_idx = np.argmin(distances)
                    if distances[best_idx] <= self.tolerance:
                        name = self.known_names[best_idx]
                        mark_attendance(name)

                top, right, bottom, left = [v * 4 for v in face_location]  # scale up
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # Convert to RGB and QImage (NO extra swapping)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_updated.emit(qimg)

        cap.release()
        self.status_update.emit("Recognition stopped")

    def stop(self):
        self._running = False
        self.wait()

    def refresh_known_faces(self):
        self.known_encodings, self.known_names = load_known_faces()
        self.status_update.emit(f"Loaded {len(self.known_names)} known faces")


# -----------------------------
# Registration thread (preview only)
# -----------------------------
class RegistrationThread(QThread):
    frame_updated = pyqtSignal(QImage)
    status_update = pyqtSignal(str)

    def __init__(self, parent=None, video_idx=0):
        super().__init__(parent)
        self._running = False
        self.video_idx = video_idx
        self.latest_frame = None  # store latest BGR frame for capture

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.video_idx)
        if not cap.isOpened():
            self.status_update.emit("Unable to open camera")
            return
        self.status_update.emit("Registration camera started")
        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.status_update.emit("Failed to read frame")
                break
            self.latest_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_updated.emit(qimg)
        cap.release()
        self.status_update.emit("Registration camera stopped")

    def stop(self):
        self._running = False
        self.wait()


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Attendance - PyQt5 Dashboard")
        self.setGeometry(150, 80, 1000, 640)

        # Threads
        self.rec_thread = RecognitionThread()
        self.reg_thread = RegistrationThread()

        self.rec_thread.frame_updated.connect(self.update_home_image)
        self.rec_thread.status_update.connect(self.on_status)

        self.reg_thread.frame_updated.connect(self.update_register_image)
        self.reg_thread.status_update.connect(self.on_status)

        self.init_ui()
        self.on_status("Ready")

    def init_ui(self):
        self.tabs = QTabWidget(self)

        # Home Tab
        self.home_tab = QWidget()
        self.tabs.addTab(self.home_tab, "Home")

        home_layout = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        self.home_camera_label = QLabel("No Camera")
        self.home_camera_label.setFixedSize(720, 480)
        self.home_camera_label.setStyleSheet("background-color: #111; color: white;")
        self.home_camera_label.setAlignment(Qt.AlignCenter)
        left_col.addWidget(self.home_camera_label)

        self.status_label = QLabel("")
        right_col.addWidget(self.status_label)

        self.start_btn = QPushButton("Start Recognition")
        self.start_btn.clicked.connect(self.start_recognition)
        right_col.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Recognition")
        self.stop_btn.clicked.connect(self.stop_recognition)
        right_col.addWidget(self.stop_btn)

        self.reload_btn = QPushButton("Reload Known Faces")
        self.reload_btn.clicked.connect(self.reload_faces)
        right_col.addWidget(self.reload_btn)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close_app)
        right_col.addWidget(self.exit_btn)

        right_col.addStretch(1)
        home_layout.addLayout(left_col)
        home_layout.addLayout(right_col)
        self.home_tab.setLayout(home_layout)

        # Register Tab
        self.register_tab = QWidget()
        self.tabs.addTab(self.register_tab, "Register Face")

        reg_layout = QHBoxLayout()
        reg_left = QVBoxLayout()
        reg_right = QVBoxLayout()

        self.reg_camera_label = QLabel("No Camera")
        self.reg_camera_label.setFixedSize(640, 480)
        self.reg_camera_label.setStyleSheet("background-color: #111; color: white;")
        self.reg_camera_label.setAlignment(Qt.AlignCenter)
        reg_left.addWidget(self.reg_camera_label)

        form = QFormLayout()
        self.name_input = QLineEdit()
        form.addRow("Name:", self.name_input)
        reg_right.addLayout(form)

        self.reg_start_btn = QPushButton("Start Camera")
        self.reg_start_btn.clicked.connect(self.start_registration_cam)
        reg_right.addWidget(self.reg_start_btn)

        self.reg_capture_btn = QPushButton("Capture & Save")
        self.reg_capture_btn.clicked.connect(self.capture_registration_image)
        reg_right.addWidget(self.reg_capture_btn)

        self.reg_stop_btn = QPushButton("Stop Camera")
        self.reg_stop_btn.clicked.connect(self.stop_registration_cam)
        reg_right.addWidget(self.reg_stop_btn)

        reg_right.addStretch(1)
        reg_layout.addLayout(reg_left)
        reg_layout.addLayout(reg_right)
        self.register_tab.setLayout(reg_layout)

        # Attendance Tab
        self.attendance_tab = QWidget()
        self.tabs.addTab(self.attendance_tab, "Attendance")

        att_layout = QVBoxLayout()
        self.att_table = QTableWidget()
        self.att_table.setColumnCount(3)
        self.att_table.setHorizontalHeaderLabels(["Name", "Date", "Time"])
        self.att_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        att_layout.addWidget(self.att_table)

        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self.export_csv)
        att_layout.addWidget(export_btn)

        self.attendance_tab.setLayout(att_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        self.load_attendance_table()

    # -----------------------------
    # Home (recognition) logic
    # -----------------------------
    def start_recognition(self):
        if not self.rec_thread.isRunning():
            # refresh faces before starting
            self.rec_thread.refresh_known_faces()
            self.rec_thread.start()
            self.on_status("Recognition started")
        else:
            self.on_status("Recognition already running")

    def stop_recognition(self):
        if self.rec_thread.isRunning():
            self.rec_thread.stop()
            self.on_status("Recognition stopped")
        else:
            self.on_status("Recognition not running")

    def reload_faces(self):
        self.rec_thread.refresh_known_faces()

    def update_home_image(self, qimage):
        pix = QPixmap.fromImage(qimage).scaled(self.home_camera_label.width(),
                                              self.home_camera_label.height(),
                                              Qt.KeepAspectRatio)
        self.home_camera_label.setPixmap(pix)
        # refresh attendance table display
        self.load_attendance_table()

    # -----------------------------
    # Registration logic
    # -----------------------------
    def start_registration_cam(self):
        if not self.reg_thread.isRunning():
            self.reg_thread.start()
            self.on_status("Registration camera started")
        else:
            self.on_status("Registration camera already running")

    def stop_registration_cam(self):
        if self.reg_thread.isRunning():
            self.reg_thread.stop()
            self.on_status("Registration camera stopped")
        else:
            self.on_status("Registration camera not running")

    def update_register_image(self, qimage):
        pix = QPixmap.fromImage(qimage).scaled(self.reg_camera_label.width(),
                                              self.reg_camera_label.height(),
                                              Qt.KeepAspectRatio)
        self.reg_camera_label.setPixmap(pix)

    def capture_registration_image(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Input required", "Please enter a name before capturing")
            return
        if not self.reg_thread.latest_frame is None:
            frame = self.reg_thread.latest_frame
            # crop center square
            h, w = frame.shape[:2]
            min_side = min(h, w)
            start_y = (h - min_side) // 2
            start_x = (w - min_side) // 2
            crop = frame[start_y:start_y + min_side, start_x:start_x + min_side]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(KNOWN_FACES_DIR, f"{name}_{timestamp}.jpg")
            cv2.imwrite(filename, crop)
            QMessageBox.information(self, "Saved", f"Saved {filename}")
            # refresh known faces for recognition thread
            self.rec_thread.refresh_known_faces()
        else:
            QMessageBox.critical(self, "No frame", "No camera frame available to capture")

    # -----------------------------
    # Attendance table methods
    # -----------------------------
    def load_attendance_table(self):
        global attendance_df
        self.att_table.setRowCount(0)
        for idx, row in attendance_df.iterrows():
            r = self.att_table.rowCount()
            self.att_table.insertRow(r)
            self.att_table.setItem(r, 0, QTableWidgetItem(str(row.get("Name", ""))))
            self.att_table.setItem(r, 1, QTableWidgetItem(str(row.get("Date", ""))))
            self.att_table.setItem(r, 2, QTableWidgetItem(str(row.get("Time", ""))))

    def export_csv(self):
        global attendance_df
        path, _ = QFileDialog.getSaveFileName(self, "Save attendance CSV", "attendance.csv", "CSV Files (*.csv)")
        if path:
            attendance_df.to_csv(path, index=False)
            QMessageBox.information(self, "Saved", f"Attendance exported to {path}")

    # -----------------------------
    # Misc
    # -----------------------------
    def on_status(self, text):
        self.status_label.setText(text)

    def close_app(self):
        try:
            if self.rec_thread.isRunning():
                self.rec_thread.stop()
        except Exception:
            pass
        try:
            if self.reg_thread.isRunning():
                self.reg_thread.stop()
        except Exception:
            pass
        self.close()


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())