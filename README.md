# Face Recognition Attendance System

A small desktop attendance application using OpenCV, `face_recognition`, PyQt5, and a local CSV file. It reads enrolled face images from `known_faces/`, marks attendance locally, and can play a short sound when a person is recorded.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python face_attendance_pyqt.py
```

Add authorized `.jpg`, `.jpeg`, or `.png` enrollment images to `known_faces/` before launching. The application creates `attendance.csv` locally.

## Privacy

Face images, attendance records, and personal project documents are excluded from this repository. Use biometric data only with the people’s knowledge and permission, and follow applicable privacy rules.
