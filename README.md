# Face Recognition Attendance System

A desktop attendance app built with Python, OpenCV, `face_recognition`, PyQt5, and pandas. It uses a webcam to recognize enrolled people and records each recognized person once per calendar day in a local CSV file.

## Features

- **Live recognition:** Starts and stops the webcam from the Home tab, draws a box around each detected face, and labels it with the matched name or `Unknown`.
- **Face enrollment:** Captures a webcam frame from the Register Face tab and saves a timestamped image under `known_faces/` using the entered name.
- **Duplicate prevention:** Records a person's first recognized attendance for the day; later detections that day do not add another row.
- **Attendance view:** Shows the locally recorded name, date, and time in the Attendance tab.
- **CSV export:** Saves the attendance table to a CSV path you choose.
- **Reload enrolled faces:** Refreshes the recognition thread after adding or changing face images.
- **Local processing:** The application does not send face images or attendance records over a network.

## Requirements

- Python 3
- A webcam
- Camera access enabled for your terminal or Python app

The `face-recognition` package depends on native computer-vision components. Installation can vary by operating system and Python version.

## Installation

Run these commands from the repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python face_attendance_pyqt.py
```

Launch it from the repository folder so the app uses the expected local paths.

## Enroll a person

1. Open **Register Face**.
2. Enter the person's name or identifier.
3. Select **Start Camera** and position the person in view.
4. Select **Capture & Save**. The app saves one timestamped JPEG to `known_faces/`.
5. Select **Stop Camera** when finished.

Use a clear, well-lit image with one face visible. The recognition code loads the first face encoding it finds in each supported image (`.jpg`, `.jpeg`, or `.png`). Select **Reload Known Faces** on the Home tab if you add images outside the enrollment flow.

## Record and export attendance

1. Open **Home** and select **Start Recognition**.
2. The app labels detected faces. Unmatched faces appear as `Unknown` and are not recorded.
3. Open **Attendance** to review the local table.
4. Select **Export CSV** to save a copy to a location you choose.

The app also writes its working attendance file as `attendance.csv` in the repository folder. It creates this file when a known face is first recorded.

## Project files

```text
face_attendance_pyqt.py   Main PyQt5 application
requirements.txt          Python dependencies
beep.mp3                  Attendance feedback sound
known_faces/              Local enrollment photos (ignored by Git)
attendance.csv            Local attendance log (ignored by Git)
```

## Privacy and limitations

Face images are biometric data. Get informed permission before enrolling anyone, restrict access to the computer and local files, and delete records when they are no longer needed. Enrollment photos and attendance logs are excluded from this Git repository by `.gitignore`.

This is a basic educational project, not a secure access-control or production attendance system. It has no liveness or anti-spoofing check, and recognition can be affected by camera quality, lighting, pose, and the enrollment image. Review attendance records before relying on them.
