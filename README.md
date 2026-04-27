# 🎥 Anomaly Detection for Video Surveillance

A Python-based desktop application that uses a trained Convolutional Neural Network (CNN) to detect anomalous or abnormal events (such as accidents or falls) in surveillance video footage — displayed through a full-featured GUI with user authentication.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [How It Works](#how-it-works)
- [Model Details](#model-details)
- [Database Schema](#database-schema)
- [File Descriptions](#file-descriptions)
- [Known Issues & Limitations](#known-issues--limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 📌 Overview

This project is a **real-time video anomaly detection system** built as a desktop GUI application using Python. It processes `.mp4` video files frame by frame, passes each frame through a pre-trained deep learning model (`abnormalevent.h5`), and classifies each frame as either:

- ✅ **No event detected** (normal activity)
- 🚨 **Accident event Detected** (anomaly/abnormal activity)

The system is secured with a user **registration and login system** backed by an SQLite database, making it suitable for controlled access environments.

---

## ✨ Features

- 🔐 **User Authentication** — Register and login before accessing the detection system
- 🖥️ **Tkinter GUI** — Full-screen desktop interface with background images and styled buttons
- 📹 **Video Upload** — Browse and select `.mp4` video files for analysis
- 🤖 **CNN-Based Detection** — Frame-by-frame classification using a Keras `.h5` model
- 🏷️ **Real-Time Annotation** — Each frame is annotated with frame number and detection label
- 🗃️ **SQLite Database** — Persistent user data storage with no external database required
- 🔒 **Password Validation** — Enforces strong passwords (uppercase, number, special character)

---

## 📁 Project Structure

```
Anomaly-Detection-for-Video-Surveillance-main/
│
├── anomalyGUI_main.py          # Main entry point — splash screen with Login/Register buttons
├── GUI_Master.py               # Core detection screen with video upload and CNN inference
├── anomaly_registration.py     # User registration form with validation
├── login.py                    # Login form — authenticates and routes to GUI_Master.py
│
├── abnormalevent.h5            # Pre-trained Keras CNN model (~406 KB)
│
├── frame0.jpg – frame108.jpg   # Sample video frames (used for testing/reference)
│
├── README.md                   # Original brief readme
└── LICENSE                     # GNU General Public License v3.0
```

> **Note:** The application expects image assets (`new5.jpg`, `back5.jpg`, `new3.jpg`) in the working directory that are not included in this repository. You will need to supply your own background images.

---

## 🛠️ Tech Stack

| Component         | Technology                        |
|-------------------|-----------------------------------|
| GUI Framework     | Python `tkinter` + `ttk`          |
| Image Processing  | `OpenCV (cv2)`, `Pillow (PIL)`    |
| Deep Learning     | `Keras` (TensorFlow backend)      |
| Database          | `SQLite3` (built-in Python)       |
| Numerical Ops     | `NumPy`                           |
| Language          | Python 3.8+                       |

---

## ✅ Prerequisites

Make sure you have **Python 3.8** or higher installed. The following packages are required:

```
tensorflow / keras
opencv-python
Pillow
numpy
tkinter (usually bundled with Python)
sqlite3 (built-in)
```

---

## 🚀 Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/Anomaly-Detection-for-Video-Surveillance.git
cd Anomaly-Detection-for-Video-Surveillance
```

**2. (Optional) Create and activate a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install tensorflow opencv-python Pillow numpy
```

**4. Add required background images**

Place the following image files in the project root directory (any `.jpg` images of your choice):

- `new5.jpg` — background for the main splash screen
- `new3.jpg` — background for the registration form
- `back5.jpg` — background for the detection dashboard

**5. Update the model path in `GUI_Master.py`**

On line ~60 of `GUI_Master.py`, update the hardcoded model path to match your system:

```python
# Change this:
FALLModel = load_model('C:/Anomaly Detection 2/abnormalevent.h5')

# To this (relative path):
FALLModel = load_model('abnormalevent.h5')
```

---

## ▶️ Running the Application

Start the application from the project root:

```bash
python anomalyGUI_main.py
```

### Application Flow

```
anomalyGUI_main.py
     │
     ├── [REGISTER] ──► anomaly_registration.py ──► login.py
     │
     └── [LOGIN]    ──► login.py ──► GUI_Master.py (Detection Dashboard)
```

1. **Main Screen** — Choose to Register or Login
2. **Register** — Fill in your details; account is saved to `evaluation.db`
3. **Login** — Enter credentials to access the detection dashboard
4. **Detection Dashboard** — Click "Anomaly Detection System", select an `.mp4` file, and watch the model annotate each frame in real time

---

## ⚙️ How It Works

### Frame-by-Frame Inference Pipeline

```
MP4 Video
    │
    ▼
cv2.VideoCapture()        ← Read frames one at a time
    │
    ▼
Resize to 64×64           ← Model input size
    │
    ▼
Convert to Grayscale      ← cv2.COLOR_BGR2GRAY
    │
    ▼
Normalize (÷ 255)         ← Scale pixel values to [0, 1]
    │
    ▼
Reshape → (1, 64, 64, 1)  ← CNN input tensor
    │
    ▼
CNN Model Prediction      ← abnormalevent.h5
    │
    ├── predicted[0][0] < 0.5  →  "Accident event Detected" (RED label)
    └── predicted[0][0] >= 0.5 →  "No event detected" (GREEN label)
    │
    ▼
cv2.putText() on frame    ← Annotate with frame number + label
    │
    ▼
cv2.imshow('FDD', frame)  ← Display annotated frame
```

Press **ESC** at any time to stop playback.

---

## 🧠 Model Details

| Property         | Value                          |
|------------------|-------------------------------|
| File             | `abnormalevent.h5`            |
| Format           | Keras HDF5 model              |
| Input Shape      | `(1, 64, 64, 1)` — grayscale  |
| Output           | 2-class softmax (normal / anomaly) |
| File Size        | ~406 KB                       |
| Architecture     | CNN (Fall/Anomaly Detection)  |

The model outputs a 2-element probability array:
- `predicted[0][0]` — probability of **normal** activity
- `predicted[0][1]` — probability of **anomaly**

Classification threshold is set at `0.5` on `predicted[0][0]`.

---

## 🗄️ Database Schema

The app creates a local SQLite database file `evaluation.db` automatically on first run.

**Table: `registration`**

| Column     | Type | Description                        |
|------------|------|------------------------------------|
| `Fullname` | TEXT | User's full name                   |
| `address`  | TEXT | User's address                     |
| `username` | TEXT | Unique login username              |
| `Email`    | TEXT | Valid email address                |
| `Phoneno`  | TEXT | 10-digit phone number              |
| `Gender`   | TEXT | Male / Female (radio button value) |
| `age`      | TEXT | User age (1–100)                   |
| `password` | TEXT | Password (stored as plain text)    |

> ⚠️ **Security Note:** Passwords are currently stored as **plain text** in the database. For any real-world deployment, passwords must be hashed using a library like `bcrypt` or `hashlib`.

### Password Requirements

The registration form enforces:
- Minimum 6 characters, maximum 20 characters
- At least one **uppercase** letter
- At least one **lowercase** letter
- At least one **digit**
- At least one **special symbol**: `$`, `@`, `#`, or `%`

---

## 📄 File Descriptions

| File | Purpose |
|------|---------|
| `anomalyGUI_main.py` | Entry point. Launches a full-screen splash window with Login, Register, and Exit buttons. |
| `GUI_Master.py` | Main detection dashboard. Handles video file selection, CNN inference loop, and annotated frame display. |
| `anomaly_registration.py` | Registration form with full field validation. Saves user to SQLite on success, then redirects to login. |
| `login.py` | Login window. Queries SQLite for matching credentials; on success, launches `GUI_Master.py`. |
| `abnormalevent.h5` | Pre-trained Keras CNN model for binary anomaly classification. |
| `frame*.jpg` | ~95 sample JPEG frames extracted from a surveillance video, likely used during model training/testing. |

---

## ⚠️ Known Issues & Limitations

- **Hardcoded model path** — `GUI_Master.py` uses an absolute Windows path (`C:/Anomaly Detection 2/...`) for loading the model. This must be updated before running on any other machine.
- **Missing background images** — `new5.jpg`, `back5.jpg`, and `new3.jpg` are referenced but not included in the repository. The app will crash on launch without them.
- **Plain-text passwords** — User passwords are stored without hashing, which is a security risk.
- **Windows-specific path separators** — `basepath` in `GUI_Master.py` uses `\\` which will fail on macOS/Linux.
- **No live webcam support** — The detection currently only works on pre-recorded `.mp4` files (webcam code is commented out).
- **No logout functionality** — Once logged in, there is no way to log out without closing the window.
- **No model training UI** — The train model button and related code are commented out; training must be done separately.

---

## 🔮 Future Improvements

- [ ] Add password hashing (e.g., `bcrypt`) for secure credential storage
- [ ] Fix cross-platform path handling using `os.path.join()`
- [ ] Add live webcam/RTSP stream support for real-time surveillance
- [ ] Bundle background images or allow the user to configure them from the UI
- [ ] Add a logout button and session management
- [ ] Display confidence scores alongside detection labels
- [ ] Add email alerts when an anomaly is detected
- [ ] Package as a standalone executable using `PyInstaller`
- [ ] Add a model training interface so users can retrain on custom datasets

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0**.  
See the [LICENSE](LICENSE) file for full details.

---

*Built with Python, Keras, OpenCV, and Tkinter.*
