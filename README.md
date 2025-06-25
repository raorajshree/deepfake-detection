


# 🧠 DeepFake Face Detection using ResNext & LSTM

This project is designed to detect DeepFake videos by identifying spatial and temporal inconsistencies in facial frames. It leverages a hybrid **ResNext50 CNN** for frame-level feature extraction and **LSTM** for temporal sequence modeling.

---

## 📌 Overview

With the rise of synthetic media, detecting manipulated videos (DeepFakes) has become a critical challenge. This project provides:

- A **trained deep learning model** to detect DeepFake videos.
- A **Django web application** for users to upload videos and get predictions.
- A complete training and inference pipeline using **PyTorch** and **OpenCV**.

---

## 🛠️ Tech Stack

| Component        | Technology                  |
|------------------|------------------------------|
| Backend          | Python, Django, PyTorch      |
| Deep Learning    | ResNext50 + LSTM             |
| Video Processing | OpenCV                       |
| Deployment       | Web Interface (Django), CLI  |

---

## 📁 Project Structure

DeepFake-Detection/
├── model/                  # Training code & architecture
│   ├── train\_model.py
│   ├── resnext\_lstm\_model.py
│   └── utils.py
├── webapp/                 # Django Web App
│   ├── manage.py
│   ├── detector/
│   │   ├── detector.py
│   │   ├── views.py
│   │   └── ...
│   └── templates/
├── sample\_input/           # Sample video
├── sample\_output/          # Prediction result (screenshot)
├── requirements.txt
├── README.md
└── .gitignore



---

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/your-username/DeepFake-Detection.git
cd DeepFake-Detection
pip install -r requirements.txt
```

### 🏁 Run Web Application

```bash
cd webapp
python manage.py runserver
```

Then open `http://127.0.0.1:8000` in your browser.

---

## 🧪 Model Architecture

* **ResNext50\_32x4d**: Extracts 2048-dimensional feature vectors from each face-cropped frame.
* **LSTM (2048 units)**: Analyzes temporal dependencies across frames.
* Final **sigmoid layer** classifies videos as *Real* or *Fake*.

---

## 🎯 Dataset

This project supports custom datasets and public datasets like:

* [FaceForensics++](https://github.com/ondyari/FaceForensics)
* [DeepFake Detection Challenge (DFDC)](https://www.kaggle.com/c/deepfake-detection-challenge)

---

## 📷 Sample Output

<p align="center">
  <img src="sample_output/prediction_screenshot.png" width="500"/>
</p>

---

## 🧠 Future Work

* Add support for real-time video stream detection.
* Deploy as a browser plugin or mobile app.
* Improve detection robustness against GAN-generated videos.

---

## 👥 Contributors


* S. Rajshree Rao


---

## 📄 License

This project is under the MIT License - see the [LICENSE](LICENSE) file for details.


