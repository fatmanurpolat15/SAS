# 🌊 Underwater Object Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange)

## 📖 Overview

SAS Mine Classification System is an artificial intelligence project developed to classify **Side-Scan Sonar (SAS) underwater images** using the YOLOv8 deep learning model.

The system automatically classifies sonar images into the following categories:

- 💣 Mine
- 🪨 Rock
- 🔩 Wreck
- 🌊 Other

The project includes the complete machine learning workflow, including synthetic dataset generation, dataset preparation, model training, evaluation, and an interactive Streamlit application for prediction.

---

## ✨ Features

- Underwater sonar image classification
- YOLOv8 image classification model
- Synthetic dataset generation
- Automated training pipeline
- Interactive Streamlit web interface
- Image preprocessing using CLAHE
- Prediction confidence visualization

---

## 🛠️ Technologies

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- Streamlit
- OpenCV
- NumPy
- Pillow

---

## 📂 Project Structure

```text
sas_project
│
├── app.py                 # Streamlit application
├── train.py               # Model training
├── run_pipeline.py        # End-to-end pipeline
├── prepare_dataset.py     # Dataset preparation
├── generate_synthetic.py  # Synthetic data generation
├── requirements.txt
└── dataset/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/fatmanurpolat15/SAS.git
```

Go to the project directory

```bash
cd SAS/sas_project
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the complete pipeline:

```bash
python run_pipeline.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

## 🎯 Classification Categories

| Class | Description |
|------|-------------|
| 💣 Mine | Underwater mine |
| 🪨 Rock | Rock formation |
| 🔩 Wreck | Shipwreck or debris |
| 🌊 Other | Other underwater objects |

---

## 👩‍💻 Contributors

Developed as a **team project**.

- Fatmanur Polat
