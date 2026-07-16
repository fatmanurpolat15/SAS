# 🌊 Underwater Object Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-red?logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📖 Overview

**Underwater Object Classification** is an AI-powered computer vision project developed to classify underwater objects from **Side-Scan Sonar (SAS)** imagery using the **YOLOv8** deep learning model.

The system classifies sonar images into four categories:

- 💣 Mine
- 🪨 Rock
- 🚢 Wreck
- 🌊 Other

The project includes the complete machine learning workflow, from synthetic dataset generation and dataset preparation to model training, evaluation, and prediction through an interactive Streamlit interface.

---

## ✨ Features

- 🌊 Underwater object classification
- 📡 Side-Scan Sonar (SAS) image processing
- 🤖 YOLOv8 image classification model
- ⚙️ Synthetic dataset generation
- 📊 Model training and evaluation
- 🖥️ Interactive Streamlit web application
- 📈 Prediction confidence visualization

---

## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- Streamlit
- NumPy
- Pillow

---

## 📂 Project Structure

```text
underwater-object-classification
│
├── README.md
└── sas_project
    ├── app.py
    ├── train.py
    ├── run_pipeline.py
    ├── prepare_dataset.py
    ├── generate_synthetic.py
    ├── requirements.txt
    └── ...
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/fatmanurpolat15/underwater-object-classification.git
```

Move into the project directory

```bash
cd underwater-object-classification/sas_project
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the complete pipeline

```bash
python run_pipeline.py
```

Launch the Streamlit application

```bash
streamlit run app.py
```

---

## 🎯 Classification Categories

| Category | Description |
|----------|-------------|
| 💣 Mine | Underwater mine |
| 🪨 Rock | Rock formation |
| 🚢 Wreck | Shipwreck / underwater debris |
| 🌊 Other | Other underwater objects |

---

## 👩‍💻 Contributors

This project was developed as a **team project**.

- Fatmanur Polat
