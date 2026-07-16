# 🌊 Side-Scan Sonar Object Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-red?logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📖 Overview

**Side-Scan Sonar Object Classification** is an AI-powered computer vision project developed to classify underwater objects from **Side-Scan Sonar (SSS)** imagery using the **YOLOv8** deep learning model.

The system classifies sonar images into four categories:

- 💣 Mine
- 🪨 Rock
- 🚢 Wreck
- 🌊 Other

The project includes the complete machine learning workflow, including synthetic dataset generation, dataset preparation, model training, model evaluation, and real-time prediction through an interactive Streamlit application.

---

## ✨ Features

- 🌊 Underwater object classification from Side-Scan Sonar images
- 🤖 YOLOv8 deep learning classification model
- ⚙️ Synthetic dataset generation
- 📊 Model training and evaluation
- 🖥️ Interactive Streamlit web application
- 📈 Prediction confidence visualization
- 🧩 Modular machine learning pipeline

---

## 🛠️ Tech Stack

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
side-scan-sonar-object-classification
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
git clone https://github.com/fatmanurpolat15/side-scan-sonar-object-classification.git
```

Move into the project directory

```bash
cd side-scan-sonar-object-classification/sas_project
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

## 💡 Project Highlights

- AI-powered sonar image analysis
- Deep learning with YOLOv8
- End-to-end machine learning workflow
- Interactive Streamlit interface
- Automated dataset preparation and prediction pipeline

---

## 👩‍💻 Contributors

Developed collaboratively as part of a team project.

- Fatmanur Polat
