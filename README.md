<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6C63FF,100:00C2FF&height=200&section=header&text=Deep%20Learning%20Projects&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=AI%20Engineering%20Internship%20Portfolio&descAlignY=55&descSize=18"/>

### *A growing collection of Deep Learning projects — built from scratch with TensorFlow, Keras & Python*

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github)

<br>

![Status](https://img.shields.io/badge/Status-Active%20Development-yellow?style=for-the-badge)
![Projects](https://img.shields.io/badge/Completed%20Projects-4-success?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Deep%20Learning-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

<br>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muzammil-ahmed-795527271/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/muzammilahmed321)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:muzammilahmed1h2@gmail.com)

</div>

---

## 🌟 About This Repository

This repository documents my hands-on journey through **Deep Learning**, where I implement neural network architectures from the ground up using **TensorFlow** and **Keras** — following clean coding practices, object-oriented design, and professional project structure.

Every project follows the **same rigorous pipeline**: data preprocessing → model design → training → evaluation → visualization → model persistence → documentation. The goal isn't just working code — it's a portfolio that shows I understand *why* each piece exists.

> 💡 **Philosophy:** *Understand before optimizing.* Every model here is built from scratch rather than copy-pasted, so the underlying concepts — not just the accuracy score — are the real deliverable.

---

## 📊 Portfolio Snapshot

<div align="center">

| Project | Type | Dataset | Key Metric | Status |
|:---|:---:|:---|:---:|:---:|
| 🧠 [**ANN**](#-1-artificial-neural-network--customer-churn-prediction) — Customer Churn | Binary Classification | Churn Modelling (10K records) | **85.15%** Accuracy | ✅ |
| 🖼 [**CNN**](#-2-convolutional-neural-network--cifar-10-image-classification) — Image Classification | Multi-Class Classification | CIFAR-10 (60K images) | **70.74%** Accuracy | ✅ |
| ✈️ [**RNN**](#-3-recurrent-neural-network--airline-passenger-forecasting) — Passenger Forecasting | Time-Series Regression | AirPassengers (144 records) | **R² = 0.89** | ✅ |
| 🌤️ [**LSTM**](#-4-lstm--karachi-temperature-prediction) — Temperature Prediction | Time-Series Regression | Pakistan Weather (Karachi) | **MAE = 3.42°C** | ✅ |

</div>

---

## 📑 Table of Contents

- [About This Repository](#-about-this-repository)
- [Portfolio Snapshot](#-portfolio-snapshot)
- [Objectives](#-objectives)
- [Project Details](#-project-details)
  - [1. ANN — Customer Churn Prediction](#-1-artificial-neural-network--customer-churn-prediction)
  - [2. CNN — CIFAR-10 Image Classification](#-2-convolutional-neural-network--cifar-10-image-classification)
  - [3. RNN — Airline Passenger Forecasting](#-3-recurrent-neural-network--airline-passenger-forecasting)
  - [4. LSTM — Karachi Temperature Prediction](#-4-lstm--karachi-temperature-prediction)
- [Repository Structure](#-repository-structure)
- [Technology Stack](#-technology-stack)
- [Skills Demonstrated](#-skills-demonstrated)
- [Installation & Usage](#-installation--usage)
- [Project Roadmap](#-project-roadmap)
- [About Me](#-about-me)
- [Connect With Me](#-connect-with-me)

---

## 🎯 Objectives

✅ Learn Deep Learning concepts from fundamentals  &nbsp;&nbsp; ✅ Implement neural networks from scratch  &nbsp;&nbsp; ✅ Master TensorFlow & Keras
✅ Follow Object-Oriented Programming  &nbsp;&nbsp; ✅ Practice clean, modular code  &nbsp;&nbsp; ✅ Compare architectures across domains
✅ Build reusable AI components  &nbsp;&nbsp; ✅ Build a recruiter-ready AI portfolio  &nbsp;&nbsp; ✅ Continuously improve through experimentation

---

## 📌 Project Details

### 🧠 1. Artificial Neural Network — Customer Churn Prediction

Predicts whether a bank customer will **stay or churn** using a 4-layer fully connected neural network trained on 10,000 real-world customer records.

<table>
<tr><td>

**Architecture:** `Input(11) → Dense(6, ReLU) → Dropout(0.2) → Dense(6, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)`

**Concepts:** Data cleaning · One-hot & label encoding · Feature scaling · Dropout regularization · Binary classification · Confusion matrix · Classification report

</td></tr>
</table>

| Metric | Result |
|:---|:---:|
| Test Accuracy | **85.15%** |
| Test Loss | 0.3571 |
| Precision (macro) | 0.8388 |
| Recall (macro) | 0.6626 |
| F1-score (macro) | 0.6993 |

📂 `src/ANN/` &nbsp;·&nbsp; 📄 [Full documentation](src/ANN/README.md)

---

### 🖼 2. Convolutional Neural Network — CIFAR-10 Image Classification

Classifies **32×32 RGB images** into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using a CNN built from scratch.

<table>
<tr><td>

**Architecture:** `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(10, Softmax)`

**Concepts:** Image normalization · Convolution & pooling · Feature extraction · Dropout · Multi-class classification · Confusion matrix

</td></tr>
</table>

| Metric | Result |
|:---|:---:|
| Test Accuracy | **70.74%** |
| Test Loss | 0.8553 |
| Training Images | 50,000 |
| Testing Images | 10,000 |

📂 `src/CNN/` &nbsp;·&nbsp; 📄 [Full documentation](src/CNN/README.md)

---

### ✈️ 3. Recurrent Neural Network — Airline Passenger Forecasting

Forecasts **next-month airline passenger traffic** from the previous 12 months of historical data using a SimpleRNN.

<table>
<tr><td>

**Architecture:** `Input(12 months) → SimpleRNN(100, tanh) → Dense(1)`

**Concepts:** Time-series sequence generation · MinMax normalization · Early stopping · Regression evaluation

</td></tr>
</table>

| Metric | Result |
|:---|:---:|
| R² Score | **0.8929** |
| MAE | 21.05 passengers |
| RMSE | 26.09 passengers |
| Test Loss (MSE, scaled) | 0.0025 |

📂 `src/RNN/` &nbsp;·&nbsp; 📄 [Full documentation](src/RNN/README.md)

---

### 🌤️ 4. LSTM — Karachi Temperature Prediction

Predicts **next-day average temperature** for Karachi from 30 days of historical weather data — the most complex pipeline in the repo (14 engineered features, chronological splitting, dual scalers).

<table>
<tr><td>

**Architecture:** `Input(30×14) → LSTM(100) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(25, ReLU) → Dense(1)`

**Concepts:** Time-based interpolation · Cyclic seasonal encoding · Lag & rolling features · Percentile outlier clipping · Chronological 70/15/15 split · Early stopping + LR scheduling

</td></tr>
</table>

| Metric | Result |
|:---|:---:|
| MAE | 3.42 °C |
| RMSE | 4.06 °C |
| R² | -0.0003 *(baseline — optimization in progress)* |
| Parameters | 77,501 |

📂 `src/LSTM/` &nbsp;·&nbsp; 📄 [Full documentation](src/LSTM/README.md)

---

## 🏗 Repository Structure

```text
Deep_Learning_Projects
│
├── 📂 datasets
│   ├── Churn_Modelling.csv
│   ├── CIFAR-10
│   ├── pakistan_weather_2000_2024.csv
│   └── AirPassengers.csv
│
├── 📂 models
│   ├── ANN/  CNN/  RNN/  LSTM/  ...
│
├── 📂 outputs
│   ├── ANN/  CNN/  RNN/  LSTM/  ...
│
├── 📂 reports
│
├── 📂 src
│   ├── ANN/   { Main_ANN.py, ANN_Model.py, ANN_Training.py, ANN_Evaluation.py, README.md }
│   ├── CNN/   { cnn_main.py, CNN_Model.py, CNN_Training.py, CNN_Evaluation.py, README.md }
│   ├── RNN/   { RNN_Main.py, RNN_Model.py, RNN_Training.py, RNN_Evaluation.py, README.md }
│   ├── LSTM/  { Data/, LSTM/, Main/, outputs/, README.md }
│   └── ...
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙ Technology Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=python,tensorflow,git,github,vscode" />

</div>

| Category | Tools |
|:---|:---|
| **Language** | Python |
| **Deep Learning** | TensorFlow, Keras |
| **Machine Learning** | Scikit-Learn |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib |
| **Computer Vision** | OpenCV |
| **Dev Tools** | Git, GitHub, VS Code |

---

## 🧠 Skills Demonstrated

<table>
<tr>
<td width="33%" valign="top">

**Programming**
- Python
- Object-Oriented Design
- Modular Architecture

</td>
<td width="33%" valign="top">

**Data Engineering**
- Cleaning & Encoding
- Feature Scaling
- Lag / Rolling / Seasonal Features
- Outlier Handling

</td>
<td width="33%" valign="top">

**Deep Learning**
- ANN, CNN, RNN, LSTM
- Dropout & Regularization
- Early Stopping / LR Scheduling
- Model Evaluation & Saving

</td>
</tr>
</table>

---

## 📦 Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/muzammilahmed321/Deep-Learning-Projects.git
cd Deep-Learning-Projects

# 2. Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

**Run any project:**

```bash
cd src/ANN   && python Main_ANN.py       # Customer Churn
cd src/CNN   && python cnn_main.py       # CIFAR-10
cd src/RNN   && python RNN_Main.py       # Passenger Forecasting
cd src/LSTM/Main && python Main_LSTM.py  # Temperature Prediction
```

---

## 🗺 Project Roadmap

| Model | Status |
|:---|:---:|
| Artificial Neural Network (ANN) | 🟢 Completed |
| Convolutional Neural Network (CNN) | 🟢 Completed |
| Recurrent Neural Network (RNN) | 🟢 Completed |
| Long Short-Term Memory (LSTM) | 🟢 Completed |
| Gated Recurrent Unit (GRU) | 🟡 Planned |
| Transfer Learning (ResNet / VGG / MobileNet) | 🟡 Planned |
| Autoencoders | 🟡 Planned |
| Generative Adversarial Networks (GAN) | 🟡 Planned |
| Vision Transformers (ViT) | 🟡 Planned |
| NLP / Text Classification | 🟡 Planned |
| Object Detection & Image Segmentation | 🟡 Planned |

---

## 💡 Why This Repository?

Every project follows industry-inspired practices: modular structure, reusable components, documentation-first development, and honest performance reporting — including a baseline model that still needs tuning (the LSTM). That's intentional: recruiters can see not just results, but the full engineering process, including where the next iteration is headed.

---

## 👨‍💻 About Me

<div align="center">
<img src="https://readme-typing-svg.herokuapp.com?font=Poppins&size=24&duration=3500&pause=1000&color=00C2FF&center=true&vCenter=true&width=650&lines=Muzammil+Ahmed;AI+Engineering+Intern;Deep+Learning+Enthusiast;Computer+Vision+%26+Time-Series+Learner;Building+AI+Projects+with+TensorFlow"/>
</div>

I'm **Muzammil Ahmed**, a Computer Science undergraduate at **NED University of Engineering & Technology, Karachi**, with a strong interest in Artificial Intelligence, Machine Learning, and Deep Learning. I enjoy building practical AI applications end-to-end — from raw data to a trained, evaluated, and documented model — and continuously expanding my skills through hands-on projects.

**🎓 Education:** BS Computer Science, NED University of Engineering & Technology, Karachi, Pakistan

**💻 Areas of Interest:** Artificial Intelligence · Deep Learning · Machine Learning · Computer Vision · Time-Series Forecasting · Full Stack Development

---

## 📬 Connect With Me

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muzammil-ahmed-795527271/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/muzammilahmed321)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:muzammilahmed1h2@gmail.com)

</div>

---

## 📜 License

This repository is licensed under the **MIT License** — free to use, modify, and distribute with appropriate credit.

---

<div align="center">

### ⭐ If this repository helped you, consider giving it a star!

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6C63FF,100:00C2FF&height=120&section=footer"/>

</div>
