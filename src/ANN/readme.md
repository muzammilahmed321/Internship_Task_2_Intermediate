<div align="center">

# 🧠 ANN Customer Churn Prediction

### *Predicting Bank Customer Churn with a Deep Learning Neural Network*

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-2.17-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Keras-3.4-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
<img src="https://img.shields.io/badge/Deep%20Learning-ANN-8A2BE2?style=for-the-badge"/>

<img src="https://img.shields.io/badge/Test%20Accuracy-85.15%25-2ECC71?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Test%20Loss-0.3571-E67E22?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Completed-2ECC71?style=for-the-badge"/>

</div>

<br>

> A complete, end-to-end Deep Learning pipeline — from raw tabular data to a trained,
> evaluated, and saved neural network — built to predict whether a bank customer
> will **stay** or **churn**.

<br>

---

## 📖 Table of Contents

| | | | |
|---|---|---|---|
| [📌 Overview](#-project-overview) | [🎯 Objectives](#-objectives) | [🖼 Dataset](#-dataset-information) | [🏗 Structure](#-project-structure) |
| [🧠 Architecture](#-ann-architecture) | [🔄 Workflow](#-project-workflow) | [⚙ Hyperparameters](#-hyperparameters) | [📊 Output](#-actual-output-sample-run) |
| [🎯 Results](#-final-results) | [🚀 Installation](#-installation) | [🔮 Roadmap](#-future-improvements) | [👨‍💻 Author](#-author) |

---

## 📌 Project Overview

This project implements an **Artificial Neural Network (ANN)** using **TensorFlow/Keras** to predict **bank customer churn** — whether a customer will exit the bank or stay — using the **Churn_Modelling** dataset (10,000 real-world bank customer records).

<table>
<tr>
<td width="50%" valign="top">

### 🔧 What was built
- End-to-end OOP-based Deep Learning pipeline
- Custom data cleaning & encoding logic
- 4-layer fully connected neural network
- Full evaluation suite (confusion matrix, classification report, training curves)

</td>
<td width="50%" valign="top">

### 🎓 Why it was built
- Internship deliverable — **Intermediate Level: ANN**
- Hands-on practice with tabular Deep Learning
- Practical exposure to real training/debugging workflows

</td>
</tr>
</table>

---

## 🎯 Objectives

- 🧩 Learn how an ANN works internally on tabular data
- 🏗 Build an ANN from scratch using TensorFlow
- 🧹 Clean and encode categorical + numerical features correctly
- 🚂 Train a Deep Learning model on real customer data
- 📏 Evaluate performance using precision, recall, and F1-score — not accuracy alone
- 💾 Save the trained model for reuse
- 🔍 Understand every stage of a Deep Learning pipeline, end to end

---

## 🖼 Dataset Information

### 📁 `Churn_Modelling.csv`

<div align="center">

| 🔢 Property | 📊 Value |
|:---|:---:|
| **Total Records** | 10,000 |
| **Training Records** | 8,000 (80%) |
| **Testing Records** | 2,000 (20%) |
| **Original Features** | 14 |
| **Features Used** | 11 (after cleaning) |
| **Target Column** | `Exited` |
| **Classes** | 2 — Stayed / Exited |

</div>

### 🧬 Feature Columns

```
✔ CreditScore
✔ Geography        → One-Hot Encoded  (France / Spain / Germany)
✔ Gender            → Label Encoded    (Male / Female)
✔ Age
✔ Tenure
✔ Balance
✔ NumOfProducts
✔ HasCrCard
✔ IsActiveMember
✔ EstimatedSalary
✔ Exited            → 🎯 Target Variable
```

❌ **Dropped columns:** `RowNumber`, `CustomerId`, `Surname` — pure identifiers with zero predictive value.

---

## 🏗 Project Structure

```text
Task_2_Intermediate
│
├── 📂 datasets
│   └── 📄 Churn_Modelling.csv
│
├── 📂 models
│   └── 📂 ANN
│       └── 🧠 ann_model.keras
│
├── 📂 outputs
│   └── 📂 ANN
│       ├── 🖼 confusion_matrix.png
│       ├── 📈 accuracy_curve.png
│       └── 📉 loss_curve.png
│
├── 📂 report
│
├── 📂 src
│   └── 📂 ANN
│       ├── 🐍 Main_ANN.py
│       ├── 🐍 ANN_Model.py
│       ├── 🐍 ANN_Training.py
│       └── 🐍 ANN_Evaluation.py
│
├── 📄 requirements.txt
│
└── 📄 README.md
```

---

## 🧠 ANN Architecture

<div align="center">

```
                    ┌──────────────────────────┐
                    │       INPUT LAYER          │
                    │      11 Features           │
                    └────────────┬──────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   DENSE — 6 Neurons        │
                    │        ReLU                │
                    └────────────┬──────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   DROPOUT — rate 0.2       │
                    └────────────┬──────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   DENSE — 6 Neurons        │
                    │        ReLU                │
                    └────────────┬──────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   DROPOUT — rate 0.2       │
                    └────────────┬──────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   DENSE — 1 Neuron         │
                    │       Sigmoid               │
                    └────────────┬──────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  🎯 PREDICTED CLASS        │
                    │   Stayed  /  Exited        │
                    └──────────────────────────┘
```

</div>

---

## 🔄 Project Workflow

<div align="center">

| Step | Stage | Description |
|:---:|:---|:---|
| 1️⃣ | **Load Dataset** | Read `Churn_Modelling.csv` into a DataFrame |
| 2️⃣ | **Clean & Encode** | Drop identifiers, encode `Geography` & `Gender` |
| 3️⃣ | **Split & Scale** | 80/20 stratified split + `StandardScaler` |
| 4️⃣ | **Build ANN** | Construct the Sequential model |
| 5️⃣ | **Compile** | Adam optimizer + Binary Crossentropy |
| 6️⃣ | **Train** | 100 epochs, batch size 32 |
| 7️⃣ | **Evaluate** | Metrics, confusion matrix, curves |
| 8️⃣ | **Save** | Persist model as `.keras` file |

</div>

---

## 📂 File Description

<table>
<tr><th width="25%">File</th><th>Responsibilities</th></tr>
<tr>
<td><code>Main_ANN.py</code></td>
<td>🚀 Entry point — orchestrates the Training and Evaluation pipelines</td>
</tr>
<tr>
<td><code>ANN_Model.py</code></td>
<td>🏗 Builds the ANN architecture — Dense, Dropout, Output layers</td>
</tr>
<tr>
<td><code>ANN_Training.py</code></td>
<td>📥 Loads data, cleans & encodes, scales, compiles, trains, and saves the model</td>
</tr>
<tr>
<td><code>ANN_Evaluation.py</code></td>
<td>📊 Evaluates the model, generates confusion matrix, classification report, and training curves</td>
</tr>
</table>

---

## ⚙ Hyperparameters

<div align="center">

| ⚙ Parameter | 🎛 Value |
|:---|:---:|
| Optimizer | `Adam` (lr = 0.001) |
| Loss Function | `Binary Crossentropy` |
| Activation (Hidden) | `ReLU` |
| Output Activation | `Sigmoid` |
| Epochs | `100` |
| Batch Size | `32` |
| Validation Split | `0.2` |
| Dropout Rate | `0.2` |

</div>

---

## 🧮 Deep Learning Pipeline

```
📊 Customer Data
        ↓
🧹 Cleaning + Encoding + Scaling
        ↓
🧠 ANN
        ↓
🎯 Prediction (Probability)
        ↓
📉 Loss Calculation
        ↓
🔄 Backpropagation
        ↓
⚖️ Weight Update
        ↓
✅ Improved Prediction
```

---

## 📊 Actual Output (Sample Run)

<details>
<summary><b>🖥 Click to expand full training log</b></summary>

```
Dataset Loaded Successfully
Data Shape : (10000, 14)

Data Cleaning Completed
Data Preparation Completed

ANN Model Created Successfully
Model Compiled Successfully

Training Started...

Epoch 1/100
200/200 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.5822 - loss: 0.6661 - val_accuracy: 0.7925 - val_loss: 0.5194
...
Epoch 50/100
200/200 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8354 - loss: 0.3954 - val_accuracy: 0.8525 - val_loss: 0.3577
...
Epoch 100/100
200/200 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8343 - loss: 0.3909 - val_accuracy: 0.8544 - val_loss: 0.3555

Training Completed
Model Saved Successfully

Test Loss     : 0.3571
Test Accuracy : 0.8515

Confusion Matrix Saved

classification report
              precision    recall  f1-score   support

           0       0.85      0.98      0.91      1593
           1       0.82      0.34      0.49       407

    accuracy                           0.85      2000
   macro avg       0.84      0.66      0.70      2000
weighted avg       0.85      0.85      0.83      2000

f1_score  = 0.6993
accuracy  = 0.8515
recall    = 0.6626
precision = 0.8388

Training History Plots Saved

Project Completed Successfully.
Model has been saved in models/ANN/
Graphs have been saved in outputs/
```

</details>

---

## 📈 Model Summary

<div align="center">

| Layer | Output | Role |
|:---|:---|:---|
| `Dense (6, ReLU)` | Hidden Layer 1 | Extracts feature combinations |
| `Dropout (0.2)` | Regularization | Prevents overfitting |
| `Dense (6, ReLU)` | Hidden Layer 2 | Refines learned patterns |
| `Dropout (0.2)` | Regularization | Prevents overfitting |
| `Dense (1, Sigmoid)` | Output Layer | Binary churn probability |

</div>

---

## 🎯 Final Results

<div align="center">

### Overall Performance

| Metric | Value | |
|:---|:---:|:---|
| **Test Accuracy** | `85.15%` | 🟢🟢🟢🟢🟢🟢🟢🟢⬜⬜ |
| **Test Loss** | `0.3571` | 🟢 Low |
| **Precision (macro)** | `0.8388` | 🟢🟢🟢🟢🟢🟢🟢🟢⬜⬜ |
| **Recall (macro)** | `0.6626` | 🟡🟡🟡🟡🟡🟡🟡⬜⬜⬜ |
| **F1-score (macro)** | `0.6993` | 🟡🟡🟡🟡🟡🟡🟡⬜⬜⬜ |

### Per-Class Breakdown

| Class | Precision | Recall | F1-score | Support |
|:---|:---:|:---:|:---:|:---:|
| 🟩 **0 — Stayed** | 0.85 | 0.98 | 0.91 | 1,593 |
| 🟥 **1 — Exited** | 0.82 | 0.34 | 0.49 | 407 |

</div>

> 💡 **Observation:** The model performs strongly on the majority class (customers who stayed) but shows lower recall on churned customers — expected given the dataset's class imbalance (~20% churn rate). Class weighting, oversampling (SMOTE), or threshold tuning are natural next steps to improve minority-class recall.

---

## 🚀 Installation

### 1️⃣ Clone the repository
```bash
git clone <repository-url>
```

### 2️⃣ Move to project directory
```bash
cd Task_2_Intermediate
```

### 3️⃣ Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 5️⃣ Run the project
```bash
cd src/ANN
python Main_ANN.py
```

<details>
<summary><b>📦 requirements.txt</b></summary>

```
tensorflow==2.17.0
numpy>=1.26.0,<2.0.0
pandas>=2.2.0
matplotlib>=3.9.0
scikit-learn>=1.5.0
opencv-python>=4.10.0
seaborn>=0.13.2
joblib>=1.4.2
```

> ⚠️ Keras is intentionally **not** pinned separately — TensorFlow installs a matching version automatically. This avoided a known circular-import bug in the TensorFlow 2.21 / Keras 3.15 combination.

</details>

---

## 📷 Sample Outputs

Saved automatically inside:

```
outputs/ANN/
├── 🖼 confusion_matrix.png
├── 📈 accuracy_curve.png
└── 📉 loss_curve.png
```

---

## 🔮 Future Improvements

| Category | Improvement |
|:---|:---|
| ⚖️ Model Quality | Handle class imbalance (SMOTE / class weights / threshold tuning) |
| 🎛 Tuning | Hyperparameter tuning — units, learning rate, batch size |
| 🛑 Training | Early Stopping & Model Checkpointing |
| 🔁 Validation | K-Fold Cross-Validation |
| 📉 Scheduling | Learning Rate Scheduler |
| 📊 Monitoring | TensorBoard Integration |
| 🧭 Engineering | Standardize output paths relative to project root |

---

## 👨‍💻 Author

<div align="center">

### **Muzammil**
**BS Computer Science** — NED University of Engineering & Technology, Karachi
*AI Engineering Intern*

</div>

---

## 📚 References

- 📘 TensorFlow Documentation
- 📘 Keras Documentation
- 📊 Churn_Modelling Dataset
- 📖 *Deep Learning* — Ian Goodfellow
- 📖 *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*

---

## ⭐ Learning Outcomes

<div align="center">

| ✅ | ✅ | ✅ |
|:---|:---|:---|
| Data Cleaning & Encoding | Feature Scaling | Dense (Fully Connected) Layers |
| Dropout Regularization | Activation Functions | Loss Function |
| Optimizer | Backpropagation | Model Evaluation (Precision, Recall, F1) |
| Saving Deep Learning Models | | |

</div>

---

<div align="center">

**⭐ If this project helped you understand ANNs better, consider giving it a star!**

</div>