# 🌤️ Karachi Temperature Prediction Using LSTM

<p align="center">

**Deep Learning • Time-Series Forecasting • Weather Prediction**

A complete end-to-end **Long Short-Term Memory (LSTM)** pipeline for predicting Karachi's average daily temperature from historical weather observations.

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge\&logo=tensorflow\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![LSTM](https://img.shields.io/badge/Deep%20Learning-LSTM-8A2BE2?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Baseline%20Completed-success?style=for-the-badge)

</p>

---

## 📌 Table of Contents

* [🌟 Project Overview](#-project-overview)
* [🎯 Objective](#-objective)
* [🧠 Why LSTM?](#-why-lstm)
* [📊 Dataset](#-dataset)
* [🌦️ Features](#️-features)
* [🔄 Data Pipeline](#-data-pipeline)
* [🧹 Data Preprocessing](#-data-preprocessing)
* [📈 Feature Engineering](#-feature-engineering)
* [📏 Scaling](#-scaling)
* [⏳ Sequence Generation](#-sequence-generation)
* [🧠 Model Architecture](#-model-architecture)
* [⚙️ Training Configuration](#️-training-configuration)
* [📊 Training Results](#-training-results)
* [📈 Visualizations](#-visualizations)
* [📁 Project Structure](#-project-structure)
* [🧩 Project Components](#-project-components)
* [💾 Model Output](#-model-output)
* [🚀 Installation](#-installation)
* [▶️ Running the Project](#️-running-the-project)
* [📋 Expected Output](#-expected-output)
* [🔮 Future Improvements](#-future-improvements)
* [📚 Learning Outcomes](#-learning-outcomes)
* [🏁 Conclusion](#-conclusion)
* [👨‍💻 Author](#-author)

---

# 🌟 Project Overview

Weather forecasting is a naturally sequential problem.

Today's temperature is influenced by previous weather conditions, seasonal cycles, atmospheric conditions, and recent temperature trends.

This project develops a **deep learning time-series forecasting system** that learns these patterns using an **LSTM neural network**.

The model takes the previous **30 days of weather observations** and predicts the next day's:

```text
Average Temperature (tavg)
```

The complete pipeline includes:

```text
Raw Weather Data
       ↓
Karachi Filtering
       ↓
Data Cleaning
       ↓
Missing Value Handling
       ↓
Feature Engineering
       ↓
Outlier Handling
       ↓
Chronological Split
       ↓
Min-Max Scaling
       ↓
30-Day Sequence Generation
       ↓
LSTM Training
       ↓
Prediction
       ↓
Visualization & Metrics
```

---

# 🎯 Objective

The main objective of this project is to build an LSTM-based system capable of learning temporal weather patterns and predicting:

> 🌡️ **Karachi's next-day average temperature**

using historical meteorological information.

### Key Goals

* Build a complete time-series preprocessing pipeline.
* Engineer meaningful temporal and seasonal features.
* Preserve chronological ordering during data splitting.
* Generate sequential input windows for LSTM.
* Train a multi-layer LSTM architecture.
* Use callbacks for controlled training.
* Save the trained model.
* Generate predictions.
* Visualize actual vs predicted temperatures.
* Establish a baseline for future optimization.

---

# 🧠 Why LSTM?

Traditional neural networks generally process individual observations without explicitly modeling temporal dependencies.

Weather data, however, looks like:

```text
Day 1 → Day 2 → Day 3 → Day 4 → ... → Day 30
```

The information from earlier days can influence the prediction for a future day.

LSTM networks are specifically designed for sequential data.

### LSTM can learn patterns such as:

🌡️ Previous temperature trends
💧 Humidity relationships
🌬️ Wind patterns
☁️ Cloud coverage
🌧️ Precipitation
🌤️ Seasonal changes
📅 Weekly temperature behavior

### Concept

```text
Previous 30 Days
       │
       ▼
┌─────────────────┐
│      LSTM       │
│                 │
│ Temporal        │
│ Pattern         │
│ Learning        │
└────────┬────────┘
         │
         ▼
Next-Day Temperature
```

---

# 📊 Dataset

The project uses historical weather data from:

```text
pakistan_weather_2000_2024.csv
```

### Original Dataset

| Property            |                     Value |
| ------------------- | ------------------------: |
| Total Records       |                **31,779** |
| Total Columns       |                    **27** |
| Coverage            |             **2000–2024** |
| Geographic Coverage | Multiple Pakistani cities |

For this project, the dataset is filtered specifically for:

```text
📍 Karachi
```

### Karachi Records

```text
5,844 records
```

After feature engineering:

```text
5,837 records
```

---

# 🌦️ Features

The model uses **14 input features**.

## Weather Features

| Feature       | Description               |
| ------------- | ------------------------- |
| `tmin`        | Minimum daily temperature |
| `tmax`        | Maximum daily temperature |
| `humidity`    | Relative humidity         |
| `pressure`    | Atmospheric pressure      |
| `wspd`        | Wind speed                |
| `prcp`        | Precipitation             |
| `dew_point`   | Dew point temperature     |
| `cloud_cover` | Cloud coverage            |
| `visibility`  | Visibility                |

## Temporal Features

| Feature     | Description                    |
| ----------- | ------------------------------ |
| `month_sin` | Cyclic representation of month |
| `month_cos` | Cyclic representation of month |

## Historical Temperature Features

| Feature      | Description                                |
| ------------ | ------------------------------------------ |
| `tavg_lag1`  | Previous day's average temperature         |
| `tavg_lag7`  | Temperature approximately one week earlier |
| `tavg_roll7` | Seven-day rolling average                  |

### 🎯 Target Variable

```text
tavg
```

The target represents:

> **Average daily temperature in °C**

---

# 🔄 Data Pipeline

The complete preprocessing workflow is:

```text
┌───────────────────────────────┐
│ Pakistan Weather Dataset      │
│ 31,779 × 27                   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Filter Karachi                │
│ 5,844 records                 │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Date Cleaning & Sorting       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Missing Value Handling        │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Feature Engineering           │
│ Seasonal + Lag + Rolling      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Percentile Outlier Clipping   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Chronological Split           │
│ 70% / 15% / 15%               │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Min-Max Scaling               │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 30-Day Sequence Generation    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ LSTM Neural Network           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Temperature Prediction        │
└───────────────────────────────┘
```

---

# 🧹 Data Preprocessing

## 1. City Filtering

The complete dataset contains multiple cities.

The preprocessing pipeline selects:

```python
city = "Karachi"
```

Only Karachi observations are used for model development.

---

## 2. Date Cleaning

Dates are converted using:

```python
pd.to_datetime()
```

Invalid dates are removed.

The dataset is then sorted chronologically.

This is extremely important for time-series forecasting.

---

## 3. Missing Value Handling

Weather data can contain missing observations.

The project uses **time-based interpolation**:

```python
interpolate(
    method="time",
    limit_direction="both"
)
```

Remaining missing values are handled using:

```python
ffill()
bfill()
```

### Pipeline

```text
Missing Value
     ↓
Time-Based Interpolation
     ↓
Forward Fill
     ↓
Backward Fill
     ↓
Clean Data
```

---

# 📈 Feature Engineering

Feature engineering is used to provide the LSTM with additional temporal information.

## 🌙 Seasonal Encoding

Month is represented using sine and cosine:

```python
month_sin = sin(2π × month / 12)

month_cos = cos(2π × month / 12)
```

This represents the circular nature of seasons.

For example:

```text
December → January
```

are correctly treated as neighboring points in the yearly cycle.

---

## ⏮️ Lag Features

### Previous Day

```text
tavg_lag1
```

Provides the temperature from the previous day.

### Previous Week

```text
tavg_lag7
```

Provides temperature information from approximately seven days earlier.

---

## 📊 Rolling Temperature

The model also receives:

```text
tavg_roll7
```

which represents the seven-day moving average.

This helps capture recent temperature trends while reducing the effect of short-term fluctuations.

---

# 📊 Outlier Handling

The project uses **percentile clipping** rather than deleting extreme weather observations.

Processed variables:

```text
humidity
pressure
wspd
visibility
prcp
```

The boundaries are:

```text
1st Percentile
      ↓
   Valid Range
      ↓
99th Percentile
```

Values outside this range are clipped using:

```python
np.clip()
```

### Why clipping?

Extreme weather observations may be genuine.

Instead of removing entire records, clipping limits the influence of extreme values while preserving the corresponding observations.

---

# 📚 Chronological Data Split

Time-series data should not normally be randomly shuffled before splitting.

The project therefore preserves chronological order.

```text
70% ───────────────► Training
15% ───────► Validation
15% ───────► Testing
```

### Processed Dataset

| Dataset    |   Records |
| ---------- | --------: |
| Training   | **4,085** |
| Validation |   **875** |
| Testing    |   **877** |

This ensures that later observations are not used to train the model.

---

# 📏 Scaling

The project uses:

```python
MinMaxScaler(feature_range=(0, 1))
```

Two independent scalers are used:

```text
scaler_x → Input Features
scaler_y → Target Temperature
```

The input scaler is fitted **only on training data**:

```python
scaler_x.fit_transform(X_train)
```

Validation and testing data are transformed using the already-fitted scaler:

```python
scaler_x.transform(X_val)
scaler_x.transform(X_test)
```

The same principle is applied to the target scaler.

This helps prevent information from the test set leaking into the training process.

---

# ⏳ Sequence Generation

The selected historical window is:

```text
30 Days
```

Therefore, each LSTM sample contains:

```text
30 time steps
×
14 features
```

### Input Shape

```text
(samples, 30, 14)
```

### Example

```text
Days 1 ─────────────────── Day 30
 │                             │
 └────────── 30 days ──────────┘
                │
                ▼
              LSTM
                │
                ▼
        Day 31 Prediction
```

The sliding-window approach then moves forward:

```text
Days 2–31  → Day 32
Days 3–32  → Day 33
Days 4–33  → Day 34
...
```

---

# 🧠 Model Architecture

The project uses a **two-layer LSTM architecture**.

```text
                    INPUT
                (30 × 14)
                     │
                     ▼
          ┌─────────────────────┐
          │      LSTM           │
          │      100 Units      │
          │ return_sequences    │
          └──────────┬──────────┘
                     │
                     ▼
               Dropout 20%
                     │
                     ▼
          ┌─────────────────────┐
          │      LSTM           │
          │       50 Units      │
          └──────────┬──────────┘
                     │
                     ▼
               Dropout 20%
                     │
                     ▼
          ┌─────────────────────┐
          │    Dense Layer      │
          │      25 Units       │
          │       ReLU          │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │    Output Layer     │
          │       1 Unit        │
          └──────────┬──────────┘
                     │
                     ▼
             🌡️ tavg Prediction
```

## Architecture Summary

| Layer      | Configuration   |
| ---------- | --------------- |
| Input      | `(30, 14)`      |
| LSTM 1     | 100 units       |
| Dropout    | 20%             |
| LSTM 2     | 50 units        |
| Dropout    | 20%             |
| Dense      | 25 units + ReLU |
| Output     | 1 unit          |
| Parameters | **77,501**      |

---

# ⚙️ Model Configuration

The model is compiled with:

```python
Adam(learning_rate=0.001)
```

### Loss

```text
Mean Squared Error
```

### Metric

```text
Mean Absolute Error
```

---

# 🚀 Training Configuration

| Parameter               |    Value |
| ----------------------- | -------: |
| Epochs                  |      100 |
| Batch Size              |       32 |
| Learning Rate           |    0.001 |
| Optimizer               |     Adam |
| Loss                    |      MSE |
| Metric                  |      MAE |
| Time Step               |       30 |
| LSTM Units              | 100 → 50 |
| Dropout                 |     0.20 |
| Dense Units             |       25 |
| Early Stopping Patience |       15 |
| LR Reduction Patience   |        5 |
| LR Reduction Factor     |      0.5 |
| Minimum LR              |  0.00001 |

---

# 🛑 Training Control

Two Keras callbacks are used.

## Early Stopping

```python
EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)
```

This prevents unnecessary training once validation performance stops improving.

---

## Learning Rate Reduction

```python
ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=0.00001
)
```

When validation loss reaches a plateau, the learning rate is reduced.

Example:

```text
0.001
  ↓
0.0005
  ↓
0.00025
  ↓
0.000125
```

---

# 📊 Training Results

The model completed training successfully and stopped after:

```text
32 epochs
```

### Training Progress

| Epoch | Train Loss | Val Loss | Train MAE | Val MAE |
| ----: | ---------: | -------: | --------: | ------: |
|     1 |     0.2737 |   0.2279 |    0.4998 |  0.4527 |
|     5 |     0.0436 |   0.0396 |    0.1851 |  0.1824 |
|    10 |     0.0230 |   0.0235 |    0.1296 |  0.1304 |
|    15 |     0.0229 |   0.0232 |    0.1270 |  0.1276 |
|    20 |     0.0229 |   0.0233 |    0.1269 |  0.1278 |
|    25 |     0.0229 |   0.0233 |    0.1270 |  0.1279 |
|    32 |     0.0229 |   0.0232 |    0.1268 |  0.1276 |

The training process shows a strong reduction in loss during the initial epochs followed by stabilization.

---

# 📈 Model Evaluation

The final model was tested on the unseen test sequences.

### Scaled Test Performance

```text
Test Loss : 0.0266
Test MAE  : 0.1375
```

### Temperature-Scale Metrics

After inverse transformation:

| Metric |      Result |
| ------ | ----------: |
| MAE    | **3.42 °C** |
| MSE    |   **16.52** |
| RMSE   | **4.06 °C** |
| R²     | **-0.0003** |

These results represent the current **baseline model** and provide a starting point for future optimization.

---

# 📉 Performance Interpretation

The current model completes the complete forecasting pipeline successfully, but the predictive performance leaves significant room for improvement.

The current baseline is:

```text
MAE  ≈ 3.42 °C
RMSE ≈ 4.06 °C
R²   ≈ -0.0003
```

The negative R² indicates that the current model does not yet explain the test-set variance effectively.

This does **not** mean the pipeline itself failed. The preprocessing, sequence generation, model training, prediction, and visualization stages all execute successfully.

The next development phase is focused on improving the forecasting quality.

---

# 📈 Visualizations

The evaluation module automatically generates three visualizations.

## 🌡️ Actual vs Predicted Temperature

```text
outputs/LSTM_Prediction.png
```

Shows the relationship between:

```text
Actual Temperature
        vs.
Predicted Temperature
```

This is useful for visually inspecting how closely the model follows temperature trends.

---

## 📉 Training vs Validation Loss

```text
outputs/LSTM_Loss.png
```

Displays:

```text
Training Loss
Validation Loss
```

Useful for observing:

* Model convergence
* Training stability
* Overfitting
* Underfitting

---

## 📊 Training vs Validation MAE

```text
outputs/LSTM_MAE.png
```

Displays:

```text
Training MAE
Validation MAE
```

This provides another view of model learning throughout training.

---

# 📁 Project Structure

```text
LSTM/
│
├── 📂 Data/
│   └── Data_Preprocessing.py
│
├── 📂 LSTM/
│   ├── LSTM_Model.py
│   ├── LSTM_Training.py
│   └── LSTM_Evaluation.py
│
├── 📂 Main/
│   └── Main_LSTM.py
│
├── 📂 outputs/
│   ├── LSTM_Model.keras
│   ├── LSTM_Prediction.png
│   ├── LSTM_Loss.png
│   └── LSTM_MAE.png
│
├── 📄 requirements.txt
└── 📄 README.md
```

---

# 🧩 Project Components

## `Data_Preprocessing.py`

Responsible for the complete data pipeline:

```text
Load Dataset
     ↓
Filter Karachi
     ↓
Clean Dates
     ↓
Handle Missing Values
     ↓
Feature Engineering
     ↓
Outlier Handling
     ↓
Data Splitting
     ↓
Scaling
     ↓
Sequence Generation
```

---

## `LSTM_Model.py`

Responsible for:

* Creating the LSTM architecture
* Adding dropout
* Adding dense layers
* Configuring Adam optimizer
* Configuring MSE loss
* Adding MAE metric

---

## `LSTM_Training.py`

Responsible for:

* Loading processed data
* Building the model
* Training the model
* Early stopping
* Learning-rate scheduling
* Saving the model

---

## `LSTM_Evaluation.py`

Responsible for:

* Test-set evaluation
* Prediction generation
* Inverse scaling
* Regression metrics
* Actual vs predicted graph
* Loss graph
* MAE graph

---

## `Main_LSTM.py`

The main controller connects training and evaluation:

```python
trainer = LSTMTraining()

model, history, scaler_y, x_test, y_test = trainer.run()

evaluator = LSTMEvaluation(
    model=model,
    history=history,
    scaler_y=scaler_y,
    x_test=x_test,
    y_test=y_test
)

evaluator.run()
```

This keeps the project modular and separates:

```text
Preprocessing
     ↓
Training
     ↓
Evaluation
```

---

# 💾 Model Output

After successful training, the model is saved in:

```text
outputs/LSTM_Model.keras
```

The `.keras` format allows the trained model to be loaded later for:

* Prediction
* Further training
* Testing
* Deployment

---

# 📦 Requirements

Install the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

Or:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```text
numpy
pandas
scikit-learn
matplotlib
tensorflow
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

## 2. Navigate to the Project

```bash
cd src/LSTM
```

## 3. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
```

## 4. Activate the Environment

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

From the LSTM directory:

```bash
python Main_LSTM.py
```

The program will automatically:

```text
Load Dataset
     ↓
Preprocess Data
     ↓
Create Features
     ↓
Create Sequences
     ↓
Build LSTM
     ↓
Train
     ↓
Save Model
     ↓
Predict
     ↓
Generate Metrics
     ↓
Generate Graphs
```

---

# 📋 Expected Output

A successful execution will produce output similar to:

```text
KARACHI TEMPERATURE PREDICTION USING LSTM

Dataset Shape: (31779, 27)
Karachi Records: 5844

Rows after date cleaning: 5844

Feature Engineering Completed
Rows after features: 5837

Train: (4085, 32)
Validation: (875, 32)
Test: (877, 32)

Scaling Completed

X Train: (4085, 14)
Y Train: (4085, 1)

Training X Shape: (4055, 30, 14)
Training Y Shape: (4055, 1)

Validation X Shape: (845, 30, 14)
Validation Y Shape: (845, 1)

Testing X Shape: (847, 30, 14)
Testing Y Shape: (847, 1)

Number of Features: 14
Time Step: 30

Data Loaded Successfully

Model Built Successfully

Epoch 1/100
...

Training Completed Successfully

Model Saved Successfully

outputs/LSTM_Model.keras

Evaluating Model...

Test Loss : 0.0266
Test MAE  : 0.1375

Prediction Completed Successfully.

LSTM MODEL EVALUATION

MAE      : 3.42
MSE      : 16.52
RMSE     : 4.06
R² Score : -0.0003

Evaluation Completed Successfully.

PROJECT COMPLETED SUCCESSFULLY
```

---

# 🔮 Future Improvements

The current implementation provides a solid baseline. Several improvements can be explored.

## 🌦️ 1. Improve Feature Quality

The `visibility` feature contains missing values for Karachi.

Future versions can investigate whether it should be:

* Removed
* Replaced
* Reconstructed from another source

---

## 🧠 2. Hyperparameter Tuning

Experiment with:

```text
LSTM Units
Dropout
Learning Rate
Batch Size
Dense Units
Number of Layers
```

---

## ⏳ 3. Experiment with Time Windows

Compare:

```text
7 Days
14 Days
30 Days
60 Days
90 Days
```

to determine the most useful historical window.

---

## 🔄 4. Compare LSTM with GRU

A GRU architecture can be implemented and compared against the LSTM baseline.

```text
LSTM
  │
  ├──── Compare ────► GRU
  │
  ▼
Performance Analysis
```

---

## 🌳 5. Compare Against Traditional ML

Useful baseline models include:

```text
Linear Regression
Random Forest
XGBoost
```

This can determine whether deep learning provides a meaningful advantage.

---

## 📊 6. Stronger Time-Series Validation

Future versions can implement:

```text
Walk-Forward Validation
Rolling-Window Validation
Expanding-Window Validation
```

These approaches can provide a more robust estimate of forecasting performance.

---

## 🔬 7. Hyperparameter Optimization

Potential tools:

```text
KerasTuner
Optuna
Random Search
Bayesian Optimization
```

---

## 🌤️ 8. Additional Weather Variables

Potential additional features:

```text
Wind Direction
Solar Radiation
UV Index
Dew Point Trends
Season
Day of Year
Atmospheric Conditions
```

---

## 🎯 Long-Term Goal

The main optimization objective is:

```text
Lower MAE
      ↓
Lower RMSE
      ↓
Higher R²
      ↓
Better Temperature Forecasts
```

while maintaining a valid time-series methodology.

---

# 📌 Current Project Status

| Component                | Status |
| ------------------------ | :----: |
| Dataset Loading          |    ✅   |
| Karachi Filtering        |    ✅   |
| Date Processing          |    ✅   |
| Missing Value Handling   |    ✅   |
| Feature Engineering      |    ✅   |
| Seasonal Features        |    ✅   |
| Lag Features             |    ✅   |
| Rolling Features         |    ✅   |
| Outlier Handling         |    ✅   |
| Chronological Split      |    ✅   |
| Feature Scaling          |    ✅   |
| Sequence Generation      |    ✅   |
| LSTM Architecture        |    ✅   |
| Model Training           |    ✅   |
| Early Stopping           |    ✅   |
| Learning Rate Scheduling |    ✅   |
| Model Saving             |    ✅   |
| Prediction               |    ✅   |
| Regression Metrics       |    ✅   |
| Visualization            |    ✅   |
| Performance Optimization |   🔄   |

---

# 📚 Learning Outcomes

This project provides practical experience with:

### 🐍 Python

* NumPy
* Pandas
* Object-oriented project structure

### 📊 Data Science

* Data cleaning
* Missing-value handling
* Outlier handling
* Feature engineering
* Normalization

### 🧠 Deep Learning

* Neural networks
* LSTM
* Dropout
* Dense layers
* Adam optimizer
* MSE loss
* MAE metric

### ⏳ Time-Series

* Chronological splitting
* Lag features
* Rolling averages
* Seasonal encoding
* Sliding windows
* Sequential prediction

### ⚙️ Model Training

* Validation data
* Early stopping
* Learning-rate scheduling
* Model checkpointing/saving

### 📈 Evaluation

* MAE
* MSE
* RMSE
* R²
* Actual vs predicted visualization
* Training/validation curves

---

# 🏁 Conclusion

This project implements a complete **LSTM-based time-series forecasting pipeline for Karachi temperature prediction**.

Starting from a multi-city weather dataset, the system:

```text
📂 Loads historical data
        ↓
📍 Selects Karachi
        ↓
🧹 Cleans the data
        ↓
🔧 Engineers temporal features
        ↓
📊 Handles outliers
        ↓
📚 Creates chronological splits
        ↓
📏 Scales the data
        ↓
⏳ Builds 30-day sequences
        ↓
🧠 Trains an LSTM
        ↓
💾 Saves the model
        ↓
🌡️ Generates predictions
        ↓
📈 Creates visualizations
```

The current implementation establishes a **working baseline forecasting system** with:

```text
MAE  : 3.42 °C
RMSE : 4.06 °C
R²   : -0.0003
```

These results provide a starting point for future experimentation with feature engineering, architecture design, hyperparameter tuning, alternative sequence models, and stronger time-series validation.

> 🚀 **Baseline completed — optimization is the next stage.**

---

# 👨‍💻 Author

## Muzammil Ahmed

**AI / Machine Learning Student**
**NED University of Engineering & Technology**

### Areas of Interest

```text
Artificial Intelligence
Machine Learning
Deep Learning
Data Science
Time-Series Forecasting
```

---

# ⭐ Project Status

```text
┌─────────────────────────────────────┐
│       KARACHI TEMPERATURE AI        │
├─────────────────────────────────────┤
│                                     │
│  Dataset          ✅                 │
│  Preprocessing    ✅                 │
│  Feature Eng.     ✅                 │
│  LSTM Model       ✅                 │
│  Training         ✅                 │
│  Evaluation       ✅                 │
│  Visualization    ✅                 │
│                                     │
│  Optimization     🔄 In Progress     │
│                                     │
└─────────────────────────────────────┘
```

---

## 📜 License

This project is developed for **educational, academic, and internship purposes**.

---
