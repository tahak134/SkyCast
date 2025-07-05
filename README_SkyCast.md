
# 🌤️ SkyCast – Weather Summary Classification using Machine Learning

A machine learning project that predicts **weather summaries** (Clear, Cloudy, Rainy) using meteorological data. This project compares the performance of **Logistic Regression**, **Random Forest**, and **Neural Networks** on a structured weather dataset from Kaggle.

## 📊 Project Overview

**SkyCast** aims to classify complex weather descriptions into 3 simplified categories for practical prediction. The project showcases a complete ML workflow including:
- Data preprocessing & normalization
- Handling class imbalance with SMOTE
- Multi-model training & tuning
- Model evaluation using accuracy, ROC-AUC, and balanced accuracy
- A minimal user interface for real-time prediction

## 📁 Dataset

- **Source**: [Weather Dataset – Kaggle](https://www.kaggle.com/datasets/muthuj7/weather-dataset)
- **Type**: Structured tabular data
- **Features**: Temperature, Humidity, Wind Speed, Pressure, Visibility
- **Target**: `Summary` – Multiclass (Clear = 0, Cloudy = 1, Rainy = 2)

## 🔧 Preprocessing Steps

- Removed missing values using `dropna()`
- Standardized features with `StandardScaler`
- Label encoded the target variable (`Summary`)
- Handled class imbalance using **SMOTE** (applied only to Neural Network models)
- Train-test split: 80/20 ratio using `train_test_split()`

## 🧠 Models Implemented

### 🔹 Logistic Regression (Parametric)
- Three configurations tested using different penalties (L2, L1, ElasticNet)
- `class_weight` balancing and solver tuning

### 🔹 Random Forest (Non-Parametric)
- Three tuned models with:
  - Different tree depths
  - Custom `class_weight`
  - `oob_score` evaluation
- Performed better on imbalanced data but showed mild overfitting

### 🔹 Neural Network (Keras)
- Three architectures tested:
  - NN1: Single hidden layer
  - NN2: Two hidden layers
  - NN3: Deep network with 3 layers
- Used **SMOTE** to improve minority class (Rainy) prediction
- Tuned for dropout, batch size, learning rate, and class weights

## 🧪 Evaluation Metrics

- Accuracy & Balanced Accuracy
- ROC-AUC Curve (multi-class)
- Overfitting/Underfitting Analysis
- Comparative plots and tables (80-20 vs 70-30 splits)

## 🖥️ User Interface

A simple interactive UI was built (CLI or web-based) allowing users to enter real-time weather parameters and get predictions on the spot.

## 🧩 Key Takeaways

- **Logistic Regression**: Fast and interpretable but limited in non-linear learning
- **Random Forest**: Robust and better on imbalanced data, slight overfitting
- **Neural Network**: Best balance with SMOTE + proper tuning, NN2 architecture was optimal

## 🚀 Possible Improvements

- Add L2 regularization & dropout for overfitting control
- Feature engineering: create derived features (e.g., humidity/temperature ratio)
- Hyperparameter optimization via `RandomizedSearchCV`
- Cross-validation for better generalization
- Try advanced models like **LSTM**, **XGBoost**, or **CatBoost**

## 🛠️ Tech Stack

- **Languages**: Python  
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn, imbalanced-learn, Keras, SMOTE  
- **Tools**: Jupyter Notebook, Google Colab, Streamlit (optional), CLI

## 📂 Repository Structure

```
skycast-weather-classifier/
├── data/                   # raw and processed dataset
├── notebooks/              # EDA and model building notebooks
├── models/                 # saved model weights
├── src/                    # scripts for preprocessing, training
├── app/                    # UI for predictions (CLI/Streamlit)
├── requirements.txt
├── README.md
└── skycast_report.pdf      # Full project report (optional)
```

## 👨‍💻 Authors

- **Taha Khan** – CS-22134  
- **Hussain Kazmi** – CS-22090  

## 📜 License

This project is for academic and educational purposes. Contact the authors for reuse or collaboration.
