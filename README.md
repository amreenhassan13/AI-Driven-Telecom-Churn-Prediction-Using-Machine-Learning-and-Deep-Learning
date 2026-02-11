# AI-Driven-Telecom-Churn-Prediction-Using-Machine-Learning-and-Deep-Learning
End-to-End AI Framework for Telecom Customer Churn Prediction Using Machine Learning, Deep Learning, and Intelligent Customer Segmentation
# AI-Driven Telecom Churn Prediction with ML & DL

An end-to-end Artificial Intelligence framework for predicting customer churn in the telecom industry using Machine Learning, Deep Learning, and Customer Segmentation techniques.

---

## 📌 Project Overview

This project was developed for **CSE422 – Artificial Intelligence (BRAC University)**.

Customer churn refers to customers discontinuing telecom services, which directly impacts company revenue. The objective of this project is to:

- Identify key drivers of churn
- Compare supervised ML models
- Implement a Deep Learning model (MLP)
- Handle class imbalance
- Apply unsupervised learning for segmentation
- Evaluate models using advanced performance metrics

📄 Full Report: See `Final_CSE422_Telco_Churn_Report.pdf`

---

## 📊 Dataset Description

- **Dataset:** Telco Customer Churn
- **Total Records:** 7,043 customers
- **Total Features:** 21 (20 input + 1 target)
- **Problem Type:** Binary Classification
- **Target Variable:** Churn (0 = No, 1 = Yes)
- **Class Distribution:** 5,174 Non-Churn | 1,869 Churn

The dataset includes:
- Demographics
- Subscription details
- Internet services
- Payment methods
- Billing information

---

## 🔍 Correlation Analysis

A correlation heatmap was generated after one-hot encoding categorical features.

Key Insights:
- Strong positive correlation with churn:
  - InternetService_Fiber optic
  - PaymentMethod_Electronic check
  - MonthlyCharges
  - PaperlessBilling
- Strong negative correlation with churn:
  - tenure
  - Contract_Two year

![Correlation Heatmap](images/heatmap.png)

---

## 📦 Data Preprocessing

The following preprocessing steps were applied:

- Missing value handling (TotalCharges imputed)
- One-hot encoding for categorical variables
- Binary encoding of target variable
- StandardScaler normalization
- Stratified 80/20 train-test split
- Oversampling for imbalance handling

A pipelined approach was used to prevent data leakage and ensure fair evaluation.

---

## 📊 Exploratory Data Analysis (EDA)

### Tenure vs Churn

Customers who churned had significantly lower tenure.

![Tenure Boxplot](images/boxplot.png)

---

## 🤖 Supervised Machine Learning Models

Three models were implemented:

### 1️⃣ Logistic Regression
- Accuracy: ~80.6%
- ROC-AUC: ~0.84
- Best overall balanced performance

### 2️⃣ Decision Tree
- Accuracy: ~79%
- Slightly lower churn recall

### 3️⃣ Deep Learning (MLP)
- Accuracy: ~78.9%
- ROC-AUC: ~0.84
- Tuned with early stopping (49 iterations)

---

## 📊 Model Accuracy Comparison

Logistic Regression achieved the highest accuracy.

![Model Accuracy Comparison](images/barchart.png)

---

## 📈 Confusion Matrix Analysis

Confusion matrices were generated for all supervised models.

Key Observation:
- Logistic Regression correctly predicted more churn cases.
- MLP missed the highest number of churn customers.
- Decision Tree performed moderately.

![Confusion Matrices](images/matrix.png)

---

## ⚖️ Class Imbalance Handling

The dataset is imbalanced.

Oversampling was applied to improve churn recall:

- Recall improved to ~67%
- Accuracy decreased to ~75%
- Trade-off is expected in imbalanced classification

In churn prediction, recall is often more valuable than raw accuracy.

---

## 🧠 Unsupervised Learning – K-Means Clustering

K-Means clustering was applied to identify natural customer segments.

PCA was used for 2D visualization.

Results:
- Cluster 0 → ~7% churn (low-risk customers)
- Cluster 1 → ~32% churn (high-risk segment)

![K-Means Clustering (PCA 2D)](images/kmeans.png)

Unsupervised learning provided business insight but was less effective for direct churn prediction compared to supervised models.

---

## 📈 ROC Curve Comparison

All models performed significantly better than random guessing.

- Logistic Regression: AUC ≈ 0.84
- MLP: AUC ≈ 0.84
- Decision Tree: AUC ≈ 0.83

Logistic Regression maintained a small but consistent advantage.

---

## 🏆 Final Model Comparison

| Model | Accuracy | ROC-AUC | Churn Recall |
|--------|----------|----------|--------------|
| Logistic Regression | 80.6% | 0.84 | 55.9% |
| Decision Tree | 79.1% | 0.84 | 50.5% |
| Neural Network (MLP) | 78.9% | 0.84 | 48.4% |

### 🥇 Best Performing Model:
**Logistic Regression**

---

## 📦 Requirements

```txt
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
joblib>=1.2.0
jupyter>=1.0.0
notebook>=6.5.0

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
jupyter notebook
