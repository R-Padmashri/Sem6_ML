# Experiment 2 – Binary Classification using Naïve Bayes and K-Nearest Neighbors

## 📌 Overview
This repository contains the Jupyter Notebook **a2.ipynb**, which implements **Experiment 2** of the Machine Learning laboratory.  
The experiment focuses on implementing and comparing **probabilistic** and **instance-based** classification algorithms for a binary classification problem.

The notebook demonstrates a complete machine learning pipeline including:
- Dataset loading
- Data preprocessing
- Feature scaling
- Model training
- Hyperparameter tuning
- Performance evaluation and visualization

---

## 🧪 Dataset Used

### 📧 Email Spam Classification (Spambase Dataset)
- **Type:** Supervised Learning – Binary Classification  
- **Objective:** Classify emails as **spam** or **non-spam** based on word-frequency and character-based features.
- **Target Classes:**
  - `0` → Non-Spam  
  - `1` → Spam  

- **Dataset Source:** Kaggle (Spambase Dataset)
- **Characteristics:**
  - High-dimensional numerical feature space
  - No missing values
  - Suitable for both probabilistic and distance-based classifiers

---

## 🤖 Algorithms Implemented

### 1️⃣ Gaussian Naïve Bayes
- Probabilistic classifier based on Bayes’ theorem
- Assumes conditional independence between features
- Computationally efficient
- Works well for high-dimensional data

---

### 2️⃣ K-Nearest Neighbors (KNN)
- Instance-based, distance-driven classifier
- Performance depends on the choice of **k**
- Sensitive to feature scale
- Hyperparameter tuning performed using **GridSearchCV**

---

## 🔄 Machine Learning Workflow Followed

1. **Loading the dataset**
2. **Train–test split**
3. **Feature scaling using StandardScaler**
4. **Training Gaussian Naïve Bayes classifier**
5. **Training baseline KNN classifier**
6. **Accuracy vs k analysis**
7. **Hyperparameter tuning using GridSearchCV**
8. **Performance evaluation**
   - Accuracy
   - Confusion matrix
   - ROC curve and AUC

---

## 🧰 Libraries Used
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Refer to **requirements.txt** for exact library versions.

---

## 📊 Evaluation Metrics
- Accuracy
- Confusion Matrix
- ROC Curve
- AUC Score

---

## 📝 Observations
- Naïve Bayes provides fast and reliable baseline performance but may suffer from higher bias.
- KNN performance varies significantly with different values of **k**.
- Smaller values of **k** tend to overfit, while larger values may underfit.
- Hyperparameter tuning improves generalization and model stability.
- Feature scaling is critical for distance-based algorithms like KNN.

---

## 📝 Conclusion
This experiment highlights the differences between probabilistic and instance-based learning approaches for binary classification.  
Naïve Bayes offers simplicity and speed, while KNN can achieve better accuracy when properly tuned.  
The experiment reinforces the importance of preprocessing, hyperparameter selection, and appropriate evaluation metrics in machine learning workflows.

---

## 👩‍💻 Author
**R Padmashri**  
**093**  
Machine Learning Laboratory – Semester 6
