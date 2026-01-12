# Experiment 1 – Working with Python Packages for Machine Learning

## 📌 Overview
This repository contains the Jupyter Notebook **a1.ipynb**, which implements **Experiment 1** of the Machine Learning laboratory.  
The experiment focuses on exploring Python packages used for data analysis, preprocessing, visualization, and machine learning, and applying a complete ML workflow on multiple datasets.

The notebook follows the standard steps of:
- Data loading
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature selection
- Model training
- Performance evaluation

---

## 🧪 Datasets Used

### 1️⃣ Loan Approval Prediction Dataset
- **Type:** Supervised Learning – Binary Classification  
- **Objective:** Predict whether a loan will be approved or rejected.
- **Techniques Used:**
  - EDA (histograms, box plots, heatmaps)
  - Label Encoding
  - Feature Scaling
  - ANOVA feature selection
  - Logistic Regression

---

### 2️⃣ Iris Dataset
- **Type:** Supervised Learning – Multiclass Classification  
- **Objective:** Classify iris flowers into Setosa, Versicolor, and Virginica.
- **Techniques Used:**
  - Pair plots and correlation analysis
  - Label Encoding
  - Feature Scaling
  - Logistic Regression
- **Observation:** Achieved very high accuracy due to clean, linearly separable data.

---

### 3️⃣ Diabetes Prediction Dataset
- **Type:** Supervised Learning – Binary Classification  
- **Objective:** Predict diabetes based on medical attributes.
- **Techniques Used:**
  - Handling categorical and numerical features
  - Feature scaling
  - ANOVA feature selection
  - Logistic Regression with class balancing
- **Key Insight:** High recall is prioritized over precision for medical screening.

---

### 4️⃣ Email Spam Classification Dataset
- **Type:** Supervised Learning – Binary Classification  
- **Objective:** Classify emails as spam or non-spam using word-frequency features.
- **Techniques Used:**
  - High-dimensional numeric feature handling
  - Feature selection (ANOVA)
  - Logistic Regression
  - Class imbalance handling
- **Observation:** High accuracy due to class imbalance; minority-class metrics analyzed carefully.

---

### 5️⃣ English Handwritten Characters Dataset
- **Type:** Supervised Learning – Multiclass Classification (62 classes)
- **Objective:** Recognize handwritten English characters.
- **Data Format:**
  - `english.csv` → image paths and labels
  - `Img/` → image files
- **Techniques Used:**
  - Image preprocessing (grayscale, resizing, normalization)
  - Label Encoding
  - Train–test split
  - CNN / classical ML baseline (depending on environment)
- **Observation:** Moderate accuracy due to many visually similar classes and limited samples.

---

## 🔄 Machine Learning Workflow Followed

1. **Loading the dataset**
2. **Exploratory Data Analysis (EDA)**
3. **Data preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling
4. **Feature selection**
   - SelectKBest (ANOVA / Chi-square where appropriate)
5. **Train–test split**
6. **Model training**
7. **Performance evaluation**
   - Accuracy
   - Confusion matrix
   - Precision, Recall, F1-score

---

## 🧰 Libraries Used
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras (for image-based models)
- OpenCV

Refer to requirements.txt for detailed versions

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 📝 Conclusion
This experiment demonstrates the practical application of Python-based machine learning tools across multiple problem types including tabular data classification, medical prediction, text classification, and image-based recognition.  
The results highlight the importance of appropriate preprocessing, feature selection, and metric interpretation based on the dataset characteristics.


## 👩‍💻 Author
**R Padmashri**  
**093**  
Machine Learning Laboratory – Semester 6
