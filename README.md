# 🩺 Diabetes Health Indicators Analysis and Prediction

An end-to-end **Data Analytics and Machine Learning project** that analyzes health indicators associated with diabetes, compares classification models, and predicts diabetes outcomes through an interactive Streamlit dashboard.

## 📌 Project Overview

Diabetes is a major public health concern. Analyzing health indicators and lifestyle factors can help reveal patterns associated with diabetes and support data-driven health analysis.

This project uses the **2015 Behavioral Risk Factor Surveillance System (BRFSS) health indicators dataset** to explore relationships between health conditions, lifestyle factors, and diabetes status.

The project combines exploratory data analysis (EDA), data visualization, machine learning classification, model evaluation, and an interactive Streamlit dashboard to transform health-related data into meaningful analytical insights.

## 🎯 Project Objectives

- Analyze health indicators associated with diabetes.
- Perform exploratory data analysis and statistical exploration.
- Visualize distributions, patterns, and relationships among health indicators.
- Examine correlations between health-related variables.
- Build machine learning models for diabetes classification.
- Compare Logistic Regression and Random Forest performance.
- Evaluate models using multiple classification metrics.
- Develop an interactive dashboard using Streamlit.
- Generate predictions using trained machine learning models.

## 📊 Dataset

**Dataset name:** `diabetes_binary_health_indicators_BRFSS2015.csv`

The dataset is based on the 2015 Behavioral Risk Factor Surveillance System (BRFSS) survey and contains health indicators used to analyze diabetes status.

The features include indicators related to:

- Diabetes status
- High blood pressure
- High cholesterol
- Body Mass Index (BMI)
- Smoking
- Physical activity
- Fruit and vegetable consumption
- General health
- Physical health
- Mental health
- Age
- Education
- Income
- Other health and lifestyle factors

The target variable represents diabetes status.

## 🛠️ Technologies Used

- **Python** — Programming and data analysis
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization
- **Plotly** — Interactive visualizations
- **Scikit-learn** — Machine learning and evaluation
- **Streamlit** — Interactive web dashboard
- **Jupyter Notebook** — Data analysis and model development

## 🔄 Project Workflow

### 1. Data Loading and Preparation

- Load the BRFSS 2015 health indicators dataset.
- Examine dataset structure, columns, and data types.
- Prepare input features and the target variable.
- Apply appropriate preprocessing for model training.

### 2. Exploratory Data Analysis (EDA)

- Analyze the distribution of diabetes outcomes.
- Explore health indicator distributions.
- Examine diabetes rates across age groups.
- Investigate general health patterns.
- Analyze relationships between health and lifestyle factors.
- Visualize correlations using a heatmap.
- Explore physical and mental health outliers.

### 3. Data Preprocessing

- Prepare features for machine learning.
- Apply `StandardScaler` where required.
- Maintain consistency between model training and prediction preprocessing.

### 4. Machine Learning Models

**Logistic Regression**

- Provides a baseline classification approach.
- Estimates diabetes outcomes based on input features.
- Provides a useful comparison with ensemble-based methods.

**Random Forest Classifier**

- Combines multiple decision trees for classification.
- Learns patterns from health indicator features.
- Provides an alternative model for performance comparison.

### 5. Model Evaluation

The project compares the trained models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Using multiple metrics provides a more comprehensive understanding of model performance, particularly when positive cases may be harder to identify.

### 6. Interactive Streamlit Dashboard

The dashboard brings the analysis and model results together in an interactive application.

Key features include:

- Dataset overview and summary statistics
- Diabetes distribution analysis
- Diabetes rate analysis across age groups
- Health risk factor analysis
- Indicator correlation heatmap
- Physical and mental health outlier analysis
- Logistic Regression and Random Forest comparison
- Diabetes prediction interface
- Interactive visualizations
- Data filtering and CSV export, where supported by the application

## 🖥️ Dashboard Screenshots

The following screenshots showcase the major sections of the Streamlit application.

### 1. Dashboard Overview
<img width="1896" height="872" alt="image" src="https://github.com/user-attachments/assets/2f92a756-ca95-46c8-a1b4-0b693b8f8cf2" />

### 2. Data Analysis and Visualization
<img width="1537" height="782" alt="data_analysis and visualization" src="https://github.com/user-attachments/assets/ca3eb698-2b87-40fe-af49-838dff038edf" />

### 3. Health Risk Factor Analysis
<img width="1407" height="755" alt="image" src="https://github.com/user-attachments/assets/855ced27-3725-4919-b6dc-2cc6e2a5d4a8" />

### 4. Indicator Correlations
<img width="1527" height="532" alt="Correlation_heatmap" src="https://github.com/user-attachments/assets/87313eba-3bd0-4545-bc49-c8af5e735065" />

### 5. Health Outlier Analysis
<img width="1387" height="738" alt="image" src="https://github.com/user-attachments/assets/e4ab4229-f624-438a-ba5d-ded829ac2e8d" />

### 6. Model Performance Comparison
<img width="1412" height="805" alt="Model_performance" src="https://github.com/user-attachments/assets/0952a417-6af9-4df2-a568-fdc1b5fe75b8" />

### 7. Diabetes Prediction
<img width="1632" height="848" alt="image" src="https://github.com/user-attachments/assets/5824b4d8-d54e-4d41-a917-8753b3b3156f" />

## 📈 Model Performance

The saved model comparison results are presented below.

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 86.21% | 86.53% |
| Precision | 51.71% | 58.94% |
| Recall | 15.83% | 10.96% |
| F1-score | 24.24% | 18.49% |
| ROC-AUC | 81.92% | 82.40% |

### Performance Analysis

- **Accuracy:** Random Forest achieved slightly higher accuracy.
- **Precision:** Random Forest achieved higher precision.
- **Recall:** Logistic Regression achieved higher recall.
- **F1-score:** Logistic Regression achieved a higher F1-score.
- **ROC-AUC:** Random Forest achieved a slightly higher ROC-AUC.

The results demonstrate the importance of considering several evaluation metrics rather than relying on accuracy alone.

In particular, the relatively low recall values indicate that both models missed a substantial proportion of positive cases in the evaluated data. Further investigation and model improvement may be needed before either model could be considered useful for real-world screening.

*Note: These values are based on the saved model comparison results. Performance may vary with different data splits, preprocessing, or training configurations.*

## 📂 Project Structure

```text
diabetes-health-indicators-analysis/
│
├── screenshots/
│   ├── dashboard_overview.png
│   ├── data_analysis.png
│   ├── risk_factor_analysis.png
│   ├── indicator_correlations.png
│   ├── health_outlier_analysis.png
│   ├── model_performance.png
│   └── diabetes_prediction.png
│
├── app.py
├── diabetes-health-indicators.ipynb
├── diabetes_binary_health_indicators_BRFSS2015.csv
├── diabetes_model_comparison.csv
├── diabetes_feature_scaler.pkl
├── logistic_regression_model.pkl
├── random_forest_model.pkl
├── requirements.txt
└── README.md
```

### File Descriptions

| File or Folder | Description |
|---|---|
| `app.py` | Streamlit dashboard application |
| `diabetes-health-indicators.ipynb` | Notebook containing data analysis, visualizations, and model development |
| `diabetes_binary_health_indicators_BRFSS2015.csv` | Dataset used in the project |
| `diabetes_model_comparison.csv` | Saved model evaluation results |
| `diabetes_feature_scaler.pkl` | Saved feature scaler |
| `logistic_regression_model.pkl` | Trained Logistic Regression model |
| `random_forest_model.pkl` | Trained Random Forest model |
| `requirements.txt` | Python dependencies |
| `screenshots/` | Dashboard screenshots displayed in this README |
| `README.md` | Project documentation |

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system
- pip package manager
- Git (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/diabetes-health-indicators-analysis.git
```

Navigate to the project directory:

```bash
cd diabetes-health-indicators-analysis
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Dashboard

```bash
streamlit run app.py
```

Open the local URL shown in your terminal to access the application in your browser.

## 💡 Key Learnings

- Data cleaning and preprocessing
- Exploratory data analysis
- Health indicator analysis
- Data visualization and correlation analysis
- Feature scaling
- Machine learning classification
- Model evaluation and comparison
- Interactive dashboard development
- Communicating analytical findings through visualizations

## 🚀 Future Enhancements

- Perform hyperparameter tuning and cross-validation.
- Improve recall for positive diabetes cases.
- Investigate appropriate class imbalance handling techniques.
- Explore additional classification algorithms.
- Add feature importance and model interpretability visualizations.
- Enhance dashboard usability and visual design.
- Deploy the application to a cloud hosting platform.

## 👩‍💻 Author

**Lakshmidhar N**

- **Project Type:** Major Project
- **Domain:** Data Analytics
- **Project Title:** Diabetes Health Indicators Analysis and Prediction

## 📌 Conclusion

The Diabetes Health Indicators Analysis and Prediction project demonstrates the application of data analytics, machine learning, and data visualization to health-related data. By combining exploratory data analysis with Logistic Regression and Random Forest classification models, the project provides a practical way to examine health indicators, explore patterns associated with diabetes, compare model performance, and generate predictions through an interactive Streamlit dashboard.

The project highlights the importance of transforming raw data into meaningful insights and evaluating machine learning models using multiple performance metrics. It also demonstrates how interactive dashboards can make analytical results easier to explore and understand.

---

*This project was developed for educational purposes to demonstrate the application of data analytics, machine learning, and interactive visualization in health data analysis. Predictions are not medical diagnoses and should not replace professional medical advice.*
