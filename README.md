# 🎓 PlacementIQ — Campus Placement Prediction & Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**PlacementIQ** is an end-to-end Machine Learning web application and data analytics platform designed to predict campus placement outcomes, estimate salary packages, segment student profiles, and detect profile anomalies using a comprehensive dataset of 50,000 student records.

---

## 📌 Features

### 🎯 Supervised Learning & Prediction
- **Interactive Candidate Risk Engine**: Input individual student details (CGPA, test scores, stream, backlogs, projects, etc.) to get real-time placement probability and key risk indicators powered by **Gradient Boosting**.
- **Salary Package Estimation**: Regression analysis using **Linear Regression** with $R^2$, RMSE, and MAE evaluation metrics and Actual vs. Predicted visualizations.
- **Classification Benchmark Suite**: Full benchmarking across multiple algorithms:
  - Logistic Regression (with & without feature scaling)
  - Decision Trees
  - Gradient Boosting
  - AdaBoost
  - Bagging
  - Random Forest
- **Comprehensive Evaluation**: Interactive Confusion Matrices, ROC curves with AUC scores, feature importance rankings, and classification reports.
- **Cross-Validation**: 5-Fold Stratified Cross-Validation benchmarking model stability.

### 🔍 Unsupervised Learning & Clustering
- **K-Means Student Segmentation**: Segment students into distinct performance clusters with centroid analysis and placement rate breakdowns.
- **Elbow Method**: Automated within-cluster sum of squares (WCSS) calculation to identify the optimal $K$.
- **DBSCAN**: Density-based clustering to discover natural density groupings and isolate noise points.
- **Dimensionality Reduction**: 2D projections using **PCA**, **t-SNE**, and **UMAP**.
- **Anomaly Detection**: **Isolation Forest** to flag anomalous student profiles based on academic and assessment discrepancies.

### 📊 Exploratory Data Analysis & Quality
- Data health checks: Missing values percentage, duplicate records, and feature-by-feature quality metrics.
- Distribution plots for CGPA, attendance, college tiers, and salary trends.
- Preprocessing pipelines: Min-Max normalization, standard scaling, and IQR-based outlier detection.

---

## 📁 Repository Structure

```
PlacementPrediction/
├── app.py                     # Main Flask web application and ML server
├── requirements.txt           # Project dependencies
├── pyproject.toml             # Project configuration
├── EDA.py                     # Standalone exploratory data analysis script
├── datasets/
│   └── placement_predict_50k Dataset.csv  # 50,000 student records dataset
├── Models/                    # Standalone supervised model training scripts
│   ├── Logistic_Regression.py
│   ├── Logistic_Regression_Scalar.py
│   ├── linear_regression.py
│   ├── decision_tree.py
│   ├── gradient_boosting.py
│   ├── addaboost.py
│   ├── bagging.py
│   └── boosting.py
├── Clustering/                # Standalone clustering & reduction scripts
│   ├── K-means_simple.py
│   ├── K_calculation_K_means.py
│   ├── K_means_final.py
│   ├── k_means_outliers.py
│   ├── DBSCAN.py
│   ├── dimensionality_reduction.py
│   └── anomaly_detection.py
├── Preprocessing/             # Data cleaning & normalization utilities
│   ├── normalization.py
│   └── outlier.py
├── templates/
│   └── index.html             # UI templates
└── static/
    └── style.css              # Custom styling
```

---

## 📊 Dataset Overview

The dataset contains **50,000+ student records** and **32 features**, including:

| Category | Features |
|---|---|
| **Academic Performance** | `SGPA_Sem1` to `SGPA_Sem8`, `CGPA`, `CGPA_Tier`, `HistoryOfBacklogs`, `AttendancePercent` |
| **Assessments & Skills** | `AptitudeTestScore`, `CodingTestScore`, `MockInterviewScore`, `SoftSkillsRating` |
| **Experience & Portfolio** | `Internships`, `Projects`, `Workshops`, `Certifications`, `Publications`, `ExtraCurricular` |
| **Demographics & College** | `Gender`, `City`, `CollegeTier`, `Stream`, `Specialisation`, `Hostel` |
| **Target Variables** | `PlacementStatus` (0 = Not Placed, 1 = Placed), `Salary Package` (LPA), `IsAnomaly` |

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/really-rajnish/PlacementPrediction.git
cd PlacementPrediction
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🛠️ Technology Stack

- **Backend**: [Flask](https://flask.palletsprojects.com/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/), [umap-learn](https://umap-learn.readthedocs.io/)
- **Data Manipulation**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/)
- **Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Frontend**: HTML5, Vanilla CSS3 (Custom design system), Jinja2 templates

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
