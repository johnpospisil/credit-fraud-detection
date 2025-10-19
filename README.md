# Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using various classification algorithms.

## 📊 Project Overview

This project analyzes a highly imbalanced credit card transaction dataset and implements multiple machine learning models to effectively detect fraudulent transactions. The dataset contains 284,807 transactions with only 0.17% being fraudulent (492 out of 284,807).

## 🎯 Key Features

- **Exploratory Data Analysis**: Comprehensive analysis of transaction patterns, temporal trends, and feature distributions
- **Class Imbalance Handling**: Implementation of SMOTE and class weighting techniques
- **Multiple ML Models**: Comparison of Logistic Regression, Random Forest, XGBoost, and Neural Networks
- **Performance Optimization**: Hyperparameter tuning and threshold optimization
- **Model Interpretation**: Feature importance analysis and error analysis

## 📁 Project Structure

```
credit-fraud-detection/
├── credit-fraud-detection.ipynb    # Main Jupyter notebook with full analysis
├── data/
│   └── creditcard.csv             # Dataset (not included in repo)
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore rules
```

## 🔍 Dataset

The dataset contains transactions made by European cardholders in September 2013. Features V1-V28 are PCA-transformed due to confidentiality, while Time and Amount are original features.

**Dataset Characteristics:**

- 284,807 transactions
- 31 features (Time, Amount, V1-V28, Class)
- Highly imbalanced: 99.83% legitimate, 0.17% fraudulent
- No missing values

## 🚀 Project Phases

1. **Data Preprocessing & Feature Engineering**
2. **Baseline Model - Logistic Regression**
3. **Class Imbalance Handling with SMOTE**
4. **Random Forest Classifier**
5. **XGBoost Model**
6. **Neural Network Model**
7. **Model Comparison & Selection**
8. **Model Interpretation & Error Analysis**
9. **Threshold Optimization**
10. **Final Report & Deployment Preparation**

## 📊 Key Findings

- **Class Imbalance**: 1:578 ratio (fraud:non-fraud)
- **Top Correlated Features**: V17, V14, V12 (negative), V11, V4, V2 (positive)
- **Temporal Patterns**: Fraud rates vary throughout the 48-hour period
- **Amount Analysis**: Fraud transactions have lower median but higher mean amounts

## 🛠️ Technologies Used

- **Python 3.12**
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Deep Learning**: TensorFlow/Keras
- **Imbalanced Learning**: imbalanced-learn (SMOTE)

## 📈 Model Performance

(To be updated after completing model training phases)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/johnpospisil/credit-fraud-detection.git
cd credit-fraud-detection

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow imbalanced-learn
```

## 💻 Usage

1. Download the credit card fraud dataset and place it in the `data/` folder
2. Open `credit-fraud-detection.ipynb` in Jupyter Notebook or JupyterLab
3. Run cells sequentially to reproduce the analysis

## 📝 Evaluation Metrics

Due to severe class imbalance, we focus on:

- **Precision**: Proportion of predicted frauds that are actual frauds
- **Recall**: Proportion of actual frauds that are detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Area under the Precision-Recall curve

## 🤝 Contributing

This is a personal learning project, but suggestions and feedback are welcome!

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

John Pospisil

## 🙏 Acknowledgments

- Dataset source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Original research by ULB Machine Learning Group

---

**Note**: This project is for educational purposes. Always ensure proper data handling and privacy compliance when working with financial data in production environments.
