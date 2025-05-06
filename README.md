
# 🏦 Loan Approval Prediction using Logistic Regression

This project is a machine learning solution to predict whether a loan application should be approved or not, based on applicant data. It demonstrates a complete pipeline including data preprocessing, model training, and evaluation, all implemented in a Kaggle Notebook.

## 📁 Project Overview

Loan approval is a binary classification problem. This project uses a logistic regression model to predict loan status (`Y` for approved, `N` for not approved) based on various applicant features.

- **Dataset Source**: [Loan Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Notebook**: [View on Kaggle](https://www.kaggle.com/code/gowthaminanapollu/loan-approval-ml-model)
- **Model**: Logistic Regression
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## 📊 Features Used

- Gender
- Age
- Married
- Education
- Employment Experience
- ApplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit Score
- Credit_History
- Property

## ⚙️ Workflow Summary

1. **Data Loading**  
   Load training dataset and inspect shape, columns, and basic statistics.

2. **Data Preprocessing**  
   - Handle missing values (e.g., imputing with mode/median)
   - Convert categorical variables to numeric using label encoding or one-hot encoding

3. **Exploratory Data Analysis (EDA)**  
   - Understand feature distributions using histograms and count plots
   - Analyze relationships with the target variable

4. **Model Building**  
   - Apply `LogisticRegression` from `sklearn.linear_model`
   - Split data into training and test sets
   - Fit the model on training data

5. **Model Evaluation**  
   - Evaluate with accuracy score
   - Display confusion matrix for more insight

## 📈 Results

The logistic regression model provided a solid starting point for predicting loan approvals. With basic preprocessing, it was able to deliver reasonable accuracy. Further improvements can be made by enhancing features or using more advanced models.

## 🚀 Future Work

- Hyperparameter tuning
- Add cross-validation
- Try more complex models (e.g., Random Forest, XGBoost)
- Use metrics like precision, recall, F1-score
- Address class imbalance (if applicable)

## 💡 How to Run

This notebook was developed and run on Kaggle. To run locally:

1. Clone the repository
2. Install the dependencies (see below)
3. Run the notebook using Jupyter or another Python environment

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
