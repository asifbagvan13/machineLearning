# 🏦 Loan Approval Prediction using K-Nearest Neighbors (KNN)

This project implements a simple machine learning model to predict whether a loan will be approved or rejected based on applicant details. It uses the **K-Nearest Neighbors (KNN)** algorithm from **scikit-learn** and is trained on a dataset containing various financial and personal features of applicants.

# Requirements
- Python 3.13.5
- pandas
- numpy
- scikit-learn


## to run this download it, 
- step 1: conda create -n ML python=3.13.5
- step 2: conda activate ML
- step 3: pip install -r requirement.txt
- step 4: then you can run it uisng your preferred ide or code editor 
## score
- Model Score: 0.76
## 📁 Dataset

The dataset (`loan_approval_dataset.csv`) contains 12 columns including:

- `loan_id` *(dropped during preprocessing)*
- `no_of_dependents`
- `education`
- `self_employed`
- `income_annum`
- `loan_amount`
- `loan_term`
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`
- `loan_status` *(target variable)*

## ⚙️ Preprocessing

- Categorical values like `education`, `self_employed`, and `loan_status` are converted to numeric:
  - `Graduate` → `1`, `Not Graduate` → `0`
  - `Yes` → `1`, `No` → `0`
  - `Approved` → `1`, `Rejected` → `0`
- The `loan_id` column is dropped.
- Train-test split: 80% training, 20% testing.

## 🧠 Model

A **K-Nearest Neighbors Classifier** is trained with `n_neighbors=11`.

## code

<pre>  ```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

df = pd.DataFrame(pd.read_csv('./loan_approval_dataset.csv'))
df.head()
df.drop('loan_id', axis=1)
df[' education'] = df[' education'].replace({' Graduate':1,' Not Graduate':0})
df[' self_employed'] = df[' self_employed'].replace({' No':0,' Yes':1})
df[' loan_status'] = df[' loan_status'].replace({' Approved':1, ' Rejected':0})

df.columns

features1 = [' no_of_dependents', ' education', ' self_employed',
       ' income_annum', ' loan_amount', ' loan_term', ' cibil_score',
       ' residential_assets_value', ' commercial_assets_value',
       ' luxury_assets_value', ' bank_asset_value']

target = [' loan_status']
x1 = df[features1]
y1 = df[target]
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=11)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

score = model.score(x_test, y_test)
print("Model Score:", score)
new_loan = pd.DataFrame({
    ' no_of_dependents': [2],
    ' education': [1],
    ' self_employed': [0],
    ' income_annum': [5000000],
    ' loan_amount': [10000],
    ' loan_term': [10],
    ' cibil_score': [700],
    ' residential_assets_value': [1000000],
    ' commercial_assets_value': [500000],
    ' luxury_assets_value': [2000000],
    ' bank_asset_value': [1000000]
})

approve_or_not = model.predict(new_loan)
print("Predicted approve orr not:", approve_or_not)

if approve_or_not == 1:
  print("Loan Approved")
else:
  print("Loan Rejected") ```</pre>
## output
Predicted approve or not: [1]
Loan Approved



