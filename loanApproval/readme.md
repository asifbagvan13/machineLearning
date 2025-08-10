# 🏦 Loan Approval Prediction using K-Nearest Neighbors (KNN)

This project implements a simple machine learning model to predict whether a loan will be approved or rejected based on applicant details. It uses the **K-Nearest Neighbors (KNN)** algorithm from **scikit-learn** and is trained on a dataset containing various financial and personal features of applicants.

# Requirements
Python 3.13.5
pandas
numpy
scikit-learn

Install dependencies using:
## to run this download it, 
 step 1: conda create -n ML python=3.13.5
 step 2: conda activate ML
 step 3: pip install -r requirement.txt
 step 4: then you can run it uisng your preferred ide or code editor 


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

```python
model = KNeighborsClassifier(n_neighbors=11)
model.fit(x_train, y_train)

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

if approve_or_not == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")


