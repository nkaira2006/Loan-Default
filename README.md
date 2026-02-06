# Loan Default Prediction

This repository contains an end-to-end credit risk modelling project that predicts loan defaults using machine learning. The focus is on business-aligned evaluation — specifically maximizing recall for defaulters, as missing a risky borrower is costlier than incorrectly flagging a safe borrower.

---

## **Problem Statement**

Loan default prediction is a critical task in finance where:

- **Target variable: `Status`**
  - `0`: Safe borrower (likely to repay)
  - `1`: Risky borrower (likely to default)

**Primary goal:**  
Capture as many defaulters as possible (high recall for class `1`)  
Fewer false negatives is prioritized over overall accuracy.

---

## **Dataset Overview**

- ~150,000 records of mortgage/loan applications  
- Mix of borrower demographics, loan features, collateral details, and application metadata  
- The dataset reflects real banking data with class imbalance and noisy entries

**Feature categories:**
- Numeric: loan amount, income, LTV, dtir1, credit score
- Categorical: loan purpose, loan type, age group, region
- Binary/ordinal: neg amortization, interest only, lumpsum payment

---

## **Exploratory Data Analysis (EDA)**

Conducted insights include:

- Higher Loan-to-Value ratios correlate with higher defaults  
- Negative amortization loans show elevated risk  
- Debt-to-Income patterns vary by income and default status  
- No single feature perfectly separates defaulters  
- Default risk arises from feature interactions

(Figures and plots included in the notebook.)

---

## **Feature Engineering**

- Created financial ratios such as `loan_to_income` and `loan_to_property`
- Applied log transforms to reduce skew (e.g., `log_income`)
- Binned continuous metrics like credit score and LTV for ordinality
- Carefully managed missing values with median imputation and “Unknown” tagging

---

## **Preprocessing & Encoding Strategy**

- Proper imputation for numeric columns
- **Ordinal encoding** for ordered categories like LTV bins
- **One-hot encoding** for nominal categorical columns
- Infinite values handled and capped to avoid model breakage

---

## **Modeling**

### Models used:
- Decision Tree (main model, high interpretability)
- Random Forest (baseline tree ensemble)
- Logistic Regression (linear baseline)

All models were trained with `class_weight='balanced'` to handle class imbalance.

---

## **Evaluation Strategy**

- Train/test split (stratified) for unbiased evaluation
- **5-fold Stratified Cross-Validation** focused on recall
- **Label shuffle test** to verify no data leakage

## Dataset Source: Kaggle
