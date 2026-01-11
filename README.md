# Employee Attrition Prediction (Recall-Oriented ML Project)

## Problem Definition

Employee attrition is a critical problem for organizations due to its high financial and operational cost.
The goal of this project is to predict whether an employee is likely to leave the company, with a strong focus on **minimizing missed attrition cases**.

Because failing to identify an employee who will leave is more costly than raising a false alarm, the problem is treated as a **recall-oriented binary classification task**.

---

## Dataset

- Number of records: 1,470
- Target variable: `Attrition` (1 = Yes, 0 = No)
- Class imbalance present (~16% attrition rate)
- Mix of numerical, ordinal, binary, and categorical features

The dataset was cleaned and processed in multiple stages:
- `df_ori`: Original dataset
- `df_clean`: Cleaned dataset (mappings, column removal)
- `df_fe`: Feature-engineered dataset used for modeling

---

## Feature Engineering

The feature engineering strategy prioritizes **simplicity, stability, and business interpretability**.

Key steps:
- Ordinal encoding for ordered variables (e.g., `BusinessTravel`, `JobLevel`)
- Binary encoding for Yes/No variables
- One-hot encoding for selected categorical variables
- Removal of redundant features (e.g., `JobRole`, due to overlap with `JobLevel`)
- Creation of business-relevant derived features such as:
  - Career stagnation indicators
  - Income normalization features
  - Aggregated satisfaction metrics

The goal was to avoid overfitting while preserving meaningful signals related to employee behavior and career dynamics.

---

## Model Selection

**Logistic Regression** was selected as the final model based on:

- Small-to-medium dataset size
- Predominantly linear relationships
- Strong baseline performance
- High interpretability, which is essential for HR decision-making

Tree-based models (e.g., Random Forest, XGBoost) were considered.
However, experiments showed that Logistic Regression achieved comparable or better recall with significantly better stability and explainability.
For this reason, Logistic Regression was chosen as the final model.

---

## Handling Class Imbalance

- Class imbalance was handled using **custom class weights**
- The primary evaluation metric is **Recall for the Attrition class (label = 1)**
- Accuracy was tracked but not used as the main optimization objective

---

## Threshold Optimization

Instead of using the default probability threshold (0.50), multiple thresholds were evaluated.

After analyzing precision–recall trade-offs, a threshold of:

```
BEST_THRESHOLD = 0.40
```


was selected to improve attrition recall while maintaining acceptable precision.

This threshold was selected based on validation data and business cost considerations, not on the test set, to avoid data leakage.

---

## Model Evaluation

Note: All recall values refer to the Attrition class (label = 1).

### Test Set Performance (Threshold = 0.40)

- Recall (Attrition): ~0.74
- Precision (Attrition): ~0.36
- Accuracy: ~0.74

Confusion Matrix:

```
[[184 63]
[ 12 35]]
```


This result significantly reduces false negatives (missed attrition cases), which aligns with the business objective.

---

## Cross-Validation Results

### Standard Cross-Validation (Threshold = 0.50)

- Mean Recall: ~0.70
- Standard Deviation: ~0.02

### Threshold-Aware Cross-Validation (Threshold = 0.40)

- Mean Recall: ~0.75
- Standard Deviation: ~0.05

These results confirm that the recall improvement generalizes across folds and is not due to overfitting.

---

## Business Impact

From a business perspective:

- The model acts as a **risk scoring tool**, not an automated decision-maker
- Employees can be grouped into low, medium, and high attrition risk categories
- HR teams can prioritize proactive interventions for high-risk employees
- Even with some false positives, the reduction in missed attrition cases leads to lower overall cost

---

## How to Run

1. Clone the repository
2. Open the Jupyter notebook
3. Run cells sequentially from top to bottom
4. The final model is trained using the `df_fe` dataset

### Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Final Remarks

This project demonstrates a full end-to-end machine learning workflow:
- Thoughtful feature engineering
- Metric-driven model optimization
- Threshold tuning aligned with business objectives
- Robust validation using cross-validation

The final model is simple, interpretable, and effective for real-world employee attrition risk assessment.

