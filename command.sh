source .mlops/bin/activate

jupyter notebook notebooks/EDA.ipynb


python src/models/train.py


 python src/models/train.py
============================================================
  MLOPS / MLflow CONFIGURATION
============================================================
  MLflow tracking URI : sqlite:////home/dba/MLOPs-Assignment1/mlflow.db
  MLflow database     : /home/dba/MLOPs-Assignment1/mlflow.db

============================================================
  MODEL SELECTION & TUNING
============================================================

[Logistic Regression]
  Tuning hyperparameters via GridSearchCV (5-fold CV) ...
  Best params : {'C': 0.1, 'solver': 'liblinear'}
  Best CV AUC : 0.9004

  Cross-Validation Results (Logistic Regression)
  Metric           Mean       Std
  --------------------------------
  accuracy       0.8264    0.0282
  precision      0.8439    0.0329
  recall         0.7648    0.0797
  roc_auc        0.9004    0.0157
2026/10/06 23:34:07 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.

[Random Forest]
  Tuning hyperparameters via GridSearchCV (5-fold CV) ...
  Best params : {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
  Best CV AUC : 0.8837

  Cross-Validation Results (Random Forest)
  Metric           Mean       Std
  --------------------------------
  accuracy       0.8139    0.0143
  precision      0.8295    0.0436
  recall         0.7557    0.0756
  roc_auc        0.8837    0.0310
2026/10/06 23:34:20 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.

============================================================
  FINAL TEST-SET EVALUATION
============================================================

  Hold-out Test Metrics (Logistic Regression)
  ROC-AUC : 0.9578
              precision    recall  f1-score   support

  No Disease       0.93      0.82      0.87        33
     Disease       0.81      0.93      0.87        28

    accuracy                           0.87        61
   macro avg       0.87      0.87      0.87        61
weighted avg       0.88      0.87      0.87        61


  Hold-out Test Metrics (Random Forest)
  ROC-AUC : 0.9551
              precision    recall  f1-score   support

  No Disease       0.97      0.85      0.90        33
     Disease       0.84      0.96      0.90        28

    accuracy                           0.90        61
   macro avg       0.90      0.91      0.90        61
weighted avg       0.91      0.90      0.90        61


============================================================
  MODEL SELECTION DECISION
============================================================
  Logistic Regression    CV ROC-AUC = 0.9004 ± 0.0157  <-- SELECTED
  Random Forest          CV ROC-AUC = 0.8837 ± 0.0310

  Winner : Logistic Regression (highest mean CV ROC-AUC = 0.9004)
  Rationale: model chosen purely on generalisation performance measured by stratified 5-fold cross-validation ROC-AUC.

  Model saved → /home/dba/MLOPs-Assignment1/src/models/heart_model.pkl
2026/10/06 23:34:23 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
  MLflow tracking URI → sqlite:////home/dba/MLOPs-Assignment1/mlflow.db
============================================================
mlflow ui --backend-store-uri mlruns/

mlflow server \
  --backend-store-uri sqlite:////mlflow.db \
  --host 0.0.0.0 \
  --port 5000
