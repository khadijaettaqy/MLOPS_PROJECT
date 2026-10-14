"""
Model Selection & Training for Heart Disease Dataset
=====================================================

Trains two classifiers:
  1. Logistic Regression
  2. Random Forest

Uses:
  - GridSearchCV with 5-fold stratified CV
  - MLflow with SQLite tracking backend
  - MLflow sklearn model logging
  - Hold-out test evaluation
  - Confusion matrix and ROC curve artifacts
"""

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    make_scorer,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "heart.csv"

MODEL_PATH = BASE_DIR / "src" / "models" / "heart_model.pkl"

MLFLOW_DB = BASE_DIR / "mlflow.db"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filepath=None):
    """Load and clean the heart disease dataset."""

    if filepath is None:
        filepath = DATA_PATH

    df = pd.read_csv(filepath)

    for col in df.columns:

        if df[col].isnull().sum() > 0:

            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())

            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODELS = {

    "Logistic Regression": {

        "pipeline": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            ),
        ]),

        "param_grid": {

            "classifier__C": [
                0.01,
                0.1,
                1,
                10
            ],

            "classifier__solver": [
                "lbfgs",
                "liblinear"
            ],
        },
    },


    "Random Forest": {

        "pipeline": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42
                )
            ),
        ]),

        "param_grid": {

            "classifier__n_estimators": [
                100,
                200
            ],

            "classifier__max_depth": [
                5,
                10,
                None
            ],

            "classifier__min_samples_split": [
                2,
                5
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Cross-validation evaluation
# ---------------------------------------------------------------------------

def cross_val_evaluate(pipeline, X, y, cv):

    scoring = {

        "accuracy": "accuracy",

        "precision": make_scorer(
            precision_score,
            zero_division=0
        ),

        "recall": make_scorer(
            recall_score,
            zero_division=0
        ),

        "roc_auc": "roc_auc",
    }

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    summary = {

        metric: (
            results[f"test_{metric}"].mean(),
            results[f"test_{metric}"].std()
        )

        for metric in scoring
    }

    return summary


# ---------------------------------------------------------------------------
# Print CV results
# ---------------------------------------------------------------------------

def print_cv_summary(name, summary):

    print(f"\n  Cross-Validation Results ({name})")

    print(
        f"  {'Metric':<12} "
        f"{'Mean':>8}  "
        f"{'Std':>8}"
    )

    print("  " + "-" * 32)

    for metric, (mean, std) in summary.items():

        print(
            f"  {metric:<12} "
            f"{mean:>8.4f}  "
            f"{std:>8.4f}"
        )


# ---------------------------------------------------------------------------
# Print test metrics
# ---------------------------------------------------------------------------

def print_test_metrics(
    name,
    y_test,
    y_pred,
    y_prob
):

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    print(
        f"\n  Hold-out Test Metrics ({name})"
    )

    print(
        f"  ROC-AUC : {auc:.4f}"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Disease",
                "Disease"
            ]
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # -----------------------------------------------------------------------
    # MLflow configuration
    # -----------------------------------------------------------------------

    tracking_uri = f"sqlite:///{MLFLOW_DB}"

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        "heart-disease-classification"
    )

    print("=" * 60)
    print("  MLOPS / MLflow CONFIGURATION")
    print("=" * 60)

    print(
        f"  MLflow tracking URI : {tracking_uri}"
    )

    print(
        f"  MLflow database     : {MLFLOW_DB}"
    )


    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------

    df = load_data()

    X = df[FEATURE_COLS]

    y = df["target"]


    # -----------------------------------------------------------------------
    # Train / test split
    # -----------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )


    # -----------------------------------------------------------------------
    # Cross-validation
    # -----------------------------------------------------------------------

    cv = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=42
    )


    # -----------------------------------------------------------------------
    # Variables for model selection
    # -----------------------------------------------------------------------

    best_name = None

    best_pipeline = None

    best_auc = -1.0

    results_summary = {}


    # -----------------------------------------------------------------------
    # Model selection
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("  MODEL SELECTION & TUNING")
    print("=" * 60)


    for name, spec in MODELS.items():

        print(f"\n[{name}]")

        print(
            "  Tuning hyperparameters via "
            "GridSearchCV (5-fold CV) ..."
        )


        # -------------------------------------------------------------------
        # MLflow run
        # -------------------------------------------------------------------

        with mlflow.start_run(
            run_name=name
        ):

            grid = GridSearchCV(

                estimator=spec["pipeline"],

                param_grid=spec["param_grid"],

                cv=cv,

                scoring="roc_auc",

                n_jobs=-1,

                refit=True
            )


            # ---------------------------------------------------------------
            # Train
            # ---------------------------------------------------------------

            grid.fit(
                X_train,
                y_train
            )


            # ---------------------------------------------------------------
            # Best parameters
            # ---------------------------------------------------------------

            best_params = {

                k.replace(
                    "classifier__",
                    ""
                ): v

                for k, v in grid.best_params_.items()
            }


            print(
                f"  Best params : {best_params}"
            )

            print(
                f"  Best CV AUC : "
                f"{grid.best_score_:.4f}"
            )


            # ---------------------------------------------------------------
            # Evaluate tuned pipeline
            # ---------------------------------------------------------------

            tuned_pipeline = (
                grid.best_estimator_
            )


            cv_summary = cross_val_evaluate(

                tuned_pipeline,

                X_train,

                y_train,

                cv
            )


            print_cv_summary(
                name,
                cv_summary
            )


            # ---------------------------------------------------------------
            # Log parameters
            # ---------------------------------------------------------------

            mlflow.log_params(
                best_params
            )

            mlflow.log_param(
                "model_type",
                name
            )


            # ---------------------------------------------------------------
            # Log CV metrics
            # ---------------------------------------------------------------

            for metric, (mean, std) in cv_summary.items():

                mlflow.log_metric(
                    f"cv_{metric}_mean",
                    mean
                )

                mlflow.log_metric(
                    f"cv_{metric}_std",
                    std
                )


            # ---------------------------------------------------------------
            # Test-set prediction
            # ---------------------------------------------------------------

            y_pred = tuned_pipeline.predict(
                X_test
            )

            y_prob = tuned_pipeline.predict_proba(
                X_test
            )[:, 1]


            # ---------------------------------------------------------------
            # Test metrics
            # ---------------------------------------------------------------

            test_auc = roc_auc_score(
                y_test,
                y_prob
            )

            test_precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            test_recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )


            mlflow.log_metric(
                "test_roc_auc",
                test_auc
            )

            mlflow.log_metric(
                "test_precision",
                test_precision
            )

            mlflow.log_metric(
                "test_recall",
                test_recall
            )


            # ---------------------------------------------------------------
            # Log sklearn model
            #
            # RandomForest contains:
            # sklearn.tree._tree.Tree
            #
            # MLflow/skops requires this type to be explicitly trusted.
            # ---------------------------------------------------------------

            mlflow.sklearn.log_model(

                tuned_pipeline,

                artifact_path="model",

                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )


            # ---------------------------------------------------------------
            # Confusion matrix
            # ---------------------------------------------------------------

            cm = confusion_matrix(
                y_test,
                y_pred
            )


            fig_cm, ax_cm = plt.subplots(
                figsize=(5, 4)
            )


            ax_cm.imshow(
                cm,
                cmap="Blues"
            )


            for i in range(2):

                for j in range(2):

                    ax_cm.text(

                        j,
                        i,

                        str(cm[i, j]),

                        ha="center",

                        va="center",

                        color=(
                            "white"
                            if cm[i, j] >
                            cm.max() / 2
                            else "black"
                        ),

                        fontsize=14
                    )


            ax_cm.set_xlabel(
                "Predicted"
            )

            ax_cm.set_ylabel(
                "Actual"
            )

            ax_cm.set_title(
                f"Confusion Matrix — {name}"
            )


            ax_cm.set_xticks(
                [0, 1]
            )

            ax_cm.set_yticks(
                [0, 1]
            )


            ax_cm.set_xticklabels([
                "No Disease",
                "Disease"
            ])

            ax_cm.set_yticklabels([
                "No Disease",
                "Disease"
            ])


            fig_cm.tight_layout()


            screenshots_dir = (
                BASE_DIR / "screenshots"
            )

            screenshots_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            cm_path = (
                screenshots_dir
                / f"cm_{name.replace(' ', '_').lower()}.png"
            )


            fig_cm.savefig(
                cm_path,
                dpi=100
            )


            mlflow.log_artifact(
                str(cm_path)
            )


            plt.close(
                fig_cm
            )


            # ---------------------------------------------------------------
            # ROC curve
            # ---------------------------------------------------------------

            fpr, tpr, _ = roc_curve(
                y_test,
                y_prob
            )


            fig_roc, ax_roc = plt.subplots(
                figsize=(5, 4)
            )


            ax_roc.plot(

                fpr,

                tpr,

                label=(
                    f"{name} "
                    f"(AUC={test_auc:.3f})"
                )
            )


            ax_roc.plot(
                [0, 1],
                [0, 1],
                "k--",
                alpha=0.5
            )


            ax_roc.set_xlabel(
                "False Positive Rate"
            )

            ax_roc.set_ylabel(
                "True Positive Rate"
            )

            ax_roc.set_title(
                f"ROC Curve — {name}"
            )

            ax_roc.legend(
                loc="lower right"
            )


            fig_roc.tight_layout()


            roc_path = (
                screenshots_dir
                / f"roc_{name.replace(' ', '_').lower()}.png"
            )


            fig_roc.savefig(
                roc_path,
                dpi=100
            )


            mlflow.log_artifact(
                str(roc_path)
            )


            plt.close(
                fig_roc
            )


            # ---------------------------------------------------------------
            # Save results
            # ---------------------------------------------------------------

            results_summary[name] = {

                "pipeline": tuned_pipeline,

                "cv_summary": cv_summary,

                "best_params": best_params,

                "cv_auc": (
                    cv_summary["roc_auc"][0]
                )
            }


            # ---------------------------------------------------------------
            # Select best model
            # ---------------------------------------------------------------

            if (
                cv_summary["roc_auc"][0]
                > best_auc
            ):

                best_auc = (
                    cv_summary["roc_auc"][0]
                )

                best_name = name

                best_pipeline = (
                    tuned_pipeline
                )


    # -----------------------------------------------------------------------
    # Final test-set evaluation
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "  FINAL TEST-SET EVALUATION"
    )

    print("=" * 60)


    for name, info in results_summary.items():

        pipe = info["pipeline"]

        y_pred = pipe.predict(
            X_test
        )

        y_prob = pipe.predict_proba(
            X_test
        )[:, 1]


        print_test_metrics(

            name,

            y_test,

            y_pred,

            y_prob
        )


    # -----------------------------------------------------------------------
    # Model selection decision
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "  MODEL SELECTION DECISION"
    )

    print("=" * 60)


    for name, info in results_summary.items():

        mean_auc, std_auc = (
            info["cv_summary"]["roc_auc"]
        )


        marker = (
            "  <-- SELECTED"
            if name == best_name
            else ""
        )


        print(

            f"  {name:<22} "
            f"CV ROC-AUC = "
            f"{mean_auc:.4f} ± "
            f"{std_auc:.4f}"
            f"{marker}"
        )


    print(

        f"\n  Winner : {best_name} "
        f"(highest mean CV ROC-AUC = "
        f"{best_auc:.4f})"
    )


    print(

        "  Rationale: model chosen purely "
        "on generalisation performance "
        "measured by stratified 5-fold "
        "cross-validation ROC-AUC.\n"
    )


    # -----------------------------------------------------------------------
    # Save best model
    # -----------------------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    joblib.dump(
        best_pipeline,
        MODEL_PATH
    )


    print(
        f"  Model saved → {MODEL_PATH}"
    )


    # -----------------------------------------------------------------------
    # MLflow run for final selected model
    # -----------------------------------------------------------------------

    with mlflow.start_run(

        run_name=(
            f"best-model-{best_name}"
        )
    ):

        mlflow.log_param(
            "selected_model",
            best_name
        )


        mlflow.log_metric(
            "best_cv_roc_auc",
            best_auc
        )


        mlflow.sklearn.log_model(

            best_pipeline,

            artifact_path="best_model",

            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )


        mlflow.log_artifact(
            str(MODEL_PATH)
        )


    # -----------------------------------------------------------------------
    # Finish
    # -----------------------------------------------------------------------

    print(
        f"  MLflow tracking URI → "
        f"{tracking_uri}"
    )

    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
