"""
Model Comparison Module
Trains 4 classification models with cross-validation and generates comparison metrics.
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR.parent / "telecom_customer_churn.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def load_data():
    """Load and prepare the telecom churn dataset."""
    df = pd.read_csv(DATA_PATH)
    features = joblib.load(ARTIFACTS_DIR / "model_features.pkl")
    
    # Binary target
    df["Churn"] = (df["Customer Status"] == "Churned").astype(int)
    
    # Drop rows with 'Joined' status (new customers, no churn history)
    df = df[df["Customer Status"] != "Joined"].copy()
    
    X = df[features].copy()
    y = df["Churn"].copy()
    
    # Fill NaN values
    string_cols = X.select_dtypes(include="object").columns
    numeric_cols = X.select_dtypes(include="number").columns
    X[string_cols] = X[string_cols].fillna("No")
    X[numeric_cols] = X[numeric_cols].fillna(0)
    
    return X, y, features


def build_preprocessor(X):
    """Build a ColumnTransformer matching the existing pipeline pattern."""
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return preprocessor


def get_models():
    """Return dict of model name -> model instance."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric="logloss",
            verbosity=0
        )
    except ImportError:
        logger.warning("XGBoost not installed, skipping")
    
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1
        )
    except ImportError:
        logger.warning("LightGBM not installed, skipping")
    
    return models


def run_comparison():
    """Train all models with 5-fold stratified CV 和 return comparison results."""
    logger.info("Loading data...")
    X, y, features = load_data()
    
    preprocessor = build_preprocessor(X)
    models = get_models()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    roc_data = {}
    best_score = 0
    best_model_name = None
    best_pipeline = None
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
        all_y_true, all_y_prob = [], []
        cm_total = np.zeros((2, 2), dtype=int)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])
            
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            y_prob = pipe.predict_proba(X_val)[:, 1]
            
            fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
            fold_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            fold_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            fold_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            fold_metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))
            
            all_y_true.extend(y_val.tolist())
            all_y_prob.extend(y_prob.tolist())
            cm_total += confusion_matrix(y_val, y_pred)
        
        # Average metrics across folds
        avg_metrics = {k: round(float(np.mean(v)), 4) for k, v in fold_metrics.items()}
        avg_metrics["std_roc_auc"] = round(float(np.std(fold_metrics["roc_auc"])), 4)
        
        # Confusion matrix (aggregate)
        avg_metrics["confusion_matrix"] = cm_total.tolist()
        
        results[name] = avg_metrics
        
        # ROC curve data (aggregated across folds)
        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        
        # Track best model
        if avg_metrics["roc_auc"] > best_score:
            best_score = avg_metrics["roc_auc"]
            best_model_name = name
            # Train final model on full dataset
            best_pipeline = Pipeline([
                ("preprocessor", build_preprocessor(X)),
                ("classifier", model)
            ])
            best_pipeline.fit(X, y)
        
        logger.info(f"  {name}: ROC-AUC = {avg_metrics['roc_auc']:.4f} ± {avg_metrics['std_roc_auc']:.4f}")
    
    # Mark the best model
    for name in results:
        results[name]["is_best"] = (name == best_model_name)
    
    # Save results
    comparison_output = {
        "models": results,
        "best_model": best_model_name,
        "best_roc_auc": best_score,
        "n_samples": len(X),
        "n_features": len(features),
        "cv_folds": 5
    }
    
    with open(ARTIFACTS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison_output, f, indent=2)
    
    # Save best model
    joblib.dump(best_pipeline, ARTIFACTS_DIR / "best_model.pkl")
    
    # Generate ROC curve chart
    _plot_roc_curves(roc_data, results, ARTIFACTS_DIR / "roc_curves.png")
    
    # Generate precision-recall curves
    _plot_confusion_matrices(results, ARTIFACTS_DIR / "confusion_matrices.png")
    
    logger.info(f"\nBest model: {best_model_name} (ROC-AUC: {best_score:.4f})")
    logger.info(f"Artifacts saved to {ARTIFACTS_DIR}")
    
    return comparison_output


def _plot_roc_curves(roc_data, results, save_path):
    """Generate ROC curve comparison chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {"Logistic Regression": "#6366f1", "Random Forest": "#22c55e",
              "XGBoost": "#f59e0b", "LightGBM": "#ef4444"}
    
    for name, data in roc_data.items():
        auc_val = results[name]["roc_auc"]
        color = colors.get(name, "#888888")
        label = f"{name} (AUC = {auc_val:.4f})"
        if results[name]["is_best"]:
            label += " ★ Best"
        ax.plot(data["fpr"], data["tpr"], label=label, color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison — Model Selection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curves saved to {save_path}")


def _plot_confusion_matrices(results, save_path):
    """Generate confusion matrix comparison chart."""
    model_names = list(results.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    
    for ax, name in zip(axes, model_names):
        cm = np.array(results[name]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(name, fontsize=11, fontweight="bold")
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Stayed", "Churned"])
        ax.set_yticklabels(["Stayed", "Churned"])
    
    fig.suptitle("Confusion Matrices (Aggregated 5-Fold CV)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrices saved to {save_path}")


def load_comparison_results():
    """Load precomputed comparison results."""
    path = ARTIFACTS_DIR / "model_comparison.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_comparison()
