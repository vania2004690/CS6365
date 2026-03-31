
# supervised_model.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from data_preprocessing import load_and_preprocess
from config import DATA_PATH, TEST_SIZE, RANDOM_STATE

def run_supervised_model(path=DATA_PATH, model_name="gbt"):
    X, y, pipe, meta = load_and_preprocess(path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE
        )
    elif model_name == "logreg":
        model = LogisticRegression(max_iter=2000, n_jobs=None, random_state=RANDOM_STATE)
    else:  # default: GBT
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: decision_function or binary preds as probs
        y_prob = getattr(model, "decision_function", lambda X: y_pred)(X_test)
        # scale to 0-1 if needed
        if y_prob.ndim == 1:
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "rows_before": meta["rows_before"],
        "rows_after_clean": meta["rows_after_clean"],
        "n_features_after_pipeline": X.shape[1],
        "num_cols_used": meta["num_cols_used"],
        "cat_cols_used": meta["cat_cols_used"],
        "label_rule": meta["label_rule"],
        "model": model.__class__.__name__,
    }

    return model, pipe, metrics


def plot_supervised_metrics(metrics: dict, output_path: str = "supervised_metrics_visualization.png"):
    """Create a quick visualization of the core supervised metrics."""
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [metrics[m] for m in metric_names]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(metric_names, values, marker="o", linewidth=2, markersize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{metrics.get('model', 'Model')} Performance Metrics", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    for x, val in zip(metric_names, values):
        ax.text(x, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved supervised model visualization to {output_path}")


if __name__ == "__main__":
    model, pipe, metrics = run_supervised_model()
    for k, v in metrics.items():
        print(f"{k}: {v}")
    plot_supervised_metrics(metrics)
