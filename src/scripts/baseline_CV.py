import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from scripts.utils import bank_cost

    
def baseline_kfold(X, y, n_splits=10, fp_cost=5, fn_cost=1):
    """
    Stratified K-Fold cross-validation for a majority-class baseline model.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies, precisions, recalls, f1s, bank_costs = [], [], [], [], []
    y_true_all, y_prob_all = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        # Split train and test using indices
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Majority class in training fold
        majority_class = y_train.mode()[0]

        # Predict majority class for all test samples
        y_pred = [majority_class] * len(y_test)

        # Simulated probabilities using class proportion for ROC compatibility
        y_prob = [y_train.mean()] * len(y_test)

        # Collect for ROC
        y_true_all.extend(y_test)
        y_prob_all.extend(y_prob)

        # Metrics
        accuracies.append(metrics.accuracy_score(y_test, y_pred))
        precisions.append(metrics.precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        recalls.append(metrics.recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1s.append(metrics.f1_score(y_test, y_pred, pos_label=1, zero_division=0))

        # Bank cost
        bank_costs.append(bank_cost(y_test, y_pred, fp_cost, fn_cost))

    # Average metrics
    print("\nBaseline Model (Stratified K-Fold CV) Average Performance:")
    print(f"Accuracy: {np.mean(accuracies):.3f}")
    print(f"Precision: {np.mean(precisions):.3f}")
    print(f"Recall: {np.mean(recalls):.3f}")
    print(f"F1 Score: {np.mean(f1s):.3f}")
    print(f"Bank Cost: {np.mean(bank_costs):.2f}")
    
    results_baseline = {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1s),
        "bank_cost": np.mean(bank_costs),
        "true_lbl": np.array(y_true_all),
        "pred_score": np.array(y_prob_all)
    }

    return results_baseline
