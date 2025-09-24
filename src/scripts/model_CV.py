import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from scripts.utils import bank_cost

def kfold_cv(X, y, model_class, model_params=None, n_splits=10, random_state=42, threshold=0.7):
    """
    Perform Stratified K-fold cross-validation with a customizable decision threshold.
    """
    if model_params is None:
        model_params = {}
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    accuracies, precisions, recalls, f1s, bank_costs = [], [], [], [], []
    y_true_all, y_score_all = [], []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        clf = model_class(**model_params)
        clf.fit(X_train, y_train)
        
        # Predict depending on model type
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_score >= threshold).astype(int)
        else:
            y_pred = clf.predict(X_test)
            y_score = None
        
        # Store
        y_true_all.extend(y_test)
        if y_score is not None:
            y_score_all.extend(y_score)
        
        # Compute metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        bank_costs.append(bank_cost(y_test, y_pred))
    
    # Print summary
    print("\nStratified Cross-Validation Results (Averages):")
    print(f"Accuracy : {np.mean(accuracies):.2f}")
    print(f"Precision: {np.mean(precisions):.2f}")
    print(f"Recall   : {np.mean(recalls):.2f}")
    print(f"F1 Score : {np.mean(f1s):.2f}")
    print(f"Bank Cost: {np.mean(bank_costs):.2f}")
    
    results = {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1s),
        "bank_cost": np.mean(bank_costs),
        "true_lbl": np.array(y_true_all),
        "pred_score": np.array(y_score_all) if y_score_all else None
    }
    
    return results
