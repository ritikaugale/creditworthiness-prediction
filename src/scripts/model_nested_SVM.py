from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
random_state =42
from scripts.utils import bank_cost
from sklearn.svm import SVC

def nested_cv_svm(
    X, y,
    C_list=[0.1, 1, 10, 100],
    gamma_list=[0.001, 0.01, 0.1, 1, 10],
    threshold_list=np.linspace(0.1, 0.9, 9),
    num_outer_folds=5,
    num_inner_folds=3,
    fp_cost=5,
    fn_cost=1,
    random_state=42
):
    outer_cv = StratifiedKFold(n_splits=num_outer_folds, shuffle=True, random_state=random_state)
    outer_metrics = []
    best_params_list = []

    fold_idx = 1
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]

        best_inner_score = np.inf
        best_params = None

        # Inner CV
        inner_cv = StratifiedKFold(n_splits=num_inner_folds, shuffle=True, random_state=random_state)
        for C in C_list:
            for gamma in gamma_list:
                for threshold in threshold_list:
                    inner_scores = []
                    for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y_train_outer):
                        X_train_inner = X_train_outer.iloc[inner_train_idx]
                        y_train_inner = y_train_outer.iloc[inner_train_idx]
                        X_val_inner = X_train_outer.iloc[inner_val_idx]
                        y_val_inner = y_train_outer.iloc[inner_val_idx]

                        clf = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=random_state)
                        clf.fit(X_train_inner, y_train_inner)
                        y_val_proba = clf.predict_proba(X_val_inner)[:, 1]
                        y_val_pred = (y_val_proba >= threshold).astype(int)

                        cost = bank_cost(y_val_inner, y_val_pred, fp_cost=fp_cost, fn_cost=fn_cost)
                        inner_scores.append(cost)

                    mean_inner_score = np.mean(inner_scores)
                    if mean_inner_score < best_inner_score:
                        best_inner_score = mean_inner_score
                        best_params = {'C': C, 'gamma': gamma, 'threshold': threshold}

        # Train best model on outer fold
        best_clf = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'],
                       probability=True, random_state=random_state)
        best_clf.fit(X_train_outer, y_train_outer)

        y_test_proba = best_clf.predict_proba(X_test_outer)[:, 1]
        y_test_pred = (y_test_proba >= best_params['threshold']).astype(int)

        # Metrics
        acc = accuracy_score(y_test_outer, y_test_pred)
        prec = precision_score(y_test_outer, y_test_pred, zero_division=0)
        rec = recall_score(y_test_outer, y_test_pred, zero_division=0)
        auc = roc_auc_score(y_test_outer, y_test_proba)
        cost = bank_cost(y_test_outer, y_test_pred, fp_cost=fp_cost, fn_cost=fn_cost)

        outer_metrics.append({
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'roc_auc': auc,
            'cost': cost
        })
        best_params_list.append(best_params)

        print(f"Fold {fold_idx}: Best Params: {best_params}, "
              f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, "
              f"AUC: {auc:.3f}, Cost: {cost}")
        fold_idx += 1

    # Average metrics
    avg_metrics = {k: np.mean([m[k] for m in outer_metrics]) for k in outer_metrics[0]}
    return best_params_list, outer_metrics, avg_metrics