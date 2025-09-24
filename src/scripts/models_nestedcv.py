from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from scripts.utils import bank_cost

def nested_cv(
    X, y, 
    base_model,   
    n_estimators_list=[100, 200, 300],
    max_depth_list=[5, 10, None],
    threshold_list=[0.3, 0.5, 0.7, 0.8, 0.9],
    num_outer_folds=5,
    num_inner_folds=3,
    fp_cost=5,
    fn_cost=1,
    random_state=42
):
    # Shuffle data
    np.random.seed(random_state)
    shuffle_idx = np.random.permutation(len(X))
    X_shuffled, y_shuffled = X.values[shuffle_idx], y.values[shuffle_idx]

    outer_accuracies, outer_bank_costs, best_params_list = [], [], []

    skf_outer = StratifiedKFold(n_splits=num_outer_folds, shuffle=True, random_state=random_state)
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(skf_outer.split(X_shuffled, y_shuffled), 1):
        X_outer_train, X_outer_test = X_shuffled[outer_train_idx], X_shuffled[outer_test_idx]
        y_outer_train, y_outer_test = y_shuffled[outer_train_idx], y_shuffled[outer_test_idx]

        best_inner_cost, best_params = None, None
        skf_inner = StratifiedKFold(n_splits=num_inner_folds, shuffle=True, random_state=random_state)

        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                for threshold in threshold_list:
                    inner_costs = []

                    for inner_train_idx, inner_test_idx in skf_inner.split(X_outer_train, y_outer_train):
                        X_inner_train, X_val = X_outer_train[inner_train_idx], X_outer_train[inner_test_idx]
                        y_inner_train, y_val = y_outer_train[inner_train_idx], y_outer_train[inner_test_idx]

                        # Copy the base model with new params
                        params = {"max_depth": max_depth, "random_state": random_state}
                        if "n_estimators" in base_model.get_params().keys():
                            params["n_estimators"] = n_estimators

                        clf = base_model.__class__(**params)
                        clf.fit(X_inner_train, y_inner_train)

                        y_val_proba = clf.predict_proba(X_val)[:, 1]
                        y_val_pred = (y_val_proba >= threshold).astype(int)

                        inner_costs.append(bank_cost(y_val, y_val_pred, fp_cost, fn_cost))

                    mean_inner_cost = np.mean(inner_costs)
                    if best_inner_cost is None or mean_inner_cost < best_inner_cost:
                        best_inner_cost = mean_inner_cost
                        best_params = {**params, "threshold": threshold}

        # Train best model
        best_clf = base_model.__class__(**{k:v for k,v in best_params.items() if k!="threshold"})
        best_clf.fit(X_outer_train, y_outer_train)

        y_outer_proba = best_clf.predict_proba(X_outer_test)[:, 1]
        y_outer_pred = (y_outer_proba >= best_params["threshold"]).astype(int)

        outer_acc = accuracy_score(y_outer_test, y_outer_pred)
        outer_prec = precision_score(y_outer_test, y_outer_pred, zero_division=0)
        outer_rec = recall_score(y_outer_test, y_outer_pred, zero_division=0)
        outer_auc = roc_auc_score(y_outer_test, y_outer_proba)
        outer_cost = bank_cost(y_outer_test, y_outer_pred, fp_cost, fn_cost)

        outer_accuracies.append(outer_acc)
        outer_bank_costs.append(outer_cost)
        best_params_list.append(best_params)

        print(f"Fold {fold_idx}: Best Params: {best_params}, "
              f"Accuracy: {outer_acc:.3f}, Precision: {outer_prec:.3f}, "
              f"Recall: {outer_rec:.3f}, AUC: {outer_auc:.3f}, Bank Cost: {outer_cost}")

    avg_outer_acc = np.mean(outer_accuracies)
    avg_outer_cost = np.mean(outer_bank_costs)
    print(f"\nAverage Accuracy across outer folds: {avg_outer_acc:.3f}")
    print(f"Average Bank Cost across outer folds: {avg_outer_cost:.3f}")

    return best_params_list, outer_bank_costs, avg_outer_acc, avg_outer_cost
