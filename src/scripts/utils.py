from sklearn.metrics import confusion_matrix

def bank_cost(y_true, y_pred, fp_cost=5, fn_cost=1):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp*fp_cost + fn*fn_cost
