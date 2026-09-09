from sklearn.metrics import (
    root_mean_squared_error,
    accuracy_score,
    confusion_matrix
)


def evaluate_regression(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    return rmse


def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, cm