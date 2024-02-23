import numpy as np

def train_ridge_regression(K_train, y_train, rho):
    """
    Train a ridge regression model.

    :param K_train: Kernel matrix for the training data, shape (n_train, n_train)
    :param y_train: Training target values, shape (n_train,)
    :param rho: Regularization parameter
    :return: Model parameters alpha
    """
    n_train = K_train.shape[0]
    # Ridge regression solution: (K_train + rho * I)^(-1) * y_train
    alpha = np.linalg.pinv(K_train + rho * np.eye(n_train)) @ y_train
    return alpha

def predict_ridge_regression(K_test, alpha):
    """
    Make predictions using ridge regression model.

    :param K_test: Kernel matrix for the test data, shape (n_train, n_test)
    :param alpha: Model parameters from the trained ridge regression, shape (n_train,)
    :return: Predictions y_pred, shape (n_test,)
    """
    # Prediction: y_pred = K_test^T * alpha
    y_pred = K_test.T @ alpha
    return y_pred

def compute_mse(y,yhat):
    return np.mean((y - yhat)**2)
    
def compute_training_error(K_train, y_train, alpha):
    """
    Compute the training error for the ridge regression model.

    :param K_train: Kernel matrix for the training data, shape (n_train, n_train)
    :param y_train: Training target values, shape (n_train,)
    :param alpha: Model parameters from the trained ridge regression, shape (n_train,)
    :return: Training error
    """
    # Predict training data
    y_train_pred = K_train @ alpha

    # Compute Mean Squared Error (MSE) as the training error
    return compute_mse(y_train, y_train_pred)
