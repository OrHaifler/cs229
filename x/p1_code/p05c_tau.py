import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    # Search tau_values for the best tau (lowest MSE on the validation set)
    best_tau = tau_values[0]
    model = LocallyWeightedLinearRegression(best_tau)
    model.fit(x_train, y_train)
    best_mse = float('inf')
    for tau in tau_values:
        model.tau = tau
        y_pred = model.predict(x_val)
        MSE = np.mean((y_pred - y_val) ** 2)
        plt.figure()
        plt.title('tau = {}'.format(tau))
        plt.plot(x_train, y_train, 'bx', linewidth=2)
        plt.plot(x_val, y_pred, 'ro', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('output/p05c_tau_{}.png'.format(tau))
        if MSE < best_mse:
            best_tau = tau
            best_mse = MSE
    # Fit a LWR model with the best tau value
    model.tau = best_tau
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = model.predict(x_test)
    MSE = np.mean((y_pred - y_test) ** 2)
    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred)
    print(model.tau)
    print(f'Best tau: {best_tau}, test set mse: {MSE}')
    # Plot data

    # *** END CODE HERE ***
