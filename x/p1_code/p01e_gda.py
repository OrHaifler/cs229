import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.theta = np.array([1,1,1])
    util.plot(x_train, y_train, model.theta, 'save/01ed.png')
    model.fit(x_train, y_train)

    util.plot(x_train, y_train, model.theta, 'save/01e.png')

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n = x.shape
        self.theta = np.zeros(n + 1)
        #Calculate phi bases on the labels
        phi = sum(y) / m
        #Calculate the means based on the labels and corresponding values
        mu_0 = np.sum(x * (np.ones_like(y) - y).reshape(m,1), axis=0) / np.sum(np.ones_like(y) - y)
        mu_1 = np.sum(x * y.reshape(m,1), axis=0) / np.sum(y)
        #A matrix for the calculations of the covariance matrix
        x_mu = np.array([x[i] - mu_0 if y[i] == 0 else x[i] - mu_1 for i in range(len(y))])
        #The covariance matrix and its inverse
        sigma = x_mu.T.dot(x_mu) / m
        inv_sigma = np.linalg.inv(sigma)
        #Save the model parameters
        self.theta[0] = (1 / 2 * (mu_0 + mu_1).dot(inv_sigma).dot(mu_0 - mu_1)
                         - np.log((1 - phi) / phi))
        self.theta[1:] = inv_sigma.dot(mu_1 - mu_0)

        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE
