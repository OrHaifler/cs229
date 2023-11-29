import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    #Initialize and fit a new model to the training data
    model = LogisticRegression()
    model.fit(x_train, y_train)

    #Plot the decision boundary
    util.plot(x_train, y_train, model.theta, 'save/p01b')
    print(model.theta.shape)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    pred = model.predict(x_eval)
    np.savetxt(pred_path, pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)
        h = lambda X: 1 / (1 + np.exp(-X.dot(self.theta)))
        iter = 0
        prev_theta = 0
        iter = 0
        while iter <= self.max_iter:
            g = x.T.dot(h(x) - y) / m
            H = (x.T * h(x) * (1 - h(x))).dot(x) / m
            self.theta -= self.step_size * np.linalg.inv(H).dot(g)
            if np.linalg.norm((self.theta - prev_theta), ord=1) < self.eps:
                print('Finished!')
                break
            if self.verbose:
                print(f'Iteration: {iter}')
            iter += 1


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
        # *** END CODE HERE ***
