import numpy as np
import pandas as pd
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    x_train, _ = util.load_dataset(train_path, add_intercept=True)
    train_csv = pd.read_csv(train_path)
    y_train = np.array(train_csv['t'])
    model = LogisticRegression()
    model.fit(x_train, y_train)

    util.plot(x_train, y_train, model.theta, 'save/p02c_train')

    x_test, _ = util.load_dataset(test_path, add_intercept=True)
    test_csv = pd.read_csv(test_path)
    y_test = np.array(test_csv['t'])
    util.plot(x_test, y_test, model.theta, 'save/p02c_test')
    pred = model.predict(x_test)
    np.savetxt(pred_path_c, pred > 0.5, fmt='%d')
    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels
    y_train = np.array(train_csv['y'])
    model.fit(x_train, y_train)
    util.plot(x_train, y_train, model.theta, 'save/p02d_train')
    util.plot(x_test, y_test, model.theta, 'save/p02d_test')

    pred = model.predict(x_test)
    np.savetxt(pred_path_d, pred > 0.5, fmt='%d')
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    x_val, labels = util.load_dataset(val_path, add_intercept=True)
    x_val = x_val[labels == 1]
    val_csv = pd.read_csv(val_path)
    y_val = np.array(val_csv['t'])
    Vp = x_val.shape[0]
    p_pred = model.predict(x_val)
    alpha = np.sum(p_pred) / Vp
    

    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
