# Linear Regression CV
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from misc_dataanalysis.get_flydata import get_x_and_y_data
from misc_dataanalysis.standard_data_config import data_config
from utils.utils import get_train_test_split


def perform_CV(n_splits=1):
    """
    Sweep over various values for predict_window_size and ncos and do multiple splits of data with different seeds.
    :param n_splits:
    :return:
    """
    tr_te_values = []
    n_seeds = range(n_splits)
    for _ in range(n_splits):
        for predict_window_size in [5, 10, 15, 30, 45]:
            for ncos in [2, 5, 8, 10, 20]:

                data_config['ncos'] = ncos
                data_config['num_timesteps'] = 150000
                data_config['predict_window_size'] = predict_window_size

                # get fly data with this config
                data = get_x_and_y_data(data_config)
                num_tsessions = data['data_config']['num_sessions']
                print("Total sessions returned:", num_tsessions)

                # number of sessions to use
                num_fitsessions = num_tsessions
                train_data, test_data = get_train_test_split(data, num_fitsessions, seed=n_seeds[_], train_frac=0.7)     # keep same data across many values of a hyperparam
                train_emissions, train_inputs = train_data
                test_emissions, test_inputs = test_data
                X_tr = train_inputs.reshape(-1, train_inputs.shape[-1])
                Y_tr = train_emissions.reshape(-1, train_emissions.shape[-1])
                X_te = test_inputs.reshape(-1, test_inputs.shape[-1])
                Y_te = test_emissions.reshape(-1, test_emissions.shape[-1])

                # fit LR on this data
                lr = LinearRegression(fit_intercept=True)
                lr.fit(X_tr, Y_tr)

                # calculate train and test mse
                Y_tr_pred = lr.predict(X_tr)
                Y_te_pred = lr.predict(X_te)

                d = 0   # first emission
                tr_mse0 = mean_squared_error(Y_tr[:, d], Y_tr_pred[:, d])
                te_mse0 = mean_squared_error(Y_te[:, d], Y_te_pred[:, d])

                d = 1   # second emission
                tr_mse1 = mean_squared_error(Y_tr[:, d], Y_tr_pred[:, d])
                te_mse1 = mean_squared_error(Y_te[:, d], Y_te_pred[:, d])
                print((ncos, _, tr_mse0, te_mse0, tr_mse1, te_mse1))

                # append
                tr_te_values.append((predict_window_size, ncos, _, tr_mse0, te_mse0, tr_mse1, te_mse1))
    joblib.dump(tr_te_values, f'tr_te_values.pkl')
    return


if __name__ == '__main__':
    perform_CV()



