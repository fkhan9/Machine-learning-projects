import os
import numpy as np
import pandas as pd
import csv

with open('out.csv', 'w') as csvfile:
    fieldnames = ['Iteration number', 'weight','weight','weight','Squared_Error']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
def get_training_data(path):    # path to read data from
    raw_data = pd.read_csv(path, header=None)

    raw_data.insert(0, 'Ones', 1)

    num_columns = raw_data.shape[1]                       # Get number of columns for slicing
    input_X = raw_data.iloc[:, 0:num_columns-1]            # [ slice_of_rows, slice_of_columns ]
    input_y = raw_data.iloc[:, num_columns-1:num_columns]  # [ slice_of_rows, slice_of_columns ]

    X = np.matrix(input_X.values)   # converting data-frame to matrix
    y = np.matrix(input_y.values)   # converting data-frame to matrix

    return X, y


def compute_mean_square_error(X, y, weight):
    summation = np.power(X * weight.T - y, 2)
    return np.sum(summation)


def compute_gradient_descent(X, y, learning_rate, num_iterations):
    num_parameters = X.shape[1]
    print(num_parameters)# dimension of weight vector
    weights = np.matrix([0.0 for i in range(num_parameters)])   # initialize weight vector
    _gradient = [0.0 for i in range(num_iterations)]             # initialize gradient vector

    for it in range(num_iterations):
        error = np.repeat((X * weights.T) - y, num_parameters, axis=1)
        error_derivative = np.sum(np.multiply(error, X), axis=0)
        weights = weights - (learning_rate) * error_derivative
        _gradient[it] = compute_mean_square_error(X, y, weights)
        if round((_gradient[it-1] - _gradient[it]), 4) == 0.0001:

            print("iteration_number:", it)
            print('weights_vector:', weights)
            print("sum_of squared_errors:", _gradient[it])
            print("Delta_Threshold:", _gradient[it - 1] - _gradient[it])
            break

        print('iteration_number', it)
        print('weights_vector', weights)
        print('sum_of_squared_error', _gradient[it])
        print("Delta", _gradient[it-1] - _gradient[it])

     return weights, _gradient


if __name__ == '__main__':
    print("please select data-set \n 1:Random.csv \n 2:Yacht.csv")
    sel_data_set = int(input())
    if sel_data_set == 1:
        X, y = get_training_data(os.getcwd() + '/random.csv')
        weights, gradient = compute_gradient_descent(X, y, 0.0001, 100)
    elif sel_data_set == 2:
        X, y = get_training_data(os.getcwd() + '/yacht.csv')
        weights, gradient = compute_gradient_descent(X, y, 0.0001, 38000)
