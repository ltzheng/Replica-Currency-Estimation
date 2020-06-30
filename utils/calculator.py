import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
import math


'''
Methods for calculating
'''


def calculate_gap_sum(n, phi1, phi0, t_p_1):
    if n <= 1:
        return 0
    gap_sum = 0
    for j in range(2, n + 1):
        temp = 0
        for i in range(0, j - 1):
            temp += pow(phi1, i)
        gap_sum += pow(phi1, j - 1) * t_p_1 + phi0 * temp
    return int(gap_sum)


def ground_truth(test_time, current_time):
    # test_time in format: [timestamp, 'nodes']
    # current_time in format: [timestamp1, time2, nodes of time1]
    nodes = current_time[2]
    for node in test_time[1]:
        if node in nodes:
            return True
    return False


def f1score(df):
    df = df.iloc[:, 0:2]
    df.columns = ['Prediction', 'Truth']
    truth = []
    pred = []
    for index, row in df.iterrows():
        if row['Prediction']:
            pred.append(1)
        else:
            pred.append(0)
        if row['Truth']:
            truth.append(1)
        else:
            truth.append(0)
    return f1_score(truth, pred)


def global_probability(T_n, current_time, low, df, train_size, bound):
    training_set = df[df.iloc[:] < current_time].iloc[-train_size:]
    model = LinearRegression()
    y = training_set.diff().dropna().values.tolist()[1:]
    X = training_set.diff().dropna().values.tolist()[:-1]
    X = [[int(X[i])] for i in range(0, len(X))]
    try:
        model.fit(X, y)
    except ValueError:
        print('Not enough data to train, please increase test_start_point')

    z = len(y)
    y_pred = model.predict(X)
    loss = np.average(abs(y - y_pred))
    R_Z_hat, l = loss, bound
    T_nplus1_prime = int(model.coef_ * T_n + model.intercept_)

    if T_nplus1_prime > current_time - low:
        # t_n is current
        exp = ((2 * z * pow(T_nplus1_prime - current_time + low - R_Z_hat -
                            l * math.sqrt(4 * math.log(math.exp(1) * z / 2) / z), 2)) / (l ** 2))
        delta = math.exp(-exp)
        return True, 1 - delta, exp
    else:
        # t_n is stale
        exp = ((2 * z * pow(- T_nplus1_prime + current_time - low - R_Z_hat -
                            l * math.sqrt(4 * math.log(math.exp(1) * z / 2) / z), 2)) / (l ** 2))
        delta = math.exp(-exp)
        return False, 1 - delta, exp


def local_predict(current_time, T_p_1, t_p_1, T_a_1, phi1, phi0):
    T_p_n_prime_list = []
    n_list = []
    T_p_n_prime_temp = T_p_1
    T_p_nminus1_prime_temp = T_p_1
    n = 1

    while T_p_n_prime_temp <= T_a_1:
        gap = calculate_gap_sum(n, phi1, phi0, t_p_1)
        T_p_n_prime_temp = T_p_1 + gap
        # print(T_p_n_prime_temp, T_a_1)
        # print(T_p_n_prime_temp, T_a_1, calculate_gap_sum(n, phi1, phi0, t_p_1))
        if gap < -1e6:
            raise TimeoutError('An error occurred: phi0 is too small, please increase back_length')
        T_p_nminus1_prime_temp = T_p_1 + calculate_gap_sum(n - 1, phi1, phi0, t_p_1)
        n = n + 1

    pred = T_p_n_prime_temp >= current_time

    if pred:
        return pred, T_p_n_prime_temp, T_p_nminus1_prime_temp, n
    else:
        # T_a_1 < T_p_n_prime_temp < current_time
        while T_p_n_prime_temp <= current_time:
            T_p_n_prime_list.append(int(T_p_n_prime_temp))
            n_list.append(n)
            gap = calculate_gap_sum(n, phi1, phi0, t_p_1)
            T_p_n_prime_temp = T_p_1 + gap
            # print('T_p_n_prime_temp:', T_p_n_prime_temp)
            # print('phi1:', phi1)
            # print('phi0:', phi0)
            # print('gap:', calculate_gap_sum(n, phi1, phi0, t_p_1))
            if gap < -1e6:
                raise TimeoutError('An error occurred: phi0 is too small, please increase back_length')
            T_p_nminus1_prime_temp = T_p_1 + calculate_gap_sum(n - 1, phi1, phi0, t_p_1)
            n = n + 1

        return pred, T_p_n_prime_list, T_p_nminus1_prime_temp, n_list


def local_current_probability(n, z, T_a_1, T_p_n_prime, T_p_nminus1_prime, current_time, phi1, loss, bound):
    R_Z_hat, l = loss, bound
    temp1 = 0
    temp2 = 0
    prob = 1

    for j in range(2, n + 1):
        for i in range(0, j - 1):
            temp1 = temp1 + phi1 ** i
    for j in range(2, n):
        for i in range(0, j - 1):
            temp2 = temp2 + phi1 ** i

    delta_zeta = math.exp(- ((2 * z * pow((T_p_n_prime - current_time) / temp1 - R_Z_hat -
                                l * math.sqrt(4 * math.log(math.exp(1) * z / 2) / z), 2))
                            / (l ** 2)))
    delta_nminus1 = math.exp(- ((2 * z * pow((T_a_1 - T_p_nminus1_prime) / temp2 - R_Z_hat -
                                l * math.sqrt(4 * math.log(math.exp(1) * z / 2) / z), 2))
                            / (l ** 2)))

    for j in range(1, n):
        prob = prob * (1 - delta_zeta) ** j
    for j in range(1, n - 1):
        prob = prob * (1 - delta_nminus1) ** j

    return prob


def local_stale_probability(n, z, T_a_1, T_p_n_prime, T_p_nminus1_prime, current_time, phi1, loss, bound):
    R_Z_hat, l = loss, bound
    temp1 = 0
    prob = 1
    prob_list = []
    product = 1
    for val_n, val in zip(n, T_p_n_prime):
        for j in range(2, val_n + 1):
            for i in range(0, j - 1):
                temp1 = temp1 + phi1 ** i

        delta_zeta = math.exp(- ((2 * z * pow((current_time - val) / temp1 - R_Z_hat -
                                              l * math.sqrt(4 * math.log(math.exp(1) * z / 2) / z), 2)) / (l ** 2)))
        delta_n = math.exp(- ((2 * z * pow((val - T_a_1) / temp1 - R_Z_hat -
                                           l * math.sqrt(4 * math.log(math.exp(1) * z / 2) / z), 2)) / (l ** 2)))

        for j in range(1, val_n):
            prob = prob * (1 - delta_zeta) ** j
        for j in range(1, val_n):
            prob = prob * (1 - delta_n) ** j

        prob_list.append(prob)
    for val in prob_list:
        product = product * (1 - val)
    prob = 1 - product

    return prob