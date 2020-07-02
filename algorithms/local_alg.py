import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import time
from utils.split import uniform_split, exponential_split, poisson_split, failure_split, partition_split
from utils.calculator import ground_truth, local_current_probability, local_stale_probability, local_predict
# pd.set_option('display.max_rows', None)


'''
Local algorithm for replica currency estimation
'''


def train_node(df, current_time, train_size):
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
    bound = np.max(abs(y - y_pred))
    T_p_1 = training_set.iloc[-1]
    t_p_1 = y[-1]

    return model.coef_, model.intercept_, loss, bound, z, int(T_p_1), int(t_p_1)


def local_alg(splitmode, random_seed, filepath, training_size, test_size, test_startpoint, lamb,
               scenario_length, train_fail_num, test_fail_num):

    if splitmode == 'uniform':
        df, current_times, test_times = uniform_split(random_seed, filepath, test_size, test_startpoint)
    elif splitmode == 'exponential':
        df, current_times, test_times = exponential_split(random_seed, filepath, test_size, test_startpoint, lamb)
    elif splitmode == 'poisson':
        df, current_times, test_times = poisson_split(random_seed, filepath, test_size, test_startpoint, lamb)
    elif splitmode == 'failure':
        df, current_times, test_times = failure_split(random_seed, filepath, test_size,
                                                      test_startpoint, scenario_length, train_fail_num, test_fail_num)
    elif splitmode == 'partition':
        df, current_times, test_times = partition_split(random_seed, filepath, test_size,
                                                        test_startpoint, scenario_length, train_fail_num, test_fail_num)
    else:
        raise NotImplementedError

    nodes = ['a', 'b', 'c', 'd', 'e', 'f']
    result = pd.DataFrame()

    param = pd.DataFrame(np.zeros((len(nodes), 7)), columns=['coef', 'intercept', 'loss', 'bound', 'z', 'T_p_1', 't_p_1'])
    param.index = nodes

    bounds = []
    start = time.time()
    for test_time in test_times:
        for node in nodes:
            _, _, _, bound, _, _, _ = train_node(df[df['Nodes'].str.contains(node)].iloc[:, 0], test_time[0], training_size)
            bounds.append(bound)
    # print(bounds)
    bound = np.max(bounds)
    print('max bound:', bound)

    for current_time, test_time in zip(current_times,test_times):
        # print(current_time, test_time)
        for node in nodes:
            param.loc[node, 'coef'], param.loc[node, 'intercept'], param.loc[node, 'loss'], \
            param.loc[node, 'bound'], param.loc[node, 'z'], param.loc[node, 'T_p_1'], param.loc[node, 't_p_1'] \
                = train_node(df[df['Nodes'].str.contains(node)].iloc[:, 0], test_time[0], training_size)
        # print(param)

        preds = pd.DataFrame(np.zeros((len(nodes), 2)), columns=['Prediction', 'Probability'])
        preds.index = nodes

        unavailable_nodes = []
        for node in nodes:
            if node not in list(current_time[2]):
                unavailable_nodes.append(node)

        for index, row in param.loc[unavailable_nodes].iterrows():
            preds.loc[index, 'Prediction'], T_p_n_prime, T_p_nminus1_prime, n = local_predict(
                test_time[0], row['T_p_1'], row['t_p_1'], int(current_time[0]), row['coef'], row['intercept'])
            if preds.loc[index, 'Prediction']:
                preds.loc[index, 'Probability'] = local_current_probability(
                    n, row['z'], int(current_time[0]), int(T_p_n_prime), int(T_p_nminus1_prime),
                    int(test_time[0]), row['coef'], row['loss'], bound)
            else:
                preds.loc[index, 'Probability'] = local_stale_probability(
                    n, row['z'], int(current_time[0]), T_p_n_prime, int(test_time[0]), row['coef'], row['loss'], bound)

        # if any unavailable replica predicts stale, result is stale
        flag = 1
        for index, row in preds.loc[unavailable_nodes].iterrows():
            if not row['Prediction']:
                flag = 0
                break

        # print(test_time[1], current_time[2])

        if flag:  # current
            prob = 1
            for index, row in preds.loc[unavailable_nodes].iterrows():
                # print(row['Prediction'], row['Probability'])
                if row['Prediction']:
                    prob = prob * row['Probability']
            truth = ground_truth(test_time, current_time)
            # print('current probability:', prob)
            if prob != 1:
                x = -math.log(1 - prob)
            else:
                x = 9999
            result = result.append(pd.DataFrame([True, truth, prob, x]).T)
        else:  # stale
            temp = 1
            for index, row in preds.loc[unavailable_nodes].iterrows():
                # print(row['Prediction'], row['Probability'])
                if not row['Prediction']:
                    temp = temp * (1 - row['Probability'])
            prob = 1 - temp
            # print('temp:', temp)
            # print('stale probability:', prob)
            truth = ground_truth(test_time, current_time)
            if prob != 1:
                x = -math.log(1 - prob)
            else:
                x = 9999
            result = result.append(pd.DataFrame([False, truth, prob, x]).T)

    end = time.time()
    print('Execution time: ',end - start)

    result.columns = ['Prediction', 'Probability', 'Ground truth', 'x']
    result.index = range(result.shape[0])
    # print(result)
    return result
