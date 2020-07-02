import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import time
from utils.split import uniform_split, exponential_split, poisson_split, failure_split, partition_split
from utils.calculator import ground_truth, global_probability
# pd.set_option('display.max_rows', None)


'''
Global algorithm for replica currency estimation
'''


def global_alg(splitmode, random_seed, filepath, training_size, test_size, test_startpoint, lamb,
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

    result = pd.DataFrame()

    bounds = []
    start = time.time()
    for test_time in test_times:
        training_set = df.iloc[:, 0][df.iloc[:, 0].iloc[:] < test_time[0]].iloc[-training_size:]
        model = LinearRegression()
        y = training_set.diff().dropna().values.tolist()[1:]
        X = training_set.diff().dropna().values.tolist()[:-1]
        X = [[int(X[i])] for i in range(0, len(X))]
        model.fit(X, y)
        y_pred = model.predict(X)
        bounds.append(np.max(abs(y - y_pred)))
    bound = np.max(bounds)
    print('max bound:', bound)

    for current_time, test_time in zip(current_times, test_times):
        # print(current_time, test_time)
        T_n = current_time[1] - current_time[0]
        truth = ground_truth(test_time, current_time)
        pred, prob, x = global_probability(T_n, test_time[0], int(current_time[0]), df.iloc[:, 0], training_size, bound)
        result = result.append(pd.DataFrame([pred, truth, prob, x]).T)
    end = time.time()
    print('Execution time: ', end - start)

    result.columns = ['Prediction', 'Ground truth', 'Probability', 'x']
    result.index = range(result.shape[0])
    # print(result)
    return result
