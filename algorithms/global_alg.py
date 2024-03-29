import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import time
from utils.split import uniform_split, exponential_split, poisson_split, failure_split, partition_split, frequency_split
from utils.calculator import ground_truth, global_probability
# pd.set_option('display.max_rows', None)


'''
Global algorithm for replica currency estimation
'''


def global_alg(splitmode, random_seed, filepath, training_size, test_size, test_startpoint, lamb,
               scenario_length, train_fail_num, test_fail_num, available_num, frequency_normal, frequency_low, X_size):

    if splitmode == 'multi':
        df, current_times, test_times = uniform_split(random_seed, filepath, test_size, test_startpoint, available_num)
    elif splitmode == 'uniform':
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
    elif splitmode == 'frequency':
        df, current_times, test_times = frequency_split(random_seed, filepath, test_size, test_startpoint, scenario_length,
                                                        test_fail_num, frequency_normal, frequency_low)
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

    for i in range(X_size, test_size):
        # print('i:', i)
        truth = ground_truth(test_times[i], current_times[i])
        if truth:
            # T_n = current_times[i][0] - current_times[i - 1][0]
            T_n = [[]]
            for j in range(X_size):
                T_n[0].append(current_times[i - j][0] - current_times[i - j - 1][0])
            pred, prob, x = global_probability(T_n, test_times[i][0], int(current_times[i][0]), df.iloc[:, 0], training_size, bound, X_size)
        else:
            T_n = [[]]
            low = 0
            for j in range(X_size):
                availables = current_times[i - j][2]
                k = -1
                flag = 2
                t1 = 0
                t2 = 0
                tempdf = df[df['Time'] < current_times[i - j][0]].values.tolist()
                while flag:
                    for a in availables:
                        if a in tempdf[k][1]:
                            if flag == 2:
                                t2 = tempdf[k][0]
                                flag = 1
                                break
                            else:
                                t1 = tempdf[k][0]
                                flag = 0
                                break
                    k -= 1
                T_n[0].append(t2 - t1)
                if j == 0:
                    low = int(t2)
            pred, prob, x = global_probability(T_n, test_times[i][0], low, df.iloc[:, 0], training_size, bound, X_size)
        result = result.append(pd.DataFrame([pred, truth, prob, x]).T)
        i += 1

    # end = time.time()
    # print('Execution time: ', end - start)

    result.columns = ['Prediction', 'Ground truth', 'Probability', 'x']
    result.index = range(result.shape[0])
    return result
