'''
Created on June 27, 2020
McMaster University research internship
@author: Longtao Zheng (zlt0116@mail.ustc.edu.cn)
'''
import numpy as np
import pandas as pd
from algorithms.global_alg import global_alg
from algorithms.local_alg import local_alg
from utils.calculator import f1score


################## Hyperparameter Setting #################

# splitmode = 'multi'
# set for multi scenario
available_num = 3

# splitmode = 'uniform'
# splitmode = 'exponential'
# splitmode = 'poisson'
# splitmode = 'failure'
# splitmode = 'partition'

repeat = 5
total_test_times = 5
random_seeds = range(0, total_test_times * repeat, repeat)

# test_startpoint should be greater than training_size
training_size = 15
test_size = 500
test_startpoint = 100

# select which algorithm to test
# algorithm = 'global'
algorithm = 'local'

# only set for exponential & poisson distribution scenarios
lamb = 1

# only set for node partition & network failure scenarios
scenario_length, train_fail_num, test_fail_num = 100, 50, 50

# select dataset
# dataset = './dataset/btcusd.csv'
dataset = './dataset/LondonBike.csv'
# dataset = './dataset/StockData.csv'
# dataset = './dataset/RedditMachineLearning.csv'
# dataset = './dataset/JoeBidenTweets.csv'
# dataset = './dataset/911.csv'
# dataset = './dataset/IndianCOVID19.csv'
# dataset = './dataset/MentalHealth.csv'
# dataset = './dataset/BankTransaction.csv'
# dataset = './dataset/sensor_same_deleted.csv'

separated_by = 'value'  # by greater than given value
# set separation of x, you can choose arbitrarily, >0 means selecting all
separations = [0, 35, 39, 43, 45]  # btcusd
# separations = [0, 10, 15, 16, 17, 18, 19, 20, 25, 30, 35]  # Reddit
# separations = [0, 15, 17, 18, 19]  # StockData
# separations = [0, 11, 13, 15, 16, 17, 18, 19, 21, 25]  # LondonBike & Reddit global

# separated_by = 'proportion'
# separations = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

scores = np.zeros(len(separations))
nums = np.zeros(len(separations))
confidence = np.zeros(len(separations))

for random_seed in random_seeds:
    print('******************')
    print('random_seed:', random_seed)

    if algorithm == 'global':
        df = global_alg(splitmode, random_seed, dataset, back_length, test_size, test_startpoint, lamb,
                        scenario_length, train_fail_num, test_fail_num, available_num)
        df.to_csv('./results/global_result.csv', index=False, header=False)

    elif algorithm == 'local':
        df = local_alg(splitmode, random_seed, dataset, back_length, test_size, test_startpoint, lamb,
                       scenario_length, train_fail_num, test_fail_num)
        df.to_csv('./results/local_result.csv', index=False, header=False)

    else:
        raise NotImplementedError

    df = df.sort_values(by='x', ascending=True)
    df.index = range(df.shape[0])

    dfs = []
    if separated_by == 'value':
        for sep in separations:
            dfs.append((df[df['x'] > sep]))
    elif separated_by == 'proportion':
        i = 0
        step = 0
        for sep in separations:
            dfs.append((df.iloc[int(df.shape[0] * sep):int(df.shape[0] * 0.7)]))
            confidence[i] = df.loc[int(df.shape[0] * sep), 'x']
            # dfs.append((df.iloc[step:int(df.shape[0] * sep)]))
            # step = int(df.shape[0] * sep)
            # print(step)
            # confidence[i] = df.loc[step - 1, 'x']
            i += 1
    else:
        raise NotImplementedError

    i = 0
    for df in dfs:
        df.index = range(df.shape[0])
        scores[i] += f1score(df)
        nums[i] += df.shape[0]
        i += 1

print('******************')
for (score, sep) in zip(scores, separations):
    print('Average f1-score for x >', sep, '= {:.4f}'.format(score / total_test_times))

nums = nums / total_test_times
print('Average tuplenums:', nums)
pd.DataFrame(nums).to_csv('./results/tuplenum.csv', index=False, header=False)
