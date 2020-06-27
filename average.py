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

splitmode = 'uniform'
# splitmode = 'exponential'
# splitmode = 'poisson'
# splitmode = 'failure'
# splitmode = 'partition'

repeat = 5
total_test_times = 5
random_seeds = range(0, total_test_times * repeat, repeat)

# test_startpoint should be greater than training_size
training_size = 10
test_size = 500
test_startpoint = 50

# select which algorithm to test
# algorithm = 'global'
algorithm = 'local'

# only set for exponential & poisson distribution scenarios
lamb = 1

# only set for node partition & network failure scenarios
scenario_length, train_fail_num, test_fail_num = 100, 50, 50

# select dataset
# dataset = './dataset/btcusd.csv'
# dataset = './dataset/LondonBike.csv'
# dataset = './dataset/StockData.csv'
dataset = './dataset/RedditMachineLearning.csv'
# dataset = './dataset/JoeBidenTweets.csv'
# dataset = './dataset/911.csv'
# dataset = './dataset/IndianCOVID19.csv'
# dataset = './dataset/MentalHealth.csv'
# dataset = './dataset/BankTransaction.csv'
# dataset = './dataset/sensor_same_deleted.csv'

# set separation of x, you can choose arbitrarily, >0 means selecting all
separations = [0, 1, 3, 5, 7, 9]
scores = np.zeros(len(separations))
nums = np.zeros(len(separations))

for random_seed in random_seeds:
    print('******************')
    print('random_seed:', random_seed)

    if algorithm == 'global':
        df = global_alg(splitmode, random_seed, dataset, training_size, test_size, test_startpoint, lamb,
                        scenario_length, train_fail_num, test_fail_num)
        df.to_csv('./results/global_result.csv', index=False, header=False)

    elif algorithm == 'local':
        df = local_alg('partition', random_seed, dataset, training_size, test_size, test_startpoint, lamb,
                       scenario_length, train_fail_num, test_fail_num)
        df.to_csv('./results/local_result.csv', index=False, header=False)

    else:
        raise NotImplementedError

    df = df.sort_values(by='x', ascending=True)

    dfs = []
    for sep in separations:
        dfs.append((df[df['x'] > sep]))

    i = 0
    for (df, sep) in zip(dfs, separations):
        scores[i] += f1score(df)
        nums[i] += df.shape[0]
        i += 1


print('******************')
for (score, sep) in zip(scores, separations):
    print('Average f1-score for x >', sep, '= {:.4f}'.format(score / total_test_times))

nums = nums / total_test_times
print('Average tuplenums:', nums)
pd.DataFrame(nums).to_csv('./results/tuplenum.csv', index=False, header=False)
