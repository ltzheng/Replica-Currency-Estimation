import numpy as np
import pandas as pd
import argparse
from algorithms.global_alg import global_alg
from algorithms.local_alg import local_alg
from utils.calculator import f1score


def average(algorithm, X_size, scenario_length, test_fail_num, test_size, train_fail_num, splitmode, available_num,
            gap, total_test_times, back_length, test_startpoint, lamb, frequency_normal, frequency_low, dataset,
            separated_by, separations, log_nums):
    random_seeds = range(0, total_test_times * gap, gap)

    # separations = [0, 35, 39, 43, 45]  # btcusd
    # separations = [0, 10, 15, 16, 17, 18, 19, 20, 25, 30, 35]  # Reddit
    # separations = [0, 15, 17, 18, 19]  # StockData
    # separations = [0, 11, 13, 15, 16, 17, 18, 19, 21, 25]  # LondonBike & Reddit global

    scores = np.zeros(len(separations))
    nums = np.zeros(len(separations))
    confidence = np.zeros(len(separations))

    for random_seed in random_seeds:
        print('******************')
        print('random_seed:', random_seed)

        if algorithm == 'global':
            df = global_alg(splitmode, random_seed, dataset, back_length, test_size, test_startpoint, lamb,
                            scenario_length, train_fail_num, test_fail_num, available_num, frequency_normal,
                            frequency_low, X_size)
            df.to_csv('./results/global_result.csv', index=False, header=False)

        elif algorithm == 'local':
            df = local_alg(splitmode, random_seed, dataset, back_length, test_size, test_startpoint, lamb,
                           scenario_length, train_fail_num, test_fail_num, frequency_normal, frequency_low, X_size)
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
            for i, sep in enumerate(separations):
                dfs.append((df.iloc[int(df.shape[0] * sep):int(df.shape[0] * 0.7)]))
                confidence[i] = df.loc[int(df.shape[0] * sep), 'x']
        else:
            raise NotImplementedError

        for i, df in enumerate(dfs):
            df.index = range(df.shape[0])
            scores[i] += f1score(df)
            nums[i] += df.shape[0]

    scores = scores / total_test_times
    for (score, sep, conf) in zip(scores, separations, confidence):
        if separated_by == 'value':
            print('Average f1-score for x >', sep, '= {:.4f}'.format(score))
        elif separated_by == 'proportion':
            print('Average f1-score'.format(sep), '= {:.4f}'.format(score), '; confidence =', conf)

    if log_nums:
        nums = nums / total_test_times
        print('Average tuplenums:', nums)
        pd.DataFrame(nums).to_csv('./results/tuplenum.csv', index=False, header=False)

    return scores[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='global', help='global/local')
    parser.add_argument('--X-size', type=int, default=3, help='')
    parser.add_argument('--scenario-length', type=int, default=0)
    parser.add_argument('--test-fail-num', type=int, default=0,
                        help='only set for node partition & network failure scenarios')
    parser.add_argument('--test-size', type=int, default=100)
    parser.add_argument('--train-fail-num', type=int, default=0,
                        help='only set for node partition & network failure scenarios')
    parser.add_argument('--split-mode', type=str, default='uniform',
                        help='multi/uniform/exponential/poisson/failure/partition/frequency')
    parser.add_argument('--available-num', type=int, default=3)
    parser.add_argument('--seed-gap', type=int, default=5)
    parser.add_argument('--total-test-times', type=int, default=5)
    parser.add_argument('--back-length', type=int, default=50,
                        help='test_startpoint should be greater than back_length')
    parser.add_argument('--test-startpoint', type=int, default=500)
    parser.add_argument('--poisson-lambda', type=int, default=0.4,
                        help='only set for exponential & poisson distribution scenarios')
    parser.add_argument('--frequency-normal', type=int, default=5,
                        help='only set for scenario that changes frequency')
    parser.add_argument('--frequency-low', type=int, default=9,
                        help='only set for scenario that changes frequency')
    parser.add_argument('--dataset', type=str, default='./dataset/btcusd.csv',
                        help='the csv file in dataset directory: '
                             'btcusd/LondonBike/StockData/RedditMachineLearning/JoeBidenTweets/911/MentalHealth'
                             '/BankTransaction, etc.')
    parser.add_argument('--separated-by', type=str, default='value',
                        help='value/proportion')
    parser.add_argument('--separations', nargs='+', type=int, default=[0, 35, 39, 43, 45],
                        help='separated by value: set separation according values in arg separation, '
                             '0 means selecting all, e.g., 0, 35, 39, 43, 45 for btcusd'
                             'separated by proportion: set separation according to proportion in arg separation, '
                             'e.g., 0, 0.1, 0.2, 0.3, 0.4, 0.5')
    parser.add_argument('--log-nums', action='store_true')

    args = parser.parse_args()

    average(algorithm=args.algorithm, X_size=args.X_size, scenario_length=args.scenario_length,
            test_fail_num=args.test_fail_num, test_size=args.test_size, train_fail_num=args.train_fail_num,
            splitmode=args.split_mode, available_num=args.available_num, gap=args.seed_gap,
            total_test_times=args.total_test_times, back_length=args.back_length, test_startpoint=args.test_startpoint,
            lamb=args.poisson_lambda, frequency_normal=args.frequency_normal, frequency_low=args.frequency_low,
            dataset=args.dataset, separated_by=args.separated_by, separations=args.separations, log_nums=args.log_nums)
