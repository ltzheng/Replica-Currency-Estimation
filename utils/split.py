import pandas as pd
import numpy as np
import random
from utils.allocator import uniform_alloc, test_set_gen, exponential_alloc, poisson_alloc, \
    failure_partition_test_set_gen, frequency_test_set_gen
# pd.set_option('display.max_rows', None)


'''
Split test set
'''


def uniform_split(random_seed, filepath, test_size, test_startpoint, available_num=3):
    random.seed(random_seed)
    df = pd.read_csv(filepath, header=None)
    df = df.iloc[:, 0]
    df = pd.concat([df, uniform_alloc(df, random_seed, available_num)], axis=1).dropna()
    df.columns = ['Time', 'Nodes']
    df.index = range(len(df))
    test_set = df.iloc[test_startpoint:]
    test_set.index = range(len(test_set))
    current_times, test_times = test_set_gen(test_set, test_size, random_seed)
    test_times = pd.concat([test_times, uniform_alloc(test_times, random_seed, available_num)], axis=1).dropna()
    current_times = np.array(current_times)
    test_times = np.array(test_times)

    return df, current_times, test_times


def exponential_split(random_seed, filepath, test_size, test_startpoint, lamb):
    random.seed(random_seed)
    df = pd.read_csv(filepath, header=None)
    df = df.iloc[:, 0]
    df = pd.concat([df, exponential_alloc(df, lamb, random_seed)], axis=1).dropna()
    df.columns = ['Time', 'Nodes']
    df.index = range(len(df))
    test_set = df.iloc[test_startpoint:]
    test_set.index = range(len(test_set))
    current_times, test_times = test_set_gen(test_set, test_size, random_seed)
    test_times = pd.concat([test_times, exponential_alloc(test_times, lamb, random_seed)], axis=1).dropna()
    current_times = np.array(current_times)
    test_times = np.array(test_times)

    return df, current_times, test_times


def poisson_split(random_seed, filepath, test_size, test_startpoint, lamb):
    random.seed(random_seed)
    df = pd.read_csv(filepath, header=None)
    df = df.iloc[:, 0]
    df = pd.concat([df, poisson_alloc(df, lamb, random_seed)], axis=1).dropna()
    df.columns = ['Time', 'Nodes']
    df.index = range(len(df))
    test_set = df.iloc[test_startpoint:]
    test_set.index = range(len(test_set))
    current_times, test_times = test_set_gen(test_set, test_size, random_seed)
    test_times = pd.concat([test_times, poisson_alloc(test_times, lamb, random_seed)], axis=1).dropna()
    current_times = np.array(current_times)
    test_times = np.array(test_times)

    return df, current_times, test_times


def failure_split(random_seed, filepath, test_size, test_startpoint, length, test_fail_num):
    random.seed(random_seed)
    df = pd.read_csv(filepath, header=None)
    df = df.iloc[:, 0]
    df = pd.concat([df, uniform_alloc(df, random_seed, 3)], axis=1).dropna()
    df.columns = ['Time', 'Nodes']
    df.index = range(len(df))

    test_set = df.iloc[test_startpoint:]
    test_set.index = range(len(test_set))
    current_times, test_times = failure_partition_test_set_gen('failure', test_set, test_size, random_seed, test_fail_num, length)
    current_times = np.array(current_times)
    test_times = np.array(test_times)

    return df, current_times, test_times


def partition_split(random_seed, filepath, test_size, test_startpoint, length, test_fail_num):
    random.seed(random_seed)
    df = pd.read_csv(filepath, header=None)
    df = df.iloc[:, 0]
    df = pd.concat([df, uniform_alloc(df, random_seed, 3)], axis=1).dropna()
    df.columns = ['Time', 'Nodes']
    df.index = range(len(df))

    test_set = df.iloc[test_startpoint:]
    test_set.index = range(len(test_set))
    current_times, test_times = failure_partition_test_set_gen('partition', test_set, test_size, random_seed, test_fail_num, length)
    current_times = np.array(current_times)
    test_times = np.array(test_times)

    return df, current_times, test_times


def frequency_split(random_seed, filepath, test_size, test_startpoint, length, scenario_num, frequency_normal, frequency_low):
    random.seed(random_seed)
    df = pd.read_csv(filepath, header=None)
    df = df.iloc[:, 0]
    df = pd.concat([df, uniform_alloc(df, random_seed, 3)], axis=1).dropna()
    df.columns = ['Time', 'Nodes']
    df.index = range(len(df))

    test_set = df.iloc[test_startpoint:]
    test_set.index = range(len(test_set))
    current_times, test_times = frequency_test_set_gen(random_seed, test_set, test_size, scenario_num,
                                                       length, frequency_normal, frequency_low)
    current_times = np.array(current_times)
    test_times = np.array(test_times)

    return df, current_times, test_times
