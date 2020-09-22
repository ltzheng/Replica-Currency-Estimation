import pandas as pd
import numpy as np
import random
import math


'''
Allocate nodes for each update in different scenarios
'''


# uniform allocation, i.e., scenario 1
def uniform_alloc(data, random_seed, available_num):
    random.seed(random_seed)
    nodes = []
    for i in range(data.shape[0]):
        nodes.append("".join(random.sample(['a', 'b', 'c', 'd', 'e', 'f'], available_num)))  # available nodes per update
    return pd.DataFrame(nodes)


# exponential distribution
def exponential_alloc(data, lamb, random_seed):
    random.seed(random_seed)
    p1, p2, p3, p4, p5, p6 = lamb * math.exp(-lamb * 1), lamb * math.exp(-lamb * 2), lamb * math.exp(
        -lamb * 3), lamb * math.exp(-lamb * 4), lamb * math.exp(-lamb * 5), lamb * math.exp(-lamb * 6)
    p_sum = p1 + p2 + p3 + p4 + p5 + p6
    p1, p2, p3, p4, p5, p6 = p1 / p_sum, p2 / p_sum, p3 / p_sum, p4 / p_sum, p5 / p_sum, p6 / p_sum
    nodes = []
    probs = [p1, p2, p3, p4, p5, p6]
    for i in range(data.shape[0]):
        nodes.append("".join(np.random.choice(['a', 'b', 'c', 'd', 'e', 'f'], 3, p=probs, replace=False).tolist()))

    return pd.DataFrame(nodes)


# poisson distribution
def poi(k, lamb):
    return (lamb ** k) * math.exp(-lamb) / math.factorial(k)


def poisson_alloc(data, lamb, random_seed):
    random.seed(random_seed)
    p1, p2, p3, p4, p5 = poi(1, lamb), poi(2, lamb), poi(3, lamb), poi(4, lamb), poi(5, lamb)
    p_sum = p1 + p2 + p3 + p4 + p5
    p1, p2, p3, p4, p5 = p1 / p_sum, p2 / p_sum, p3 / p_sum, p4 / p_sum, p5 / p_sum
    nodes = []
    for i in range(data.shape[0]):
        tempnodes = random.sample(['a', 'b', 'c', 'd', 'e', 'f'], 5)
        rand = random.random()
        if rand < p1:
            num = 1
        elif rand < p1 + p2:
            num = 2
        elif rand < p1 + p2 + p3:
            num = 3
        elif rand < p1 + p2 + p3 + p4:
            num = 4
        else:
            num = 5
        tempnodes = random.sample(tempnodes, num)
        nodes.append("".join(tempnodes))

    return pd.DataFrame(nodes)


# partition & failure scenario
def available_nodes_alloc(available_nodes, random_seed):
    random.seed(random_seed)
    nodes = []
    for available_node in available_nodes:
        if len(available_node) == 6:
            tempnodes = random.sample(available_node, 3)
        else:
            tempnodes = random.sample(available_node, 1)
        nodes.append("".join(tempnodes))
    return pd.DataFrame(nodes)


# for **uniform, poisson, exponential** to generate current_times and test_times
def test_set_gen(test_set, test_size, random_seed):
    random.seed(random_seed)
    current_times = []
    test_times = []
    for i in range(test_set.shape[0] - 1):
        current_times.append([test_set.iloc[i, 0], test_set.iloc[i + 1, 0], test_set.iloc[i, 1]])

    for current_time in current_times:
        test_time = random.sample(range(int(current_time[0]), int(current_time[1])), 1)[0]
        test_times.append(test_time)
        if len(test_times) >= test_size:
            break

    return pd.DataFrame(current_times[0:test_size]), pd.DataFrame(test_times[0:test_size])


# sample times for scenarios
def sample_scenario_period(update_times, scenario_num, length):
    scenario_periods = []
    total_times = list(range(int(update_times[0]), int(update_times[len(update_times) - 1])))
    for i in range(scenario_num):
        # print('unavailable_times:', len(unavailable_times))
        scenario_time = random.sample(total_times, 1)
        for timepoint in scenario_time:
            for time in range(timepoint - length, timepoint + length):
                if time in total_times:
                    total_times.remove(time)
                scenario_periods.append(time)

    return scenario_periods


# frequency changing scenario
def frequency_test_set_gen(random_seed, test_set, test_size, scenario_num, length, frequency_normal, frequency_low):
    random.seed(random_seed)
    current_times = []
    test_times = []
    frequency_bit = 0

    for i in range(test_set.shape[0] - 1):
        if frequency_bit % 3 == 0: # high frequency
            current_times.append([test_set.iloc[i, 0], test_set.iloc[i + 1, 0], test_set.iloc[i, 1]])
        elif frequency_bit % 3 == 1: # normal scenario
            if i % frequency_normal == 0 or i % frequency_bit == int(frequency_normal / 2):
                current_times.append([test_set.iloc[i, 0], test_set.iloc[i + 1, 0], test_set.iloc[i, 1]])
        else: # low frequency
            if i % frequency_low == 0 or i % frequency_bit == int(frequency_low / 2):
                current_times.append([test_set.iloc[i, 0], test_set.iloc[i + 1, 0], test_set.iloc[i, 1]])
        if i % length == 0:
            frequency_bit += 1
        if (frequency_bit * 2 / 3) >= scenario_num:
            frequency_bit = 1

    for current_time in current_times:
        test_time = random.sample(range(int(current_time[0]), int(current_time[1])), 1)[0]
        test_times.append(test_time)
        if len(test_times) >= test_size:
            break
    test_times = pd.DataFrame(test_times)
    # print('len(test_times):', len(test_times))
    test_times = pd.concat([test_times, uniform_alloc(test_times, random_seed, 3)], axis=1).dropna()
    return pd.DataFrame(current_times[0:test_size]), pd.DataFrame(test_times[0:test_size])


# for node failure scenario to generate available nodes for each update
def failure_nodes_gen(random_seed, fail_num, length, update_times):
    random.seed(random_seed)
    available_nodes_list = ['a', 'b', 'c', 'd', 'e', 'f']
    unavailable_times = sample_scenario_period(update_times, fail_num, length)

    # select which 3 nodes to disable for each failure
    nodes = []
    available_nodes = []
    nodecount = 0
    count = 0
    for i in range(fail_num):
        random.seed(random_seed)
        nodes.append(random.sample(available_nodes_list, 3))
    for update_time in update_times:
        if int(update_time) in unavailable_times:
            count += 1
            available_nodes.append(nodes[nodecount])
            if count % length == 0:
                nodecount += 1
        else:
            available_nodes.append(available_nodes_list)
    return available_nodes


# for network partition scenario to generate available nodes for each update
def partition_nodes_gen(random_seed, scenario_num, length, update_times):
    random.seed(random_seed)
    available_nodes_list = ['a', 'b', 'c', 'd', 'e', 'f']

    unavailable_times = sample_scenario_period(update_times, scenario_num, length)

    # 3 nodes to write and other 3 to read
    read_nodes = []
    write_nodes = []
    for i in range(scenario_num):
        random.seed(random_seed)
        nodes_to_read = random.sample(available_nodes_list, 3)
        read_nodes.append(nodes_to_read)
        write_node = []
        for node in available_nodes_list:
            if node not in nodes_to_read:
                write_node.append(node)
        write_nodes.append(write_node)

    available_nodes = []
    rnodecount = 0
    wnodecount = 0
    rcount = 0
    wcount = 0
    i = 0
    for update_time in update_times:
        available_node = []
        if i % 2 == 0:
            if int(update_time) in unavailable_times:
                rcount += 1
                available_nodes.append(read_nodes[rnodecount])
                if rcount % length == 0:
                    rnodecount += 1
            else:
                for node in available_nodes_list:
                    available_node.append(node)
                available_nodes.append(available_node)
        else:
            if int(update_time) in unavailable_times:
                wcount += 1
                available_nodes.append(write_nodes[wnodecount])
                if wcount % length == 0:
                    wnodecount += 1
            else:
                for node in available_nodes_list:
                    available_node.append(node)
                available_nodes.append(available_node)
        i += 1

    return available_nodes


# for network failure & node partition scenario to generate current_times and test_times
def failure_partition_test_set_gen(splitmode, test_set, test_size, random_seed, scenario_num, length):
    random.seed(random_seed)
    current_times = []
    test_times = []
    # generate current_times and sample test_times
    for i in range(test_set.shape[0] - 1):
        current_times.append([test_set.iloc[i, 0], test_set.iloc[i + 1, 0]])
    for current_time in current_times:
        # print(int(current_time[0]), int(current_time[1]))
        test_time = random.sample(range(int(current_time[0]), int(current_time[1])), 1)[0]
        test_times.append(test_time)
    # merge current_times and test_times to apply scenarios
    current_times = pd.DataFrame(current_times[0:test_size])
    current_times.columns = ['time', 'time2']
    test_times = pd.DataFrame(test_times[0:test_size])
    test_times.columns = ['time']
    current_times['type'] = 'current'
    test_times['type'] = 'test'
    df = pd.concat([current_times.loc[:, ['time', 'type']], test_times], axis=0).sort_values(by='time', ascending=True)
    df.index = list(range(df.shape[0]))

    if splitmode == 'failure':
        available_nodes = failure_nodes_gen(random_seed, scenario_num, length, df.iloc[:, 0].values.tolist())
    elif splitmode == 'partition':
        available_nodes = partition_nodes_gen(random_seed, scenario_num, length, df.iloc[:, 0].values.tolist())
    else:
        raise NotImplementedError

    df = pd.concat([df, available_nodes_alloc(available_nodes, random_seed)], axis=1)
    df.columns = ['time', 'type', 'nodes']
    test_times = df[df['type'] == 'test'].loc[:, ['time', 'nodes']]
    alloc_nodes = df[df['type'] == 'current']
    alloc_nodes.index = list(range(alloc_nodes.shape[0]))
    current_times = pd.concat([current_times.loc[:, ['time', 'time2']], alloc_nodes['nodes']], axis=1)
    current_times.index = list(range(current_times.shape[0]))
    test_times.index = list(range(test_times.shape[0]))

    return pd.DataFrame(current_times[0:test_size]), pd.DataFrame(test_times[0:test_size])