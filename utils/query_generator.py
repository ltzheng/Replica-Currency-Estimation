import pandas as pd
import numpy as np


def query_gen(query_len, time_range, tablename, filepath, id):
    df = pd.read_csv(filepath)
    timestamp = df.iloc[:, 0].values.tolist()[0:query_len]
    values = df.iloc[:, 1].values.tolist()[0:query_len]
    timegap = np.diff(np.array(timestamp)) / (timestamp[-1] - timestamp[0]) * time_range
    queries = []
    for value in values:
        queries.append('update ' + tablename + ' set value=' + str(value) + ' where id=\'' + id + '\'')
    return queries, timegap


if __name__ == '__main__':
    query_len = 100
    time_range = 300  # seconds
    id = 'btcusd'
    filepath = '../dataset/btcusd.csv'
    tablename = 't1'
    queries, timegap = query_gen(query_len, time_range, tablename, filepath, id)
    print(queries, timegap)