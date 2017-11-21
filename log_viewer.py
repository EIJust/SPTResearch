import sqlite3
import math
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize


class Trilateration:
    def __init__(self, anchors, distances):
        self.anchors = anchors
        self.distances = distances

    def get_point(self):
        x0 = np.array([0, 0])

        point = minimize(self.culc_loss_func, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
        return point.x

    def culc_loss_func(self, x):
        point = [x[0], x[1], 0]
        value = 0
        count_valid_dist = 0

        for i in range(0, len(self.anchors)):
            anchor = self.anchors[i]
            distance = self.distances[i]

            if distance is None:
                continue
            count_valid_dist += 1

            value += (math.sqrt((anchor[0] - point[0]) ** 2 + (anchor[1] - point[1]) ** 2 + (anchor[2] - point[2]) ** 2)
                      - distance) ** 2

        return value / count_valid_dist


def from_sqlite(db_path, tag_id=None, empty_filter=False):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM HidReportLogs;")

    distance_logs = []
    for row in rows:
        row_dist = list()
        log_time = row[2]
        if row[3] == tag_id or tag_id is None:
            str_bytes = row[1].split(',')
            anchor_count = int(str_bytes[0])
            empty_count = 0
            for i in range(0, anchor_count):
                distance_int = (list(map(int, str_bytes[i * 2 + 1: i * 2 + 3])))
                if distance_int == [255, 255]:
                    distance_int = None
                else:
                    distance_int = distance_int[0] + distance_int[1] * 0.01
                row_dist.append(distance_int)
                if distance_int is None:
                    empty_count += 1
            if empty_filter and empty_count != anchor_count:
                distance_logs.append([row_dist, log_time])
            elif not empty_filter:
                distance_logs.append([row_dist, log_time])

    df = pd.DataFrame(data=distance_logs, columns=['Distance', 'LogTime'])
    df['LogTime'] = pd.to_datetime(df['LogTime'])
    df['LogTime'] = df['LogTime'].apply(lambda x: x.minute*60+x.second)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', type=str, help='.sqlite logs file')
    args = parser.parse_args()
    logs_path = args.logs

    df = from_sqlite(logs_path, empty_filter=True)
    anchors = [[9, 3.5, 3],
               [9, 0, 3],
               [0, 0, 3],
               [0, 3.5, 3],
               [3, 1.75, 3],
               [6, 1.75, 3]]

    positions_x = list()
    positions_y = list()

    for distances in df['Distance'].values:
        trilateration = Trilateration(anchors=anchors, distances=distances)
        point = trilateration.get_point()
        positions_x.append(point[0])
        positions_y.append(point[1])

    df['X'] = pd.Series(positions_x)
    df['Y'] = pd.Series(positions_y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(df['X'], df['Y'], df['LogTime'])
    ax.legend()

    plt.show()