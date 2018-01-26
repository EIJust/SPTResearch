import sqlite3
import math
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
            rssi = int(str_bytes[len(str_bytes)-1])
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
                distance_logs.append([row_dist, rssi, log_time])
            elif not empty_filter:
                distance_logs.append([row_dist, rssi, log_time])

    df = pd.DataFrame(data=distance_logs, columns=['Distance', 'RSSI', 'LogTime'])
    df['LogTime'] = pd.to_datetime(df['LogTime'])
    df['LogTime'] = df['LogTime'].apply(lambda x: x.minute*60+x.second)

    return df


def get_sma(values, buff_size=5):
    sma = list()
    sma.append(values[0])
    for i in range(1, buff_size):
        s = 0
        for j in range(0, i):
            s += values[j]
        s /= i
        sma.append(s)
    for i in range(buff_size, len(values)):
        sma.append(sma[i-1] - values[i-buff_size]/buff_size + values[i]/buff_size)
    return sma


def get_comp_filt(values, alp=0.7):
    filt = list()
    filt.append(values[0])
    for i in range(1, len(values)):
        filt.append(filt[i-1]*alp + values[i]*(1-alp))
    return filt


if __name__ == '__main__':
    log_files = glob.glob("logs/*.sqlite")
    plt.subplots_adjust(hspace=1)
    for log_file in log_files:
        df = from_sqlite(log_file, empty_filter=True)

        clear_line = plt.plot(list(df['LogTime']), list(df['RSSI']), label='clear')
        sma_line = plt.plot(list(df['LogTime']), get_sma(list(df['RSSI'])), label='sma')
        comp_filt_line = plt.plot(list(df['LogTime']), get_comp_filt(list(df['RSSI'])), label='comp_filt')

        plt.xlabel('Time')
        plt.ylabel('RSSI')
        plt.title(log_file.split('\\')[1])
        plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        plt.ylim((0, 150))
        plt.legend(labels=['clear', 'sma', 'comp_filt'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig('graph\\'+log_file.split('\\')[1]+'.png')
        plt.show()
