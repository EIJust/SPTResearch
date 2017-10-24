import sqlite3
import struct
import numpy as np
from scipy.optimize import minimize
import math
import csv
from optparse import OptionParser


class Anchor:
    id = 0
    x = 0
    y = 0
    z = 0

    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z


class Trilateration:

    def __init__(self, anchors, distances):
        self.anchors = anchors
        self.distances = distances

    def get_point(self):
        x0 = np.array([0, 0])

        point = minimize(self.culc_loss_func, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
        return point

    def culc_loss_func(self, x):
        point = [x[0], x[1], 0]
        value = 0
        count_vaild_dist = 0

        for i in range(0, len(self.anchors)):
            anchor = self.anchors[i]
            distance = self.distances[i]

            if distance[0] == 255:
                continue
            count_vaild_dist += 1

            distance = distance[0] + distance[1] * 0.01
            value += (math.sqrt((anchor.x - point[0])**2 + (anchor.y - point[1])**2 + (anchor.z - point[2])**2) - distance)**2

        return value / count_vaild_dist


def get_distances_from_db_by_tag_id(db_path, tag_id):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM HidReportLogs;")

    distances = []
    for row in rows:
        row_dist = list()
        if row[3] == tag_id:
            str_bytes = row[1].split(',')
            for i in range(0, 6):
                distance_int = (list(map(int, str_bytes[i * 2 + 1: i * 2 + 3])))
                row_dist.append(distance_int)
            distances.append(row_dist)

    return distances


def get_points(anchors, distances):
    res_points = []

    for point_distances in distances:
        empty_values = list(filter(lambda x: x[0] == 255, point_distances))
        if len(empty_values) == len(point_distances):
            continue
        trilateration = Trilateration(anchors, point_distances)
        t = trilateration.get_point()
        res_points.append(list(t.x))

    return res_points


def save_points_to_csv(path, points):
    with open(path, 'w+') as csv_file:
        log_writer = csv.writer(csv_file, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for point in points:
            log_writer.writerow([point[0], point[1]])


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--in_f", "--input_file", dest="input_filename",
                      help="ins csv file for reading", metavar="FILE")
    parser.add_option("--out_f", "--output_file", dest="output_filename",
                      help="tof csv file for reading", metavar="FILE")

    (options, args) = parser.parse_args()
    option_dict = vars(options)

    input_filename, output_filename = option_dict["input_filename"], option_dict["output_filename"]

    anchors = [Anchor(0, 0, 0, 2.5),
               Anchor(1, 0, 7.1, 2.5),
               Anchor(2, 10, 7.1, 2.5),
               Anchor(3, 20, 7.1, 2.5),
               Anchor(4, 20, 0, 2.5),
               Anchor(5, 8, 0, 2.5)]

    distances = get_distances_from_db_by_tag_id(input_filename, 17)
    print(distances)
    points = get_points(anchors, distances)
    save_points_to_csv(output_filename, points)

    print(len(points))