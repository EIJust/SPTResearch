import csv
import datetime
import math
import tensorflow as tf
import numpy as np
from optparse import OptionParser


def get_positions_from_csv(path):
    tof_pos = []
    ins_pos = []
    times = []
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            tof_pos.append(row[0:2])
            ins_pos.append(row[2:4])
            times.append(datetime.datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S.%f"))
            # >>> datetime.datetime(2013, 9, 28, 20, 30, 55, 782000)

    return tof_pos, ins_pos, times


def get_rotate_angle(from_pos, to_pos):
    ins = tf.placeholder(tf.float32, shape=[2], name='ins')
    tof = tf.placeholder(tf.float32, shape=[2], name='tof')
    alp = tf.Variable(0, dtype=tf.float32)

    sin_alp = tf.sin(alp)
    cos_alp = tf.cos(alp)
    neg_sin_alp = tf.negative(sin_alp)

    concat = tf.concat(0, [cos_alp, neg_sin_alp, sin_alp, cos_alp])
    w = tf.reshape(concat, [2, 2])

    rotate_loss_fn = tof - w * ins

    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(rotate_loss_fn)

    with tf.Session() as sess:
        for _ in range(1000):
            sess.run(train)

        curr_alp = sess.run(alp)
    return curr_alp


def rotate_positions(from_pos, to_pos):
    angle = get_rotate_angle(from_pos, to_pos)
    rotate_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                              [math.sin(angle), math.cos(angle)]])
    rotated_positions = []

    for pos in len(from_pos):
        rotated_positions.append(rotate_matrix.dot(np.array(pos)))

    return rotated_positions


if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("-f", "--file", dest="filename",
                      help="write report to FILE", metavar="FILE")
    parser.add_option("-q", "--quiet",
                      action="store_false", dest="verbose", default=True,
                      help="don't print status messages to stdout")

    (options, args) = parser.parse_args()

    tof_pos, ins_pos, times = get_positions_from_csv(options.filename)

    updated_positions = rotate_positions(ins_pos, tof_pos)

    print(updated_positions)
