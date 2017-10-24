import csv
from optparse import OptionParser


def get_row_from_csv(path):
    data = []
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            data.append(row)

    return data


def merge_data_to_csv(path, *data):
    pass

if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("--ins_csv", dest="ins_file")
    parser.add_option("--tof_csv", dest="tof_file")

    (options, args) = parser.parse_args()

    ins_data = get_row_from_csv(options.ins_file)
    tof_data = get_row_from_csv(options.tof_file)

    merge_data_to_csv('builder_output.csv', ins_data, tof_data)
