from __future__ import print_function, division
import json
import csv
import re
import numpy as np
import vegetation_indices as vi
from sys import argv
from collections import namedtuple
from matplotlib import pyplot as plt

t = namedtuple('PLOT_RANGE', ['Y_MIN', 'Y_MAX', 'Y_COUNT'])
y_min = 0
y_max = 2500
y_count = 2501
plot_range = t(Y_MIN=y_min, Y_MAX=y_max, Y_COUNT=y_count)


def get_filename():
    if argv[1] == '--many':
        filename = [arg for arg in argv[1:]]
    else:
        filename = argv[1]
    return filename


def get_headers(filename):
    fh = open(filename, 'r')
    headers = fh.readlines()[0].strip().split('\t')
    return headers


def get_rid_of_numbers(headers):
    """
    This is for python 2.X compatibility
     because translate method has changed in python 3.X
    """
    try:
        translator = {ord(ch): None for ch in '0123456789'}
        return [(h.translate(translator)).split('.')[0] for h in headers[1:]]
    except TypeError:
        return [re.sub(r'\d+', '', head).split('.')[0] for head in headers[1:]]


def load_all_object_measurements(data, start_index, end_index):
    measurement = data[:, start_index:end_index]
    return measurement


def find_start_end_indices(headers, value):
    t_ind = namedtuple('Indices', ['START_INDEX', 'END_INDEX'])
    start_index = headers.index(value) + 1
    end_index = len(headers) - headers[::-1].index(value)
    indices = t_ind(START_INDEX=start_index, END_INDEX=end_index)
    return indices


def get_unique_headers(headers):
    unique_headers = []
    for val in headers:
        if val not in unique_headers:
            unique_headers.append(val)
    return unique_headers


def build_queryset(headers, unique_headers):
    queryset = []
    for uh in unique_headers:
        query = find_start_end_indices(headers, uh)
        queryset.append(query)
    return queryset


def get_arrays_dict(filename, queryset, unique_headers):
    data = np.loadtxt(filename, skiprows=1)
    arrays = {}
    for i, indices in enumerate(queryset):
        name = unique_headers[i]
        arrays[name] = load_all_object_measurements(data, indices.START_INDEX, indices.END_INDEX)
    return arrays


def insert_zeros(array):
    shape = (350, array.shape[1])
    zeros = np.zeros(shape)
    full_arr = np.append(zeros, array)
    full_arr = full_arr.reshape(plot_range.Y_COUNT, array.shape[1])
    return full_arr


def average_measurement(array):
    averaged = np.mean(array, axis=1)
    return averaged


def st_dev_measurement(array):
    st_dev = np.std(array, axis=1)
    # print(st_dev)
    return st_dev


def setup_plot(title, y):
    x = np.linspace(plot_range.Y_MIN, plot_range.Y_MAX, plot_range.Y_COUNT)
    try:
        array_list = np.hsplit(y, y.shape[1])
        for k in array_list:
            arr = np.array(k)
            plt.plot(x, arr, label='Spectral signature')
    except IndexError:
        plt.plot(x, y, label='Spectral signature')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Reflectance')
    plt.xlim([350, plot_range.Y_MAX])
    plt.ylim([0., 1.])
    plt.title(title)
    # plt.legend(loc='best')
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xticks(np.arange(350, plot_range.Y_MAX + 100, 215))
    plt.grid()
    fnm = title + '.pdf'
    plt.savefig(fnm, format='pdf')
    print('Plot {} exported to: {}'.format(title, fnm))
    plt.clf()


def remove_calibration_measurements(arrays_dict):
    cleaned_arrays = {}
    for k, v in arrays_dict.items():
        v1 = []
        for col in range(v.shape[1]):
            if (v[:, col]).mean() <= .9:
                v1.append(v[:, col])
        v1 = np.array(v1)
        v1 = v1.T
        cleaned_arrays[k] = np.array(v1)
    return cleaned_arrays


def generate_statistics(arrays_dict, fnm='statistics.json'):
    output = []
    fnm = '{}.json'.format(fnm.split('.')[0] + '_stats')
    for name, array in arrays_dict.items():
        output.append({
            'Name': name,
            'MIN': np.amin(array),
            'MAX': np.amax(array),
            'STDEV': np.std(array),
            'MEAN': np.mean(array),
            'VARIANCE': np.var(array),
            'MEDIAN': np.median(array),
        })

    with open(fnm, 'w') as outfile:
        json.dump(output, outfile, indent=4)


def work_interactive(filename):
    labels = get_headers(filename)
    results = get_rid_of_numbers(labels)
    unique = get_unique_headers(results)
    queryset = build_queryset(results, unique)
    arrays = get_arrays_dict(filename, queryset, unique)
    interactive_arrays = {}
    for k, v in arrays.items():
        array = insert_zeros(v)
        av_array = average_measurement(array)
        interactive_arrays[k] = av_array
    return interactive_arrays


def generate_csv(arrays, filename='measurements.csv'):
    output = [['wavelength', 'average', 'st_dev']]
    fnm = filename.split('.')[0] + '_csv.csv'
    print('Generating csv file', end='')
    for k, v in arrays.items():
        print('.', end='')
        averaged = average_measurement(v)
        st_deved = st_dev_measurement(v)
        output.append([k, '', ''])
        for i in range(len(averaged)):
            output.append([i + 350, averaged[i], st_deved[i]])
    try:
        with open(fnm, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(output)
    except TypeError:
        with open(fnm, "w") as f:
            writer = csv.writer(f)
            writer.writerows(output)
        print('Exported to: {}'.format(fnm))


def generate_indices_csv(arrays, filename='indices.csv'):
    fnm = filename.split('.')[0] + '_indicescsv.csv'
    indices_array = vi.generate_indices(arrays)
    to_write = [i[1] for i in indices_array]
    headers = [i for i in to_write[0]]
    try:
        with open(fnm, "wb") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(to_write)
            writer.writerow({})
    except TypeError:
        with open(fnm, "w") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(to_write)
            writer.writerow({})
        print('Vegatation indices exported to:{}'.format(fnm))


def _run():
    filename = get_filename()
    labels = get_headers(filename)
    results = get_rid_of_numbers(labels)
    unique = get_unique_headers(results)
    queryset = build_queryset(results, unique)
    arrays = get_arrays_dict(filename, queryset, unique)
    arrays = remove_calibration_measurements(arrays)
    generate_statistics(arrays, fnm=filename)
    generate_csv(arrays, filename=filename)
    arrays_dict = {}

    for k, v in arrays.items():
        extended_array = insert_zeros(v)
        arr = average_measurement(extended_array)
        arrays_dict[k] = arr
        setup_plot(k, extended_array)
        setup_plot(k + '_average', arr)
    vi.generate_indices(arrays_dict)
    generate_indices_csv(arrays_dict, filename=filename)


if __name__ == '__main__':
    _run()
    print('Done.')
