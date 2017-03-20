from __future__ import print_function
import json
import csv
import re
import numpy as np
import vegetation_indices as vi
from sys import argv
from collections import namedtuple
from matplotlib import pyplot as plt

t = namedtuple('PLOT_RANGE', ['Y_MIN', 'Y_MAX', 'Y_COUNT'])
plot_range = t(Y_MIN=0, Y_MAX=2500, Y_COUNT=2501)


def get_filenames():
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
    '''
    This is for python 2.X compatibility
     because translate method changed in python 3.X
    '''
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


def get_arrays_dict(filename, queryset, unique):
    data = np.loadtxt(filename, skiprows=1)
    arrays = {}
    for i, indices in enumerate(queryset):
        name = unique[i]
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
    plt.plot(x, y, label='Spectral signature')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Reflectance')
    plt.xlim([350, plot_range.Y_MAX])
    plt.ylim([0., 1.])
    plt.title(title)
    plt.legend()
    plt.grid()
    fnm = title + '.pdf'
    plt.savefig(fnm, format='pdf')
    print('Plot {} exported to: {}'.format(title, fnm))
    plt.clf()


def plot_spectral(array, X):
    plt.plot(X, array)
    pass


def generate_statistics(arrays_dict, fnm='statistics.json'):
    output = []
    fnm = '{}.json'.format(fnm.split('.')[0] + '_stats')
    for name, array in arrays_dict.items():
        point = {}
        point['Name'] = name
        point['MIN'] = np.amin(array)
        point['MAX'] = np.amax(array)
        point['STDEV'] = np.std(array)
        point['MEAN'] = np.mean(array)
        point['VARIANCE'] = np.var(array)
        point['MEDIAN'] = np.median(array)
        output.append(point)

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
    with open(fnm, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(output)
        print('Exported to: {}'.format(fnm))


def generate_indices_csv(arrays, filename='indices.csv'):
    fnm = filename.split('.')[0] + '_indicescsv.csv'
    indices_array = vi.generate_indices(arrays)
    to_write = [i[1] for i in indices_array]
    headers = [i for i in to_write[0]]
    with open(fnm, "wb") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(to_write)
        writer.writerow({})
        print('Vegatation indices exported to:{}'.format(fnm))


def _run():
    filename = get_filenames()
    labels = get_headers(filename)
    results = get_rid_of_numbers(labels)
    unique = get_unique_headers(results)
    queryset = build_queryset(results, unique)
    arrays = get_arrays_dict(filename, queryset, unique)
    generate_statistics(arrays, fnm=filename)
    generate_csv(arrays, filename=filename)
    arrays_dict = {}

    for k, v in arrays.items():
        extended_array = insert_zeros(v)
        arr = average_measurement(extended_array)
        arrays_dict[k] = arr
        setup_plot(k, arr)
    vi.generate_indices(arrays_dict)
    generate_indices_csv(arrays_dict, filename=filename)


if __name__ == '__main__':
    _run()
    print('Done.')
