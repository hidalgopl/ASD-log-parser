import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt
import json


# t = namedtuple('PLOT RANGE', ['Y_MIN', 'Y_MAX', 'Y_COUNT'])
# PLOT_RANGE = t(Y_MIN=0, Y_MAX=2500, Y_COUNT=2501)
def get_filenames():
    pass


def get_headers(filename):
    fh = open(filename, 'r')
    headers = fh.readlines()[0].strip().split('\t')
    return headers


def get_rid_of_numbers(headers):
    translator = {ord(ch): None for ch in '0123456789'}
    return [(h.translate(translator)).split('.')[0] for h in headers[1:]]


def load_all_object_measurements(data, start_index, end_index):
    measurement = data[:, start_index:end_index]
    return measurement


def find_start_end_indices(headers, value):
    t = namedtuple('Indices', ['START_INDEX', 'END_INDEX'])
    start_index = headers.index(value) + 1
    end_index = len(headers) - headers[::-1].index(value)
    indices = t(START_INDEX=start_index, END_INDEX=end_index)
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


def get_arrays_dict(filename, queryset):
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
    full_arr = full_arr.reshape(2501, array.shape[1])
    return full_arr


def average_measurement(array):
    averaged = np.mean(array, axis=1)
    return averaged


def setup_plot(title, y):
    # x = np.linspace(PLOT_RANGE.Y_MIN, PLOT_RANGE.Y_MAX, PLOT_RANGE.Y_COUNT)
    x = np.linspace(0, 2500, 2501)
    plt.plot(x, y, label='Spectral signature')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Reflectance')
    plt.xlim([0, 2500])
    plt.ylim([0., 1.])
    plt.title(title)
    plt.legend()
    plt.savefig(title+'.pdf', format='pdf')
    plt.clf()


def plot_spectral(array, X):
    plt.plot(X, array)
    pass


def generate_statistics(arrays_dict, fnm='statistics.json'):
    output = {}
    for name, array in arrays_dict.items():
        point = {}
        point['Name'] = name
        point['MIN'] = np.amin(array)
        point['MAX'] = np.amax(array)
        point['STDEV'] = np.std(array)
        point['MEAN'] = np.mean(array)
        point['VARIANCE'] = np.var(array)
        point['MEDIAN'] = np.median(array)
        output[name] = point
    with open(fnm, 'w') as outfile:
        json.dump(output, outfile, indent=4)


filename = 'pkt_21-25.txt'
data = np.loadtxt(filename, skiprows=1)
labels = get_headers(filename)
results = get_rid_of_numbers(labels)
# print ('Translated:', results)
print('Unique_headers:', get_unique_headers(results))
unique = get_unique_headers(results)
# print ('Queryset:', build_queryset(results,unique))
queryset = build_queryset(results, unique)
arrays = get_arrays_dict(filename, queryset)
generate_statistics(arrays)
for k, v in arrays.items():
    print(k, v.shape)
    rr = insert_zeros(v)
    print (rr, rr.shape)
    arr = average_measurement(rr)
    #averaged = np.mean(rr, axis=1)
    #print(averaged)
    setup_plot(k, arr)
    # rr.tofile(k, sep='\t')