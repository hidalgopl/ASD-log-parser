import sys
import inspect
import numpy as np
import json
from math import sqrt
from collections import OrderedDict


def mcari(arr):
    return ((arr[700] - arr[670]) - 0.2 * (arr[700] - arr[550])) * (arr[700] / arr[670])


def mcari2(arr):
    return (1.5 * (2.5 * (arr[800] - arr[670]) - 1.3 * (arr[800] - arr[550])))/sqrt((2 * arr[800] + 1) ** 2 - (6 * arr[800] - 5 * sqrt(arr[670])) - 0.5)


def mrendvi(arr):
    return (arr[750] - arr[705]) / (arr[750] + arr[705] - 2 * arr[445])


def mresr(arr):
    return (arr[750] - arr[445]) / (arr[705] - arr[445])


def mtvi(arr):
    return 1.2 * (1.2 * (arr[800] - arr[550]) - 2.5 * (arr[670] - arr[550]))


def mtvi2(arr):
    return (1.5 * (1.2 * (arr[800] - arr[550]) - 2.5 * (arr[670] - arr[550]))) \
           / sqrt((2 * arr[800] + 1) ** 2 - (6 * arr[800] - 5 * sqrt(arr[670])) - 0.5)


def rendvi(arr):
    return (arr[750] - arr[705]) / (arr[750] + arr[705])


def tcari(arr):
    return 3 * ((arr[700] - arr[670]) - 0.2 * (arr[700] - arr[550]) * (arr[700] / arr[670]))


def tvi(arr):
    return 0.5 * (120 * (arr[750] - arr[550]) - 200 * (arr[670] - arr[550]))


def vrei1(arr):
    return arr[740] / arr[720]


def vrei2(arr):
    return (arr[734] - arr[747]) / (arr[715] + arr[726])


def is_mod_function(mod, func):
    return inspect.isfunction(func) and inspect.getmodule(func) == mod


def list_functions(mod):
    itself = ['is_mod_function', 'list_functions', 'count_all', 'generate_indices']
    return [func.__name__ for func in iter(mod.__dict__.values())
            if is_mod_function(mod, func) and func.__name__ not in itself]


def count_all(arr):
    functions = list_functions(sys.modules[__name__])
    indices = OrderedDict()
    for foo in functions:
        indices[foo] = globals()[foo](arr)
    return indices


def generate_indices(arrays_dict, fnm='indices.json'):
    fnm = '{}.json'.format(fnm.split('.')[0] + '_indices')
    output = {}
    for name, array in arrays_dict.items():
        output[name] = count_all(array)
    with open(fnm, 'w') as outfile:
        json.dump(output, outfile, indent=4)

