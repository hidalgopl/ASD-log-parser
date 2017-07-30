from __future__ import print_function, division
import sys
import inspect
import json
from math import sqrt, log
from collections import OrderedDict


# BROADBAND GREENNESS INDICES
'''def arvi(arr):
    return


def evi(arr):
    return


def ndvi(arr):
    return


def sr(arr):
    return'''


# NARROWBAND GREENNESS INDICES
def msr705(arr):
    return (arr[750] - arr[445]) / (arr[705] + arr[445])


def ndvi705(arr):
    return (arr[750] - arr[705]) / (arr[750] + arr[705])


def mndvi705(arr):
    return (arr[750] - arr[705]) / (arr[750] + arr[705] - 2 * arr[445])


def vog1(arr):
    return arr[740] / arr[720]


def vog2(arr):
    return (arr[734] - arr[747]) / (arr[715] + arr[726])


def vog3(arr):
    return (arr[734] - arr[747]) / (arr[715] + arr[720])


# LIGHT USE EFFICIENCY INDICES
def pri(arr):
    return (arr[531] - arr[570]) / (arr[531] + arr[570])


def sipi(arr):
    return (arr[800] - arr[445]) / (arr[800] + arr[680])


# CANOPY NITROGEN INDICES
def ndni(arr):
    return (
        (log((1 / arr[1510])) - log((1 / arr[1680]))) /
        log((1 / arr[1510])) + log((1 / arr[1680]))
    )


# DRY OR SENESCENT CARBON INDICES
def ndli(arr):
    return (
        (log((1 / arr[1754])) - log((1 / arr[1680]))) /
        log((1 / arr[1754])) + log((1 / arr[1680]))
    )


def psri(arr):
    return (arr[680] - arr[500]) / arr[750]


def cai(arr):
    return .5 * (arr[2000] + arr[2200]) - arr[2100]


# LEAF PIGMENTS INDICES
def ari1(arr):
    return (1 / arr[550]) - (1 / arr[700])


def ari2(arr):
    return arr[800] * (1 / arr[550]) - (1 / arr[700])


def cri1(arr):
    return (1 / arr[510]) - (1 / arr[550])


def cri2(arr):
    return (1 / arr[510]) - (1 / arr[700])


# CANOPY WATER CONTENT INDICES
def msi(arr):
    return arr[1599] / arr[819]


def ndii(arr):
    return (arr[819] - arr[1649]) / (arr[819] + arr[1649])


def ndwi(arr):
    return (arr[857] - arr[1241]) / (arr[857] + arr[1241])


def wbi(arr):
    return arr[970] / arr[900]


'''
def mcari(arr):
    return ((arr[700] - arr[670]) - 0.2 * (arr[700] - arr[550])) * (arr[700] / arr[670])


def mcari2(arr):
    return (1.5 * (2.5 * (arr[800] - arr[670]) - 1.3 * (arr[800] - arr[550]))) / sqrt(
        (2 * arr[800] + 1) ** 2 - (6 * arr[800] - 5 * sqrt(arr[670])) - 0.5)


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
'''


def is_mod_function(mod, func):
    return inspect.isfunction(func) and inspect.getmodule(func) == mod


def list_functions(mod):
    itself = ['is_mod_function', 'list_functions', 'count_all', 'generate_indices']
    return [func.__name__ for func in iter(mod.__dict__.values())
            if is_mod_function(mod, func) and func.__name__ not in itself]


def count_all(arr, name):
    functions = list_functions(sys.modules[__name__])
    indices = OrderedDict()
    indices['Point'] = name
    for foo in functions:
        indices[foo] = globals()[foo](arr)
    return indices


def generate_indices(arrays_dict, fnm='indices.json'):
    fnm = '{}.json'.format(fnm.split('.')[0] + '_indices')
    output = []
    print('Calculating vegetation indices', end='')
    for name, array in arrays_dict.items():
        print('.', end='')
        output.append([name, count_all(array, name)])
    with open(fnm, 'w') as outfile:
        json.dump(output, outfile, indent=4)
        print('Exported to: {}'.format(fnm))
    return output
