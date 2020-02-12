import os
import pickle

def pkl_dump(obj, name, path = 'obj'):
    """Saves an object in pickle format."""
    if '.p' not in name:
        name = name + '.pkl'
    path = os.path.join(path, name)
    pickle.dump(obj, open(path, 'wb'))

def pkl_load(name, path = 'obj'):
    """Restores an object from a pickle file."""
    if '.p' not in name:
        name = name + '.pkl'
    path = os.path.join(path, name)
    try:
        obj = pickle.load(open(path, 'rb'))
    except FileNotFoundError:
        obj = None
    return obj

import json
def save_json(dict_obj, path, file_name):
    """Saves a dictionary in json format."""
    if 'txt' not in file_name:
        file_name += '.txt'
    with open(os.path.join(path, file_name), 'w') as json_file:
        json.dump(dict_obj, json_file)

def load_json(path, file_name):
    """Restores a dictionary from a json file."""
    if 'txt' not in file_name:
        file_name += '.txt'
    with open(os.path.join(path, file_name), 'r') as json_file:
        return json.load(json_file)

import functools
def join_path(list):
    """From a list of chained directories, forms a path."""
    return functools.reduce(os.path.join, list)
