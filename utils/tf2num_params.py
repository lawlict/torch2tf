import os
import sys
import h5py
import numpy as np


model_path = sys.argv[1]
assert os.path.isfile(model_path), "NO SUCH FILE: %s" % model_path

with h5py.File(model_path, 'r') as f:
    key2num_params = {}
    def search_leaf(key):
        if isinstance(f[key], h5py._hl.dataset.Dataset):
            key2num_params[key] = np.array(f[key]).reshape(-1).shape[0]
    f.visit(search_leaf)

    keys = list(key2num_params.keys())
    keys.sort(key=lambda x: key2num_params[x], reverse=True)

    last_num_params = -1
    for key in keys:
        num_params = key2num_params[key]
        if last_num_params == -1:
            last_num_params = num_params
        if last_num_params != num_params:
            print('')
        print(key, num_params)
        last_num_params = num_params