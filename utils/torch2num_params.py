import os
import sys
import torch


model_path = sys.argv[1]
assert os.path.isfile(model_path), "NO SUCH FILE: %s" % model_path

f = torch.load(model_path)
key2num_params = {k: v.reshape(-1).shape[0] for k, v in f.items()}

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