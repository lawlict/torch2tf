import os
import sys
import h5py
import torch

torch_model_path = sys.argv[1]
tf_model_path = sys.argv[2]
match_list = sys.argv[3]

for f in torch_model_path, tf_model_path, match_list:
    assert os.path.isfile(f), "NO SUCH FILE: %s " % f

torch2weight = {k: v.numpy() for k, v in torch.load(torch_model_path).items()}
tf2torch = {x.split()[1]:x.split()[0] for x in open(match_list)}
tf_keys = sorted(list(tf2torch.keys()))

with h5py.File(tf_model_path, 'r+') as f:
    for tf_key in tf_keys:
        tf_weight = f[tf_key]
        torch_key = tf2torch[tf_key]
        torch_weight = torch2weight[torch_key]

        if torch_weight.ndim == 2:
            torch_weight = torch_weight.transpose(1, 0)
        elif torch_weight.ndim == 3:
            torch_weight = torch_weight.transpose(2, 1, 0)
        elif torch_weight.ndim == 4:
            if tf_key[-len('depthwise_kernel:0'):] == 'depthwise_kernel:0':
                torch_weight = torch_weight.transpose(2, 3, 0, 1)
            else:
                torch_weight = torch_weight.transpose(2, 3, 1, 0)

        # print(torch_key, tf_key, torch_weight.shape, tf_weight.shape)
        assert torch_weight.shape == tf_weight.shape, "Shape mismatch"
        assert torch_weight.dtype == tf_weight.dtype, "Dtype mismatch"
        tf_weight[...] = torch_weight
