import os
import sys
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import kaldiio
from kaldiio.matio import load_mat
from model.mobilenetv3 import MobileNetV3_Small

model_path = sys.argv[1]
feats_scp  = sys.argv[2]
embd_wdir  = sys.argv[3]

for f in model_path, feats_scp:
    assert os.path.isfile(f), "No such file: %s" % f

model = MobileNetV3_Small()
model.build(input_shape=(1, 150, 40))
model.summary()
model.load_weights(model_path)

# Test 1: full-one input
x = np.ones([1, 150, 40], dtype=np.float32)
y = model(x)
print(y)

# Test2: embedding extraction
utt2feat_path = dict(x.split() for x in open(feats_scp))
utts = sorted(list(utt2feat_path.keys()))
utt2embd = {}
for utt in tqdm(utts, ncols=50):
    feat = load_mat(utt2feat_path[utt])
    feat = feat - feat.mean(axis=0, keepdims=True)
    feat = feat.reshape(1, *feat.shape)
    embd = model(feat).numpy().reshape(-1)
    utt2embd[utt] = embd

os.makedirs(embd_wdir, exist_ok=True)
wfile = 'ark,scp:{0}/embedding.ark,{0}/embedding.scp'.format(embd_wdir)
with kaldiio.WriteHelper(wfile) as writer:
    for utt, embd in utt2embd.items():
        writer(utt, embd)
