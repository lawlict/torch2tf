import os
import sys
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import kaldiio
from kaldiio.matio import load_mat
from model.mobilenetv3 import MobileNetV3_Small


model_path   = sys.argv[1]
feats_scp    = sys.argv[2]
tflite_wpath = sys.argv[3]

for f in model_path, feats_scp:
    assert os.path.isfile(f), "No such file: %s" % f

model = MobileNetV3_Small()
model.build(input_shape=(1, 150, 40))
model.summary()
model.load_weights(model_path)

x = np.ones([1, 150, 40], dtype=np.float32)
y = model(x)

utt2feat_path = dict(x.split() for x in open(feats_scp))
utts = sorted(list(utt2feat_path.keys()))

# Export tflite model
def representative_dataset_gen():
    for utt in tqdm(utts, ncols=50):
        feat = load_mat(utt2feat_path[utt])
        feat = feat - feat.mean(axis=0, keepdims=True)
        if len(feat) < 150:
            N = math.ceil(150 / len(feat))
            feat = np.concatenate([feat] * N, axis=0)
        feat = feat[:150]
        feat = feat.reshape(1, *feat.shape)
        yield [feat]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.representative_dataset = representative_dataset_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

open(tflite_wpath, "wb").write(tflite_model)
