#!/bin/bash
set -e

data=data/voxceleb1_test
exp=exp/MobileNetV3_Small
stage=1

if [ $stage -eq 0 ]; then
  python infer.py $exp/tf_weights.h5 $data/feats.scp $data
  python validate.py $data/trials $data
fi

if [ $stage -eq 1 ];then
  python tf_to_tflite.py $exp/tf_weights.h5 $data/feats.scp $exp/converted_model.tflite
fi
