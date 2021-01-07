#!/bin/bash
set -e

torch_path=bin/model_best.mdl 
tf_path=bin/tf_weights.h5
exp=exp/MobileNetV3_Small
stage=1

if [ $stage -eq 0 ]; then
  mkdir -p $exp
  python utils/torch2num_params.py $torch_path | grep -v num_batches_tracked > $exp/torch2num_params
  python utils/tf2num_params.py $tf_path > $exp/tf2num_params
  # check whether the number of parameters are the same
  paste $exp/torch2num_params $exp/tf2num_params | awk '{if($2!=$4) print}'
fi

# Before running stage 1, you should modify tf2num_params to match torch2num_params by hand.
# chmod -w $exp/torch2num_params $exp/tf2num_params

if [ $stage -eq 1 ]; then
  paste $exp/torch2num_params $exp/tf2num_params | awk '{if(NF>0) print $1,$3}' > $exp/match.list
  new_tf_path=$exp/`basename $tf_path`
  cp $tf_path $new_tf_path
  python utils/torch_weights_to_tf.py $torch_path $new_tf_path $exp/match.list
fi
