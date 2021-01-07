1. save pytorch model weights in bin/model_best.mdl
2. run model/mobilenetv3.py to generate bin/tf_weights.h5
3. run torch2tf_weights.sh with stage=0
4. match $exp/torch2num_params and $exp/tf2num_params by hand.
5. run torch2tf_weights.sh with stage=1

   conda activate tf2.3-cpu
6. run validate.sh with stage=0
7. run validate.sh with stage=1
