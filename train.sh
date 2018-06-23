#!/bin/sh
if ! test -f example/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
snapshot_dir=snapshot_hi_pose_2
mkdir -p ${snapshot_dir}
latest=$(ls -t ${snapshot_dir}/*.caffemodel | head -n 1)
echo $latest
../caffe/build/tools/caffe train -solver="solver_train.prototxt" \
--weights=snapshot_hi_pose/mobilenet_iter_7684.caffemodel \
-gpu 0
