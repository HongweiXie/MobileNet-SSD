#!/bin/sh
if ! test -f example/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
export PYTHONPATH=../caffe/python
#"mobilenet_iter_73000.caffemodel"
snapshot_dir=diandu_2/snapshot_point_003
mkdir -p ${snapshot_dir}
#latest=$(ls -t ${snapshot_dir}/*.caffemodel | head -n 1)
#echo $latest
#../caffe/build/tools/caffe train -solver="solver_train.prototxt" \
#-weights=$latest \
#-gpu 0,1,2,3,4,5,6,7

latest=$(ls -t ${snapshot_dir}/*.caffemodel | head -n 1)
echo $latest
echo 'start batch_size=64'
../caffe/build/tools/caffe train -solver="solver_train_64.prototxt" \
-weights=$latest \
-gpu 0,1,2,3,4,5,6,7

latest=$(ls -t ${snapshot_dir}/*.caffemodel | head -n 1)
echo $latest
echo 'start batch_size=32'
../caffe/build/tools/caffe train -solver="solver_train_32.prototxt" \
-weights=$latest \
-gpu 0,1,2,3,4,5,6,7



