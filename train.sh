#!/bin/sh
if ! test -f example/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
export PYTHONPATH=/home/sixd-ailabs/Develop/Human/Caffe/caffe/python
#"mobilenet_iter_73000.caffemodel"
snapshot_dir=snapshot_point_10
mkdir -p ${snapshot_dir}
latest=$(ls -t ${snapshot_dir}/*.caffemodel | head -n 1)
echo $latest
../caffe/build/tools/caffe train -solver="solver_train.prototxt" \
-weights=$latest \
-gpu 0




