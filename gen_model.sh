#!/bin/sh
if test -z $1 ;then
	echo usage: $0 CLASSNUM
        echo "        for voc the classnum is 21"
	exit 1
fi
echo $1 |grep '^[0-9]*$' >/dev/null 2>&1
if [ $? != 0 ];then
	echo usage: $0 CLASSNUM
        echo "        for voc the classnum is 21"
	exit 1
fi
cls_num=$1
cls_num2=$(expr $1 \* 2)
cls_num3=$(expr $1 \* 3)
cls_num6=$(expr $1 \* 6)
cls_num4=$(expr $1 \* 4)

layer_num_1x=16
layer_num_2x=$(expr ${layer_num_1x} \* 2)
layer_num_4x=$(expr ${layer_num_1x} \* 4)
layer_num_8x=$(expr ${layer_num_1x} \* 8)
layer_num_16x=$(expr ${layer_num_1x} \* 16)
layer_num_32x=$(expr ${layer_num_1x} \* 32)

trainfile=example/MobileNetSSD_train.prototxt
testfile=example/MobileNetSSD_test.prototxt
deploybnfile=example/MobileNetSSD_deploy_bn.prototxt
deployfile=example/MobileNetSSD_deploy.prototxt

mkdir -p example

cp template/MobileNetSSD_train_template.prototxt $trainfile
sed -i "s/cls6x/${cls_num6}/g" $trainfile
sed -i "s/cls3x/${cls_num3}/g" $trainfile
sed -i "s/cls2x/${cls_num2}/g" $trainfile
sed -i "s/cls1x/${cls_num}/g" $trainfile
sed -i "s/cls4x/${cls_num4}/g" $trainfile
sed -i "s/layer_num_1x/${layer_num_1x}/g" $trainfile
sed -i "s/layer_num_2x/${layer_num_2x}/g" $trainfile
sed -i "s/layer_num_4x/${layer_num_4x}/g" $trainfile
sed -i "s/layer_num_8x/${layer_num_8x}/g" $trainfile
sed -i "s/layer_num_16x/${layer_num_16x}/g" $trainfile
sed -i "s/layer_num_32x/${layer_num_32x}/g" $trainfile

cp template/MobileNetSSD_test_template.prototxt $testfile
sed -i "s/cls6x/${cls_num6}/g" $testfile
sed -i "s/cls3x/${cls_num3}/g" $testfile
sed -i "s/cls1x/${cls_num}/g" $testfile
sed -i "s/cls2x/${cls_num2}/g" $testfile
sed -i "s/cls4x/${cls_num4}/g" $testfile
sed -i "s/layer_num_1x/${layer_num_1x}/g" $testfile
sed -i "s/layer_num_2x/${layer_num_2x}/g" $testfile
sed -i "s/layer_num_4x/${layer_num_4x}/g" $testfile
sed -i "s/layer_num_8x/${layer_num_8x}/g" $testfile
sed -i "s/layer_num_16x/${layer_num_16x}/g" $testfile
sed -i "s/layer_num_32x/${layer_num_32x}/g" $testfile

#cp template/MobileNetSSD_deploy_bn_template.prototxt $deploybnfile
#sed -i "s/cls6x/${cls_num6}/g" $deploybnfile
#sed -i "s/cls3x/${cls_num3}/g" $deploybnfile
#sed -i "s/cls1x/${cls_num}/g" $deploybnfile

cp template/MobileNetSSD_deploy_template.prototxt $deployfile
sed -i "s/cls6x/${cls_num6}/g" $deployfile
sed -i "s/cls3x/${cls_num3}/g" $deployfile
sed -i "s/cls1x/${cls_num}/g" $deployfile
sed -i "s/cls2x/${cls_num2}/g" $deployfile
sed -i "s/cls4x/${cls_num4}/g" $deployfile
sed -i "s/layer_num_1x/${layer_num_1x}/g" $deployfile
sed -i "s/layer_num_2x/${layer_num_2x}/g" $deployfile
sed -i "s/layer_num_4x/${layer_num_4x}/g" $deployfile
sed -i "s/layer_num_8x/${layer_num_8x}/g" $deployfile
sed -i "s/layer_num_16x/${layer_num_16x}/g" $deployfile
sed -i "s/layer_num_32x/${layer_num_32x}/g" $deployfile

