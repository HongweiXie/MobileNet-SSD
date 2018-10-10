import numpy as np  
import sys,os  
import cv2
import glob
from pascal_voc_io import PascalVocWriter,PascalVocReader
from object_detection.core import standard_fields
from object_detection.metrics import coco_evaluation
import pickle
import tqdm
caffe_root = '/home/sixd-ailabs/Develop/Human/Caffe/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe  


snapshot_dir='diandu/snapshot_point_18'
net_file= snapshot_dir+'/MobileNetSSD_deploy.prototxt'
caffe_model=snapshot_dir+'/MobileNetSSD_deploy.caffemodel'
test_dir = "/home/sixd-ailabs/Develop/Human/Hand/diandu/test/chengren_17"
iteration=60000
save_file=snapshot_dir+'/detections-{}-{}.pickle'.format(os.path.basename(test_dir), iteration)

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
caffe.set_mode_gpu()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)

    image_name=os.path.basename(imgfile)
    bboxes=[]
    scores=[]
    classes=[]
    for i in range(len(box)):
        if conf[i]>=0.3:
            bboxes.append(box[i])
            scores.append(conf[i])
            classes.append(int(cls[i]))
    detected_result = {'image_id': image_name,'bboxes':bboxes,'scores':scores,'classes':classes}

    return detected_result

if __name__ == '__main__':
    detections=[]
    jpg_list=glob.glob(test_dir + '/*.jpg')
    # for jpg_file in tqdm.tqdm(jpg_list):
    #     detections.append(detect(jpg_file))
    # with open(save_file,'w') as out:
    #     pickle.dump(detections,out)
    with open(save_file,'r') as input:
        detections=pickle.load(input)
    category_list = [{'id': 1, 'name': 'index'},{'id': 2, 'name': 'other'}]
    # category_list = [{'id': 1, 'name': 'index'}]
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(category_list,True,True)

    lable_map={'index':1,'other':2}
    for jpg_file in tqdm.tqdm(jpg_list):
        image_name=os.path.basename(jpg_file)
        xml_file=jpg_file[:-4]+'.xml'
        reader=PascalVocReader(xml_file)
        shapes=reader.getShapes()
        bboxes=[]
        classes=[]
        for shape in shapes:
            points = shape[1]
            xmin = points[0][0]
            ymin = points[0][1]
            xmax = points[2][0]
            ymax = points[2][1]
            label = shape[0]

            bboxes.append([xmin,ymin,xmax,ymax])
            classes.append(lable_map[label])

        coco_evaluator.add_single_ground_truth_image_info(
            image_id=image_name,
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes: np.array(bboxes),
                standard_fields.InputDataFields.groundtruth_classes: np.array(classes)
            }
        )

    cnt = 0
    for d in detections:
        classes=d['classes']
        bboxes=[]
        scores=[]
        new_classes=[]

        for i in range(len(classes)):
            if classes[i]<=1:
                bboxes.append(d['bboxes'][i])
                scores.append(d['scores'][i])
                new_classes.append(1)
                cnt+=1
            else:
                bboxes.append(d['bboxes'][i])
                scores.append(d['scores'][i])
                new_classes.append(2)
        if(len(bboxes)>0):
            coco_evaluator.add_single_detected_image_info(
                image_id=d['image_id'],
                detections_dict={
                    standard_fields.DetectionResultFields.detection_boxes:
                        np.array(bboxes),
                    standard_fields.DetectionResultFields.detection_scores:
                        np.array(scores),
                    standard_fields.DetectionResultFields.detection_classes:
                        np.array(new_classes)
                })
    print(cnt)

    metric,per_metric=coco_evaluator.evaluate()
    print(metric)
    print(per_metric)

