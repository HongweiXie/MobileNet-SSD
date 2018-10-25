import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/sixd-ailabs/Develop/Human/Caffe/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe  
import dlib
import math
tf_pose_root='/home/sixd-ailabs/Develop/Human/TF/tf-pose-estimation'
sys.path.insert(0, tf_pose_root)
from tf_pose import common
from tf_pose.diandu_estimator import TfDianduEstimator
from tf_pose.networks import get_graph_path, model_wh
KEYPOINT_COLORS=[(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0)]
tf_pose_input_size=128
keypoints_estimator = TfDianduEstimator(get_graph_path('mobilenet_lite'), target_size=(tf_pose_input_size, tf_pose_input_size))

def convert2dlibbbox(bbox):
    cx=(bbox[0]+bbox[2])/2
    cy=(bbox[1]+bbox[3])/2
    w=bbox[2]-bbox[0]+1
    h=bbox[3]-bbox[1]+1
    left=max(0,cx-(w*0.5))
    top=max(0,cy-(h*0.5))
    right=cx+w*0.5
    bottom=cy+h*0.5
    return dlib.rectangle(int(left),int(top),int(right),int(bottom))
snapshot_dir='diandu_2/snapshot_point_003'
net_file= snapshot_dir+'/MobileNetSSD_deploy.prototxt'
caffe_model=snapshot_dir+'/MobileNetSSD_deploy.caffemodel'
test_dir = "images"
predictor_path='/home/sixd-ailabs/Develop/Human/Hand/Code/build-Hand-Landmarks-Detector-Desktop_Qt_5_10_0_GCC_64bit-Default/Hand_5_Landmarks_Detector.dat'
predictor = dlib.shape_predictor(predictor_path)

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
caffe.set_mode_gpu()
net = caffe.Net(net_file,caffe_model,caffe.TEST)

CLASSES = ('background',
           'point','point_portrait','point_flip_up_down','other')

COLORS =((128,128,128),(0,255,0),(0,255,255),(255,255,0),(0,0,255))
cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)


img_index=0

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



def detect(video):
    ret,origimg = cap.read()
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)
    h,w,_=origimg.shape
    ladmarkImg = np.zeros(origimg.shape)
    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       p3 = (max(p1[0], 15), max(p1[1], 15)-7)
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       if (conf[i] >= 0.3):
           if(cls[i]==1):
               cx=(box[i][0]+box[i][2])/2
               cy=(box[i][1]+box[i][3])/2
               bw=box[i][2]-box[i][0]
               bh=box[i][3]-box[i][1]
               crop_size=min(max(bw,bh)*1.1,math.sqrt(bw*bw+bh*bh))
               left=cx-crop_size/2
               right=cx+crop_size/2
               top=cy-crop_size/2
               bottom=cy+crop_size/2
               if left<0:
                    left=0
                    right=left+crop_size
               if right>w:
                    right=w
                    left=right-crop_size
               if top<0:
                    top=0
                    bottom=top+crop_size
               if bottom>h:
                   bottom=h
                   top=bottom-crop_size

               hand_img=origimg[int(top):int(bottom),int(left):int(right)]
               # cv2.imshow('hand',hand_img)
               scale=crop_size/tf_pose_input_size
               hand = keypoints_estimator.inference(hand_img, resize_to_default=True, upsample_size=4.0)

               hand_landmark_img = ladmarkImg[int(top):int(bottom), int(left):int(right)]
               pre_point=None
               for index, p in enumerate(hand):
                   if (p[2] > 0.01):
                       cv2.circle(hand_img, (int(p[0]*scale+0.5), int(p[1]*scale+0.5)), 3, KEYPOINT_COLORS[index], -1)
                       cv2.circle(hand_landmark_img, (int(p[0] * scale + 0.5), int(p[1] * scale + 0.5)), 3,
                                  KEYPOINT_COLORS[index], -1)
                       if(index>0):
                           cv2.line(hand_landmark_img,(int(p[0] * scale + 0.5), int(p[1] * scale + 0.5)),
                                    (int(pre_point[0] * scale + 0.5), int(pre_point[1] * scale + 0.5)),(255,255,255),1)
                   pre_point=p

               cv2.rectangle(origimg, p1, p2, COLORS[int(cls[i])], 5)
               cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, COLORS[int(cls[i])], 2)

               # shape = predictor(origimg, convert2dlibbbox(box[i]))
        # for j in range(5):
        #     pt = shape.part(j)
        #     cv2.circle(origimg, (int(pt.x), int(pt.y)), 5, (55, 255, 155), 2)

    cv2.imshow("SSD", origimg)
    cv2.imshow("landmark", ladmarkImg)
    video.write(origimg)
    k = cv2.waitKey(1) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('output/output.avi', fourcc, 20.0, (1280, 720))

while True:

    if detect(video) == False:
       video.release()
       break


