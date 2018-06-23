import numpy as np  
import sys,os  
import cv2
#caffe_root = '/home/sixd-ailabs/Develop/Human/caffe'
#sys.path.insert(0, caffe_root + 'python')
import caffe  


net_file= 'example/MobileNetSSD_deploy.prototxt'
caffe_model='deploy/MobileNetSSD_deploy.caffemodel'
test_dir = "images"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background',
           'hi_pose','person')

COLORS =((128,128,128),(0,255,0),(0,255,255))
cap = cv2.VideoCapture(0)

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

def detect():
    ret,origimg = cap.read()
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       p3 = (max(p1[0], 15), max(p1[1], 15)-7)
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       if(conf[i]>0.5):
        cv2.rectangle(origimg, p1, p2, COLORS[int(cls[i])],5)
        cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, COLORS[int(cls[i])], 2)
    cv2.imshow("SSD", origimg)
 
    k = cv2.waitKey(1) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True

while True:
    if detect() == False:
       break
