import numpy as np  
import sys,os  
import cv2
import glob
from pascal_voc_io import PascalVocWriter
caffe_root = '/home/sixd-ailabs/Develop/Human/Caffe/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe  


snapshot_dir='diandu/snapshot_point_18'
net_file= snapshot_dir+'/MobileNetSSD_deploy.prototxt'
caffe_model=snapshot_dir+'/MobileNetSSD_deploy.caffemodel'
test_dir = "/home/sixd-ailabs/Develop/Human/Hand/diandu/test/chengren_17"
save_dir="/home/sixd-ailabs/Develop/Human/Hand/diandu/test/eval_chengren_17_lr"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
caffe.set_mode_gpu()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background',
           'index','index','other','other')
COLORS =((128,128,128),(0,255,0),(0,255,255),(255,255,0),(0,0,255))

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
    xml_name=image_name[:-4]+".xml"
    writer = PascalVocWriter('test', image_name, imgSize=origimg.shape,
                             localImgPath=os.path.join(save_dir, image_name))
    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       if (conf[i]>=0.3):
        cv2.rectangle(origimg, p1, p2, COLORS[int(cls[i])], 3)
        cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, COLORS[int(cls[i])], 1)
        writer.addBndBox(box[i][0],box[i][1],box[i][2],box[i][3],CLASSES[(int)(cls[i])],False,confidence=conf[i]);

    writer.save(os.path.join(save_dir,xml_name))
    cv2.imshow("SSD", origimg)
    cv2.imwrite(os.path.join(save_dir,image_name),origimg)
 
    k = cv2.waitKey(1) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True
import tqdm
jpg_list=glob.glob(test_dir + '/*.jpg')
for f in tqdm.tqdm(jpg_list):
    if detect(f) == False:
       break
