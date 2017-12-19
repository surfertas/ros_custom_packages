#! /usr/bin/env python
#! /usr/bin/env python
# The original PyTorch code for inference was taken from
# https://github.com/marvis/pytorch-yolo2 and modified for use in this
# custom ROS package.

from object_detection.srv import *
import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

from darknet import Darknet
from utils import load_class_names, do_detect, plot_boxes_cv2

# Global variables
model = None
class_names = None
use_cuda = None
bridge = None

def handle_run_inference(req):
    global model, class_names, use_cuda, bridge
    try:
        cv_img = bridge.imgmsg_to_cv2(req.img_req, desired_encoding="passthrough")
    except CvBridgeError as e:
        print(e)
    bboxes = do_detect(model, cv_img, 0.5, 0.4, use_cuda)
    print(len(bboxes))
    print('------')
    draw_img = plot_boxes_cv2(cv_img, bboxes, None, class_names)
    try:
        img = bridge.cv2_to_imgmsg(draw_img, "rgb8")
    except CvBridgeError as e:
        print(e)

    return Yolo2Response(img)
    
def init_model(cfgfile, weightfile, voc_names, coco_names):
    global use_cuda
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights... Done!')

    if m.num_classes == 20:
        namesfile = voc_names
    elif m.num_classes == 80:
        namesfile = coco_names
    else:
        namesfile = voc_names
    class_names = load_class_names(namesfile)

    use_cuda = 1
    if use_cuda:
        m.cuda()

    return m, class_names
  
def run_inference_yolo2():
    global model, class_names, bridge
    rospy.init_node('run_inference_yolo2')
        
    if rospy.has_param('/yolo2_cfg_path'):
        configs_path = rospy.get_param('/yolo2_cfg_path')
    else:
        sys.exit("Cant get path to yolo2 configuration file.")
    
    if rospy.has_param('/yolo2_weights_path'):
        weights_path = rospy.get_param('/yolo2_weights_path')
    else:
        sys.exit("Cant get path to yolo2 weights path.")

    if rospy.has_param('/voc_names'):
        voc_names = rospy.get_param('/voc_names')
    else:
        sys.exit("Cant get voc names.")

    if rospy.has_param('/coco_names'):
        coco_names = rospy.get_param('/coco_names')
    else:
        sys.exit("Cant get coco names.")

    model, class_names = init_model(configs_path, weights_path, voc_names, coco_names)
    bridge = CvBridge()

    s = rospy.Service('run_inference_yolo2', Yolo2, handle_run_inference)
    rospy.spin()

if __name__=='__main__':
    run_inference_yolo2()
