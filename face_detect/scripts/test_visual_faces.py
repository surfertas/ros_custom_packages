#!/usr/bin/env python

# license removed for brevity
# author: Tasuku Miura
#
# Test to visually confirm that the coordinates passed in custom messages Faces
# can be used to draw bounding rectangles around the detected faces on server
# side.

import rospy
import cv2

from sensor_msgs.msg import Image
from face_detect.msg import Faces
from face_detect.msg import Face
from cv_bridge import CvBridge


class test_faces:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/faces_detected_visualization", Image, queue_size=10)
        self.sub = rospy.Subscriber("/face_detect/detected_faces_coord", Faces, self.callback)

    def callback(self, msg):
        
        # Converts image to cv::Mat
        cv_img = self.bridge.imgmsg_to_cv2(msg.img, desired_encoding=msg.img.encoding)

        # Draws bounding boxes around each detected face
        for face in msg.faces:
            cv2.rectangle(
                cv_img, 
                (face.x, face.y), 
                ((face.x + face.w), (face.y + face.h)), 
                (0,255,0), 
                10
            ) 

        msg_out = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")    

        self.pub.publish(msg_out)

    
def main():
    test = test_faces()
    rospy.init_node("face_visualization", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
