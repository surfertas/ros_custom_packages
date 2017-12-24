#! /usr/bin/env python
# @author Tasuku Miura
# MIT License

import rospy
import message_filters
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from turtlebot_auto_drive.srv import PredictTwist

import cv2
import numpy as np
import os


class TwistPredictor(object):
    # Subscribes to streaming raw image topic and calls service to get a
    # prediction for Twist commands to be published to the robot.
    def __init__(self,
                 image_topic,
                 cmd_topic):
        self._image_topic = image_topic
        self._cmd_topic = cmd_topic

        rospy.wait_for_service('service_get_prediction')
        rospy.loginfo("Service 'service_get_prediction' is available...")
        self._serve_get_prediction = rospy.ServiceProxy(
            'service_get_prediction',
            PredictTwist,
            persistent=True
        )
        self._image_sub = rospy.Subscriber(self._image_topic, CompressedImage, self._sub_callback)
        # TODO: Currently publish to predicted, but need to publish to
        # self._cmd_topic, to move the robot autonomously using the predictions.
        # Figure out way to be able to override predictions using teleop.
        self._cmd_pub = rospy.Publisher(self._cmd_topic + '/predicted', Twist, queue_size=10)

    def _sub_callback(self, image_msg):
        """ Handler for image subscriber. """
        try:
            resp = self._serve_get_prediction(image_msg)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))
        self._cmd_pub.publish(resp.commands)


def main():
    rospy.init_node('predict_twist')
    if rospy.has_param('test_image_topic'):
        image_topic = rospy.get_param('test_image_topic')
    else:
        rospy.error('No image topic specified.')

    if rospy.has_param('test_cmd_topic'):
        cmd_topic = rospy.get_param('test_cmd_topic')
    else:
        rospy.error('No command topic specified.')

    TwistPredictor(image_topic, cmd_topic)
    rospy.spin()

if __name__ == "__main__":
    main()
