#! /usr/bin/env python
# @author Tasuku Miura
# MIT License

import rospy
import message_filters
import cv2
from turtlebot_auto_drive.srv import *
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import Twist

import numpy as np
import os


class TopicSyncer(object):

    def __init__(self,
                 image_topic,
                 cmd_topic):
        self._image_topic = image_topic
        self._cmd_topic = cmd_topic

        # Initialize subscribers.
        self._image_sub = message_filters.Subscriber(self._image_topic, CompressedImage)
        self._command_sub = message_filters.Subscriber(self._cmd_topic, Twist)

        self._image_pub = rospy.Publisher(self._image_topic + '/synced', CompressedImage, queue_size=10)
        self._command_pub = rospy.Publisher(self._cmd_topic + '/synced', Twist, queue_size=10)

        self._ts = message_filters.ApproximateTimeSynchronizer(
            [self._image_sub, self._command_sub],

            #            [self._image_sub],
            queue_size=10,
            slop=2.0,
            allow_headerless=True
        )
        rospy.loginfo("ApproximateTimeSync initialized...")

        self._ts.registerCallback(self._ts_sub_callback)

    def _ts_sub_callback(self, image, command):
        """ Call back for synchronize image and command subscribers. """
        # https://gist.github.com/awesomebytes/e02ad0778dfea1692450
        # self._image_pub.publish(image)
#        self._command_pub.publish(command)
        print("SYNCED")


def main():
    rospy.init_node('util_timesync_publisher')
    # get path to store data
    if rospy.has_param('test_image_topic'):
        image_topic = rospy.get_param('test_image_topic')
    else:
        rospy.error('No image topic specified.')

    if rospy.has_param('test_cmd_topic'):
        cmd_topic = rospy.get_param('test_cmd_topic')
    else:
        rospy.error('No command topic specified.')

    TopicSyncer(image_topic, cmd_topic)
    rospy.spin()

if __name__ == "__main__":
    main()
