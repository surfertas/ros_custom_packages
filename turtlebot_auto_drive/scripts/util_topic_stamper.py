#! /usr/bin/env python
# @author Tasuku Miura
# MIT License

import rospy
import message_filters
from geometry_msgs.msg import Twist, TwistStamped

import os
import numpy as np


class TopicStamper(object):
    # Helper class to convert Twist message to TwistStamped so that the message
    # can be synced with other topics.

    def __init__(self, cmd_topic):
        self._cmd_topic = cmd_topic

        # Initialize subscribers.
        self._cmd_sub = rospy.Subscriber(self._cmd_topic, Twist, self._sub_callback)

        # Initialize publisher.
        self._cmd_pub = rospy.Publisher(self._cmd_topic + '/stamped', TwistStamped, queue_size=10)
        rospy.loginfo("Topic stamper ready...")

    def _sub_callback(self, cmd):
        """ Call back to convert to TwistStamped message type"""
        msg = TwistStamped()
        msg.header.stamp = rospy.get_rostime()
        msg.twist = cmd
        self._cmd_pub.publish(msg)


def main():
    rospy.init_node('util_timesync_publisher')
    if rospy.has_param('test_cmd_topic'):
        cmd_topic = rospy.get_param('test_cmd_topic')
    else:
        rospy.error('No command topic specified.')

    TopicStamper(cmd_topic)
    rospy.spin()

if __name__ == "__main__":
    main()
