#! /usr/bin/env python
# @author Tasuku Miura
# MIT License

import rospy
import message_filters
import cv2
from turtlebot_auto_drive.srv import *
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import os


class DataBuffer(object):

    def __init__(self, data_buffer_path, buffer_threshold):
        self._buffer_path = data_buffer_path
        self._buffer_threshold = buffer_threshold
        # Initialize cv bridge
        self._bridge = CvBridge()

        # Initialize subscribers.
        self._image_sub = message_filters.Subscriber('image', Image)
        self._info_sub = message_filters.Subscriber('camera_info', CameraInfo)
        self._command_sub = message_filters.Subscriber('cmd_vel', CameraInfo)
        rospy.loginfo("Subscribers initialized...")

        self._ts = message_filters.TimeSynchronizer(
            [self._image_sub, self._info_sub, self._command_sub],
            10
        )
        rospy.loginfo("TimeSync initialized...")

        self._ts.registerCallback(self._ts_sub_callback)

        self._service_train_batch = rospy.Service(
            'service_train_batch',
                    DataBatch,
                    self._train_batch_handler
        )
        rospy.loginfo("Train batch service initialized...")

        # Store data buffer information.
        self._img_array = []
        self._cmd_array = []

    def _ts_sub_callback(self, image, info, command):
        """ Call back for synchronize image and command subscribers. """
        try:
            cv_image = self._bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)

        path = self._buffer_path + '{}.png'.format(rospy.get_rostime())
        cv2.imwrite(path, cv_image)

        buffer_size = len(self._img_array)
        if buffer_size < self._buffer_threshold:
            self._img_array.append(path)
            self._cmd_array.append(command)
        else:
            # If buffer is full, pick at random to replace. Remove image from
            # memory first, before appending new.
            # Is there a better way? This will result in a lot of IO ops.
            i = np.random.randint(buffer_size)
            rmv_path = self._img_array[i]
            os.remove(rmv_path)
            self._img_array[i] = path
            self._cmd_array[i] = command

    def _train_batch_handler(self, req):
        """ Handler for service that returns a batch of data. """
        i_train = np.random.randint(len(self._img_array), size=req.batch_size)
        batch = {
            'image_path': self._img_array[i_train],
            'commands': self._cmd_array[i_train]
        }
        return batch


def main():
    rospy.init_node('data_buffer')
    # get path to store data
    if rospy.has_param('/data_buffer_path'):
        data_buffer_path = rospy.get_param('/data_buffer_path')
    else:
        data_buffer_path = '/tmp/data_buffer'

    if rospy.has_param('/buffer_threshold'):
        buffer_threshold = rospy.get_param('/buffer_threshold')
    else:
        buffer_threshold = 10000

    data_buffer = DataBuffer(data_buffer_path, buffer_threshold)

    rospy.spin()

if __name__ == "__main__":
    main()
