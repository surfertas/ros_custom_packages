#! /usr/bin/env python
# @author Tasuku Miura
# MIT License

import rospy
import message_filters
import cv2
from turtlebot_auto_drive.srv import *
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Twist

import numpy as np
import os


class DataBuffer(object):

    def __init__(self,
                 data_buffer_path,
                 buffer_threshold,
                 image_topic,
                 camera_info_topic,
                 cmd_topic):
        self._buffer_path = data_buffer_path
        self._buffer_threshold = buffer_threshold
        self._image_topic = image_topic
        self._camera_info_topic = camera_info_topic
        self._cmd_topic = cmd_topic

        # Initialize subscribers.
        self._image_sub = message_filters.Subscriber(self._image_topic, CompressedImage)
        self._info_sub = message_filters.Subscriber(self._camera_info_topic, CameraInfo)
        self._command_sub = message_filters.Subscriber(self._cmd_topic, Twist)
        rospy.loginfo("Subscribers initialized...")

        # https://github.com/ros/ros_comm/pull/433/files
        self._ts = message_filters.ApproximateTimeSynchronizer(
            [self._image_sub, self._info_sub, self._command_sub],
            10,
            0.1,
            allow_headerless=True
        )
        rospy.loginfo("ApproximateTimeSync initialized...")

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
        # command = [0,0,0,0,0,0]
        # Cant use cv_bridge for message of type CompressedImage
        # http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
        np_arr = np.fromstring(image.data, np.uint8)

        # cv2.CV_LOAD_IMAGE_COLOR == 1
        cv_image = cv2.imdecode(np_arr, 1)

        path = self._buffer_path + '/{}.png'.format(rospy.get_rostime())
        cv2.imwrite(path, cv_image)
        print("Write to {}".format(path))
        print(command)
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
    if rospy.has_param('data_buffer_path'):
        data_buffer_path = rospy.get_param('data_buffer_path')
    else:
        data_buffer_path = '/tmp/data_buffer'

    if rospy.has_param('buffer_threshold'):
        buffer_threshold = rospy.get_param('buffer_threshold')
    else:
        buffer_threshold = 10000

    if rospy.has_param('test_image_topic'):
        image_topic = rospy.get_param('test_image_topic')
    else:
        rospy.error('No image topic specified.')

    if rospy.has_param('test_camera_info'):
        camera_info_topic = rospy.get_param('test_camera_info')
    else:
        rospy.error('No camera info topic specified.')

    if rospy.has_param('test_cmd_topic'):
        cmd_topic = rospy.get_param('test_cmd_topic')
    else:
        rospy.error('No command topic specified.')

    data_buffer = DataBuffer(
        data_buffer_path,
                    buffer_threshold,
                    image_topic,
                    camera_info_topic,
                    cmd_topic
    )

    print("---------Params loaded----------")
    print("Buffer path: {}".format(data_buffer_path))
    print("Buffer threshold: {}".format(buffer_threshold))
    print("Image topic: {}".format(image_topic))
    print("Camera Info topic: {}".format(camera_info_topic))
    print("Twist command topic: {}".format(cmd_topic))

    rospy.spin()

if __name__ == "__main__":
    main()
