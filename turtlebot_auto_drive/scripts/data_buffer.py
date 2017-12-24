#! /usr/bin/env python
# @author Tasuku Miura
# MIT License

import rospy
import message_filters
import cv2
from turtlebot_auto_drive.srv import *
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist, TwistStamped

import numpy as np
import os


class DataBuffer(object):

    def __init__(self,
                 data_buffer_path,
                 buffer_threshold,
                 image_topic,
                 cmd_topic):
        self._buffer_path = data_buffer_path
        self._buffer_threshold = buffer_threshold
        self._image_topic = image_topic
        self._cmd_topic = cmd_topic

        # Initialize subscribers.
        self._image_sub = message_filters.Subscriber(self._image_topic, CompressedImage)
        self._command_sub = message_filters.Subscriber(self._cmd_topic, TwistStamped)
        rospy.loginfo("Subscribers initialized...")

        # Sync subscribers
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._image_sub, self._command_sub],
            queue_size=10,
            slop=0.2
        )
        self._sync.registerCallback(self._sync_sub_callback)
        rospy.loginfo("ApproximateTimeSync initialized...")

        self._service_train_batch = rospy.Service(
            'service_train_batch',
            DataBatch,
            self._train_batch_handler
        )
        rospy.loginfo("Train batch service initialized...")

        # Store data buffer information.
        self._img_array = []
        self._cmd_array = []

    def _sync_sub_callback(self, image, command):
        """ Call back for synchronize image and command subscribers. """
        cv_image = cv2.imdecode(np.fromstring(image.data, np.uint8), 1)
        path = self._buffer_path + '/{}.png'.format(rospy.get_rostime())
        cv2.imwrite(path, cv_image)

        buffer_size = len(self._img_array)
        if buffer_size < self._buffer_threshold:
            self._img_array.append(path)
            self._cmd_array.append(command)
        else:
            # If buffer is full, pick at random to replace. Remove image from
            # memory first, before appending new.
            # TODO:  Is there a better way? This will result in a lot of IO ops.
            i = np.random.randint(buffer_size)
            rmv_path = self._img_array[i]
            os.remove(rmv_path)
            self._img_array[i] = path
            self._cmd_array[i] = command
            rospy.loginfo("Buffer full, replaced.")

    def _train_batch_handler(self, req):
        """ Handler for service that returns a batch of data. """
        # Need to handle when batch size requested greater than data available
        rospy.loginfo("Waiting for data to populate...")
        while req.batch_size > len(self._img_array):
            continue
        rospy.loginfo("Train data to populated...")

        i_train = np.random.randint(len(self._img_array), size=req.batch_size)
        batch = {
            'image_path': np.array(self._img_array)[i_train],
            'commands': np.array(self._cmd_array)[i_train]
        }
        return batch


def main():
    rospy.init_node('data_buffer')
    # get path to store data
    if rospy.has_param('data_buffer_path'):
        data_buffer_path = rospy.get_param('data_buffer_path', '/tmp/data_buffer')

    if rospy.has_param('buffer_threshold'):
        buffer_threshold = rospy.get_param('buffer_threshold', 10)

    if rospy.has_param('test_image_topic'):
        image_topic = rospy.get_param('test_image_topic')
    else:
        rospy.error('No image topic specified.')

    if rospy.has_param('test_cmd_topic_stamped'):
        cmd_topic = rospy.get_param('test_cmd_topic_stamped')
    else:
        rospy.error('No command topic specified.')

    data_buffer = DataBuffer(
        data_buffer_path,
        buffer_threshold,
        image_topic,
        cmd_topic
    )

    rospy.loginfo("---------Params loaded----------")
    rospy.loginfo("Buffer path: {}".format(data_buffer_path))
    rospy.loginfo("Buffer threshold: {}".format(buffer_threshold))
    rospy.loginfo("Image topic: {}".format(image_topic))
    rospy.loginfo("Twist command topic: {}".format(cmd_topic))
    rospy.spin()

if __name__ == "__main__":
    main()
