#! /usr/bin/env python
# @author Tasuku Miura
# MIT License

import rospy
import message_filters
import cv2
from turtlebot_auto_drive.srv import *
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Twist, TwistStamped

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
        self._command_sub = message_filters.Subscriber(self._cmd_topic, TwistStamped)
        rospy.loginfo("Subscribers initialized...")

        # https://github.com/ros/ros_comm/pull/433/files
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._image_sub, self._command_sub],
            queue_size=10,
            slop=0.2
        )
        rospy.loginfo("ApproximateTimeSync initialized...")

        self._sync.registerCallback(self._sync_sub_callback)

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
        np_arr = np.fromstring(image.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, 1)

        path = self._buffer_path + '/{}.png'.format(rospy.get_rostime())
        cv2.imwrite(path, cv_image)
        rospy.loginfo("Write to {}".format(path))
        print   (command)
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
            rospy.loginfo("Buffer full, replaced.")

    def _train_batch_handler(self, req):
        """ Handler for service that returns a batch of data. """
        # Need to handle when batch size requested greater than data available
        """
        while req.batch_size > len(self._img_array:
            rospy.loginfo("Waiting for train data to populate...")
        
        i_train = np.random.randint(len(self._img_array), size=req.batch_size)
        batch = {
            'image_path': self._img_array[i_train],
            'commands': self._cmd_array[i_train]
        }
        """
        # FOR TESTING
        test_twist = Twist()
        test_twist.linear.x = 0.5
        batch = {
            'image_path': ['test', 'test1', 'test2'],
            'commands': [test_twist, test_twist, test_twist]
        }
        print(batch)
        return batch


def main():
    rospy.init_node('data_buffer')
    # get path to store data
    if rospy.has_param('data_buffer_path'):
        data_buffer_path = rospy.get_param('data_buffer_path','/tmp/data_buffer')

    if rospy.has_param('buffer_threshold'):
        buffer_threshold = rospy.get_param('buffer_threshold', 10)

    if rospy.has_param('test_image_topic'):
        image_topic = rospy.get_param('test_image_topic')
    else:
        rospy.error('No image topic specified.')

    if rospy.has_param('test_camera_info'):
        camera_info_topic = rospy.get_param('test_camera_info')
    else:
        rospy.error('No camera info topic specified.')

    if rospy.has_param('test_cmd_topic_stamped'):
        cmd_topic = rospy.get_param('test_cmd_topic_stamped')
    else:
        rospy.error('No command topic specified.')

    data_buffer = DataBuffer(
                    data_buffer_path,
                    buffer_threshold,
                    image_topic,
                    camera_info_topic,
                    cmd_topic
    )

    rospy.loginfo("---------Params loaded----------")
    rospy.loginfo("Buffer path: {}".format(data_buffer_path))
    rospy.loginfo("Buffer threshold: {}".format(buffer_threshold))
    rospy.loginfo("Image topic: {}".format(image_topic))
    rospy.loginfo("Camera Info topic: {}".format(camera_info_topic))
    rospy.loginfo("Twist command topic: {}".format(cmd_topic))

    rospy.spin()

if __name__ == "__main__":
    main()
