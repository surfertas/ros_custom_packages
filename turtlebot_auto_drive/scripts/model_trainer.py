#!/usr/bin/env python
# @author Tasuku Miura

import rospy
from turtlebot_auto_drive.srv import *
from geometry_msgs.msg import Twist
import os
import tensorflow as tf
import cv2
import numpy as np


class AutoDriveModel(object):

    def __init__(self, sess, ckpt_dir, mini_batch_size, n_epochs):
        self._w_img_dim = 480.
        self._h_img_dim = 640.

        # Twist comprised of linear and angular commands
        self._n_outputs = 6

        # Training related params.
        self._ckpt_dir = ckpt_dir
        self._sess = sess
        self._batch_size = mini_batch_size
        self._n_epochs = n_epochs

        self._build_graph()

    def _init_predict_service(self):
        # Service to handle calls for prediction, when auto driving.
        self._service_get_prediction = rospy.Service(
            'service_get_prediction',
            PredictTwist,
            self._get_prediction_handler
        )
        rospy.loginfo("Get prediction service initialized...")

    def _get_prediction_handler(self, req):
        """ Handler for service_get_prediction. """
        # TODO: Need to modify so that prediction uses latest best model.
        # https://www.tensorflow.org/programmers_guide/saved_model
        cv_image = cv2.imdecode(np.fromstring(req.image.data, np.uint8), 1)

        cmd = self._sess.run([self._predict],
            feed_dict={self._inputs: np.array([cv_image])}
        )
        # Convert prediction to ROS message type.
        msg = Twist()
        cmd = cmd[0][0]
        msg.linear.x = cmd[0]
        msg.linear.y = cmd[1]
        msg.linear.z = cmd[2]
        msg.angular.x = cmd[3]
        msg.angular.y = cmd[4]
        msg.angular.z = cmd[5]
        return msg

    def _model(self, x):
        out = tf.layers.batch_normalization(x)
        out = tf.layers.conv2d(x, 24, [5, 5], (2, 2), "valid", activation=tf.nn.relu)
        out = tf.layers.conv2d(out, 36, [5, 5], (2, 2), "valid", activation=tf.nn.relu)
        out = tf.layers.conv2d(out, 48, [5, 5], (2, 2), "valid", activation=tf.nn.relu)
        out = tf.layers.conv2d(out, 64, [3, 3], (1, 1), "valid", activation=tf.nn.relu)
        out = tf.layers.conv2d(out, 64, [3, 3], (1, 1), "valid", activation=tf.nn.relu)
        out = tf.reshape(out, [-1, 64 * 53 * 73])
        out = tf.layers.dense(out, 100, tf.nn.relu)
        out = tf.layers.dense(out, 50, tf.nn.relu)
        out = tf.layers.dense(out, 10, tf.nn.relu)
        out = tf.layers.dense(out, self._n_outputs)
        return out

    def _build_graph(self):
        """ Build graph and define placeholders and variables. """
        self._inputs = tf.placeholder("float", [None, self._w_img_dim, self._h_img_dim, 3])
        self._targets = tf.placeholder("float", [None, self._n_outputs])

        self._predict = self._model(self._inputs)
        # Use huber loss to make optimization robust to outliers. Use logcosh if
        # using Keras as huber loss not available.
        # http://www.cs.cornell.edu/courses/cs4780/2015fa/web/lecturenotes/lecturenote10.html
        self._loss = tf.losses.huber_loss(
            labels=self._targets,
            predictions=self._predict
        )
        self._train = tf.train.AdamOptimizer().minimize(self._loss)
        rospy.loginfo("Graph built...")

    def _train_admin_setup(self):
        """ Set up utility objects for saving, writing, etc. """
        # Saver
        self._saver = tf.train.Saver()

    def train(self, data):
        """ Train model and save best model after each training run."""
        tf.global_variables_initializer().run()
        # Have to wait until model is initialized to allow inference.
        self._init_predict_service()
        self._train_admin_setup()

        # Check if there is a previously saved checkpoint
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self._ckpt_dir))
        if ckpt and ckpt.model_checkpoint_path:
            rospy.loginfo("Restoring model from: {}".format(ckpt))
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)

        train_iter, train_next = train_input_fn(
            data,
            self._batch_size
        )
        rospy.loginfo("Data iterator initialized...")

        for epoch in range(self._n_epochs):
            self._sess.run(train_iter.initializer)
            epoch_loss = []
            while True:
                try:
                    mini_batch = self._sess.run(train_next)
                    loss, _ = self._sess.run([self._loss, self._train],
                        feed_dict={
                            self._inputs: mini_batch['images'],
                            self._targets: mini_batch['commands']}
                    )
                    epoch_loss.append(loss)
                except tf.errors.OutOfRangeError:
                    break
            rospy.loginfo("Epoch {}, Loss: {}".format(epoch, np.array(epoch_loss).mean()))
        # TODO: Validate and check if validation score is historical best. If so
        # then save checkpoint. 

def get_train_data(train_batch_size):
    rospy.wait_for_service('service_train_batch')
    rospy.loginfo("Service 'service_train_batch' is available...")
    try:
        serve_train_data = rospy.ServiceProxy('service_train_batch', DataBatch)
        resp = serve_train_data(train_batch_size)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: {}".format(e))

    # Convert from ROS Twist message type to numpy array
    to_np_array = lambda cmd: np.array([
        cmd.twist.linear.x,
        cmd.twist.linear.y,
        cmd.twist.linear.z,
        cmd.twist.angular.x,
        cmd.twist.angular.y,
        cmd.twist.angular.z]
    )

    data_set = {
        'images': np.array(resp.image_path),
        'commands': np.array(map(to_np_array, resp.commands))
    }
    return data_set


def train_input_fn(data, mini_batch_size):
    def _decode_resize(image_path, command):
        x = tf.to_float(tf.image.decode_image(tf.read_file(image_path)))
        x.set_shape([480,640,3])
        # Apply some augmentation.
        x = tf.image.per_image_standardization(x)
        x = tf.image.random_brightness(x,0.5)
        x = tf.image.random_contrast(x,0.1,0.8)
        # Clip to handle over and underflow.
        x = tf.minimum(x, 1.0)
        x = tf.maximum(x, 0.0)
#        x = tf.image.resize_images(
#            x,
#            [480, 640], # TODO: resize and change model to accept cropped size,
#                        # use resize_image_with_crop_or_pad
#            tf.image.ResizeMethod.NEAREST_NEIGHBOR
#        )
        return {'images': x, 'commands': command}

    images = data['images']
    commands = data['commands']
    data_set = tf.data.Dataset.from_tensor_slices((images, commands))
    data_set = data_set.map(_decode_resize)
    data_set = data_set.batch(mini_batch_size)
    # Create iterator
    iterator = data_set.make_initializable_iterator()
    next_element = iterator.get_next()
    return iterator, next_element


def main():
    rospy.init_node('model_trainer')
    if rospy.has_param('batch_size'):
        batch_size = rospy.get_param('batch_size', 1024)

    if rospy.has_param('mini_batch_size'):
        mini_batch_size = rospy.get_param('mini_batch_size', 128)

    if rospy.has_param('n_epoch'):
        n_epoch = rospy.get_param('n_epoch', 5)

    if rospy.has_param('ckpt_dir'):
        ckpt_dir = rospy.get_param('ckpt_dir', '/tmp/auto_drive_model')

    rospy.loginfo("----------Model Params------------")
    rospy.loginfo("Batch size: {}".format(batch_size))
    rospy.loginfo("Mini batch size: {}".format(mini_batch_size))
    rospy.loginfo("# of Epochs: {}".format(n_epoch))
    rospy.loginfo("Checkpoint directory: {}".format(ckpt_dir))

    data = get_train_data(batch_size)
    rospy.loginfo("Data received.")

    # TODO: Change it so it periodically starts a new training session, with a
    # new batch of data.
    with tf.Session() as sess:
        model = AutoDriveModel(sess, ckpt_dir, mini_batch_size, n_epoch)
        model.train(data)

    rospy.spin()

if __name__ == "__main__":
    main()
