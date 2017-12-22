#!/usr/bin/env python
# @author Tasuku Miura

# 1. call service, to get a batch of data. # Needs to wait until service is ready.
# 2. split into train and test
# 2. data augmentation on train set
# 3. train on train augmented set.
# 4. validate on test...if best score, then save checkpoint

# http://ieeexplore.ieee.org/document/8014253/?reload=true
import rospy
from turtlebot_auto_drive.srv import *


from keras import __version__
from keras.applications.xception import Xception, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import numpy as np
# https://github.com/keras-team/keras/blob/master/keras/applications/xception.py


class AutoDriveModel(object):
    # Xception only supports the data format 'channels_last' (height, width, channels).

    def __init__(self):
        self._w_img_dim = 299.
        self._h_img_dim = 299.

        self._n_outputs = 6  # linear and twist commands
        self._model_setup()

    def _transfer_learn_setup(self, base_model):
        # Set all layers as trainable.
        # Can decide to train only the top layers, and freeze the bottom layers
        # as an option later.
        for layer in base_model.layers:
            layer.trainable = True

        # As we Xception was trained for classification we want to remove the
        # final fc, and replace with the capability for regression.
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(self._n_outputs)(x)
        model = Model(outputs=predictions, inputs=base_model.input)
        return model

    def _model_setup(self):
        base_model = Xception(weights='imagenet', include_top=False)
        self._model = self._transfer_learn_setup(base_model)

        # Want to use huber loss, but Keras doesnt have loss implemented.
        # logcosh appears to offer similar properties of managing sample outliers.
        # http://www.cs.cornell.edu/courses/cs4780/2015fa/web/lecturenotes/lecturenote10.html
        self._model.compile(optimizer=Adam(), loss='logcosh', metrics=['accuracy'])

    def _build_graph(self):
        pass


def get_train_data(train_batch_size):
    rospy.wait_for_service('service_train_batch')
    rospy.loginfo("Service 'service_train_batch' is available...")
    try:
        serve_train_data = rospy.ServiceProxy('service_train_batch', DataBatch)
        resp = serve_train_data(train_batch_size)
    except rospy.ServiceException as e:
        print("Service call failed: {}".format(e))

    # Convert from ROS Twist message type to numpy array
    to_np_array = lambda cmd: np.array([
        cmd.linear.x,
        cmd.linear.y,
        cmd.linear.z,
        cmd.angular.x,
        cmd.angular.y,
        cmd.angular.z]
    )
    commands = map(to_np_array, resp.commands)
    return resp.image_path, commands


if __name__ == "__main__":
    data = get_train_data(1024)
    print("Data received.")

    model = AutoDriveModel()
