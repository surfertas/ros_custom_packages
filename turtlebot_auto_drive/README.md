Usage
---
Tested on Raspberry Pi 3 running ROS Kinetic + Ubuntu 16.04.2 Mate, and a
Turtlebot 2 from Clearpath Robotics controlled by a note PC running Ubuntu
14.04.2 LTS + ROS Indigo. ROS master was initiated on a separate PC
running ROS Kinetic + Ubuntu 16.04.2 LTS.



Main PC (Master):
1. Launch `autobot.launch` found in the custom designed `turtlebot_auto_drive`
package. The `autobot.launch` file takes care of starting all nodes related to
creating the data buffer, model for training, and necessary utility nodes used
for topic conversions. Note: If you want to work with a bag file to iterate on a
deep learning model, without having the need to start up all the nodes on the
Raspberry Pi, and Turtlebot, launch the file as `roslaunch autobot.launch
off_line:=true`. The ROS bag will be saved to `turtlebot_auto_drive/bags`.

Raspberry Pi:
1. Launch `webcam.launch` found in the `video_stream_opencv` package from a
Raspberry Pi.

Turtlebot:
1. Launch the  `minimal.launch` from the `turtlebot_bringup` package on the note
PC connected to the Turtlebot. 
1. Launch the `ps3_teleop.launch` found in the `turtlebot_teleop` package.


After teleoping the robot around a bit, enough data for training should have
accumulated to enable training. You should see output similar to the following:

```
core service [/rosout] found
process[data_buffer-1]: started with pid [26166]
process[util_topic_stamper-2]: started with pid [26167]
process[model_trainer-3]: started with pid [26168]
process[predict_twist-4]: started with pid [26169]
[INFO] [1514185923.510318]: Topic stamper ready...
[INFO] [1514185923.580780]: Subscribers initialized...
[INFO] [1514185923.581317]: ApproximateTimeSync initialized...
[INFO] [1514185923.584422]: Train batch service initialized...
[INFO] [1514185923.584862]: ---------Params loaded----------
[INFO] [1514185923.585178]: Buffer path: home/catkin_ws/src/turtlebot_auto_drive/data/images
[INFO] [1514185923.585394]: Buffer threshold: 10000
[INFO] [1514185923.585582]: Image topic: /webcam/image_raw/compressed
[INFO] [1514185923.585757]: Twist command topic: /mobile_base/commands/velocity/stamped
[INFO] [1514185940.295102]: ----------Model Params------------
[INFO] [1514185940.295567]: Batch size: 32
[INFO] [1514185940.295888]: Mini batch size: 8
[INFO] [1514185940.296165]: # of Epochs: 5
[INFO] [1514185940.296442]: Checkpoint directory: /home/catkin_ws/src/turtlebot_auto_drive/data/models
[INFO] [1514185940.300038]: Service 'service_train_batch' is available...
[INFO] [1514185940.320639]: Waiting for data to populate...
[INFO] [1514185955.412366]: Train data to populated...
[INFO] [1514185955.416035]: Data received. 
2017-12-25 16:12:35.470300: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
[INFO] [1514185958.060898]: Graph built...
[INFO] [1514185962.001560]: Get prediction service initialized...
[INFO] [1514185962.069614]: Service 'service_get_prediction' is available...
[INFO] [1514185962.889941]: Data iterator initialized...
[INFO] [1514186030.176313]: Epoch 0, Loss: 42.002532959
[INFO] [1514186093.322655]: Epoch 1, Loss: 9.709400177

```

Notes
---
12/22/2017 Added rosbag recording capabilities in order to iterate efficiently
on model design. Worked on service to get a batch of data from the data buffer
to be used in training. 

12/21/2017 Got ApproximateTimeSync to subscribe to image, camera info, and
command topic. The image is being published by a node running on a raspberry pi,
while the command topic is being publish by a node running on the mobile robot.
The image is successfully saved to disk, and the path and associated twist
command is saved as a feature, label pair.

12/19/2017 Worked on data_buffer.py. The node/service that will handle
subscribing to the raw images and velocity commands from controller, storing to
memory, and handling the service calls that will return a batch of data, where
batch size is specified by the caller. Need to write tests and test with dummy
data...

12/18/2017 Worked on implementing the model train node, that will continuously
call the service to get data from the data buffer that will be used for
training. Keras framework was used for its ease in data augmentation and
transfer learning on pre-trained models. The model chosen for the first pass is
the Xception model. Will train all layers, and retrain the final FC to allow for
regression in place of classification. Consider the context of this project, an
architect robust to outlier samples is necessary to instead of MSE, chose
logcosh. (Huber loss is not implemented in Keras, but logcosh offers similar
properties. See [logcosh vs. huber.](http://www.cs.cornell.edu/courses/cs4780/2015fa/web/lecturenotes/lecturenote10.html)
