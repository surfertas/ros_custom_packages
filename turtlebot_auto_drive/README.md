Notes
---

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
