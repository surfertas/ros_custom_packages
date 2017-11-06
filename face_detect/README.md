Simple ROS face detection package using traditional computer vision techniques.
---

Author: Tasuku Miura

The package uses HAAR_CASCADE based models. The specific model can be specified
in the launch file after placing the respective model in the `/models` folder.

This can be used with any client side that publishes raw images. In this
particular package I chose to use `video_stream_opencv` running on a Raspberry
Pi 3. See [ROS+RaspberryPi Camera Module #3: An alternative package for
publishing images from
Raspi.](http://surfertas.github.io/ros/raspberrypi/2017/09/08/detect-faces-3.html)
for specifics.

To run, launch `webcam.launch` from the raspberry pi.

Once you have confirmed that the topics are being published (check for
`/webcam/image_raw/`), then launch the `face_detect.launch` found in the
`launch` directory in the `ros_face_detect` package.

```
$ roslaunch face_detect.launch
```

If you do not want to output bounding rectangles for performance reasons, go to the `face_detect.launch` file
and set the `param` with name `bound_rectangles` to false.


Note: As this was intended to run on Jetson TX1, I have set the default as the cuda implementation,
thus CMakefile.txt and `face_detect_node.cpp` reflect this. If you would like to run the CPU version
make the necessary edits to CMakefile.txt by changing the names in `add_executables()` and also the header
file included in `face_detect_node.cpp` to `#include face_detect_cuda.h`.




