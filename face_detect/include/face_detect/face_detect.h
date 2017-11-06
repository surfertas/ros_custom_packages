/*
 * @author Tasuku Miura
 */


#ifndef FACE_DETECT_H
#define FACE_DETECT_H

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/fill_image.h>
#include <face_detect/Face.h>
#include <face_detect/Faces.h>
#include <opencv2/objdetect/objdetect.hpp>


namespace face_detect
{

class FaceDetect
{
public:
  FaceDetect(ros::NodeHandle& nh);

  virtual ~FaceDetect();

  void initFaceDetector();

  void registerPublisher();

  void registerSubscriber();

  /**
   *@brief Callback for cam_img_sub_ to convert ROS msg to CvImage
   *
   *@param nh the nodehandle
   */ 
  void convertImageCB(const sensor_msgs::ImageConstPtr& nh);

  /**
   *@brief Main function to detect face given a raw image.
   *
   *@param img opencv Mat type image used for processing.
   */ 
  void detectFace();

  /**
   *@brief Draws bounding boxes around the detected faces.
   *
   */ 
  void drawBoundRect();

private:

 /**
   *@brief Node handle for face detection.
   */ 
  ros::NodeHandle nh_;

 /**
   *@brief ROS Image transport.
   */ 
  image_transport::ImageTransport it_;

 /**
   *@brief Image subscriber to subscribe to raw image topic.
   */ 
  image_transport::Subscriber cam_img_sub_;

 /**
   *@brief Image publisher that publishes image with bounding boxes.
   */ 
  image_transport::Publisher detected_faces_pub_;

 /**
   *@brief Image publisher that publishes custom message Faces.
   */ 
  ros::Publisher detected_faces_coord_pub_;

 /**
   *@brief Bridge uses to convert ROS images to CV images.
   */
  cv_bridge::CvImage bridge_;
  
 /**
   *@brief OpenCV face classifier object.
   */
  cv::CascadeClassifier fc_;

 /**
   *@brief Stores latest image.
   */
  cv::Mat cur_img_;

 /**
   *@brief Image used for marking of bounding boxes.
   */
  cv::Mat img_with_bounding_;

 /**
   *@brief Boolean flag to set the publisher on for bounding boxes.
   */
  bool bounding_;

 /**
   *@brief Stores detected faces in array.
   */
  std::vector<cv::Rect> faces_;

 /**
   *@brief Stores path to classifier model.
   */
  std::string classifier_model_;

};
}

#endif //FACE_DETECT_H

