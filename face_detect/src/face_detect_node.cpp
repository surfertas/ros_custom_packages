/*
 * @author Tasuku Miura
 */

#include <ros/ros.h>
#include <face_detect/face_detect.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "face_detect");
  ros::NodeHandle nh;
  
  face_detect::FaceDetect face_detector(nh);

  ros::spin();
  return 0;
}
 
    
