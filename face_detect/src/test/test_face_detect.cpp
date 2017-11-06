/*
 * @author Tasuku Miura
 */

#include <ros/ros.h>
#include <gtest/gtest.h>
#include <thread>
#include <face_detect/face_detect.h>


using namespace face_detect;

TEST(FaceDetect, testDummy)
{
  ASSERT_EQ(true, true);
}

TEST(FaceDetect, initDetector)
{
  ros::NodeHandle nh;
  ASSERT_NO_THROW(FaceDetect face_detector(nh););
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "face_detect");

  testing::InitGoogleTest(&argc, argv);
  std::thread t([]{while(ros::ok()) ros::spin();});

  auto result = RUN_ALL_TESTS();
  ros::shutdown();
  return result;
}
