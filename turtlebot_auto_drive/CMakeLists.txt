cmake_minimum_required(VERSION 2.8.3)
project(turtlebot_auto_drive)

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  message_generation
)

find_package(OpenCV REQUIRED
NO_MODULE
PATHS /usr/local
NO_DEFAULT_PATH)


add_service_files(DIRECTORY srv
  FILES
  DataBatch.srv
  PredictTwist.srv 
)

generate_messages(
   DEPENDENCIES
   geometry_msgs
   sensor_msgs
)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES turtlebot_auto_drive
   CATKIN_DEPENDS cv_bridge roscpp rospy geometry_msgs sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_turtlebot_auto_drive.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
