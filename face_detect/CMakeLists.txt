cmake_minimum_required(VERSION 2.8.3)
project(face_detect)

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

find_package(OpenCV REQUIRED
NO_MODULE
PATHS /usr/local
NO_DEFAULT_PATH)

add_message_files(DIRECTORY msg
  FILES
  Faces.msg
  Face.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
)

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES face_detect
  CATKIN_DEPENDS
    cv_bridge
    image_transport 
    roscpp 
    rospy 
    sensor_msgs
    std_msgs
  DEPENDS OpenCV
)

include_directories(
  include 
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
   src/face_detect_cuda.cpp
)

target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES} 
)

add_executable(${PROJECT_NAME}_node
  src/face_detect_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
   face_detect
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest_gtest(test_face_detect
    test/test_face_detect_launch.test
    src/test/test_face_detect.cpp
  )
  target_link_libraries(test_face_detect
    ${catkin_LIBRARIES}
    face_detect
  )
endif()


