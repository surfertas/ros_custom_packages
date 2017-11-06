/*
 * @author Tasuku Miura
 * @brief Simple face detection package with cuda enabled.
 */

#include "face_detect/face_detect_cuda.h"

namespace face_detect {


FaceDetect::FaceDetect(ros::NodeHandle& nh) :
  nh_(nh),
  it_(nh_)
{
  initFaceDetector();
  registerPublisher();
  registerSubscriber();
}


FaceDetect::~FaceDetect() {}


void FaceDetect::initFaceDetector()
{
  if (!nh_.getParam("haar_cascade", classifier_model_)) {
    std::cerr << "No model file name found in parameter server." << std::endl;
  }

  // If not specified at launch time from launch file, then do not showing
  // bounding boxes on image.
  if (!nh_.getParam("bounding_rectangles", bounding_)) {
    bounding_ = false;
  }

  if(!(fc_ = cv::cuda::CascadeClassifier::create(classifier_model_))) {
    std::cerr << "Error: Could not load face classifer. Check file is valid." << std::endl;
  } else {
    std::cout << "Model loaded..." << std::endl;
  }
}


void FaceDetect::registerPublisher()
{
  detected_faces_pub_ = it_.advertise("/face_detect/detected_faces", 0);
  detected_faces_coord_pub_ = nh_.advertise<face_detect::Faces>("/face_detect/detected_faces_coord", 0);

  std::cout << "Publisher initialized.." << std::endl;
}


void FaceDetect::registerSubscriber() 
{
  cam_img_sub_ = it_.subscribe(
    "/webcam/image_raw", 10, &FaceDetect::convertImageCB, this, image_transport::TransportHints("compressed"));

  std::cout << "Subscriber initialized.." << std::endl;
}


void FaceDetect::convertImageCB(const sensor_msgs::ImageConstPtr& img)
{
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e) {
    std::cerr << "cv_bridge exception: " << e.what();
    return;
  }

  cur_img_ = cv_ptr->image;
  detectFace();
}


void FaceDetect::detectFace()
{
  cv::Mat img_gray;
  cv::cuda::GpuMat img_gray_gpu;
  cv::cuda::GpuMat img_cur_gpu;
  
  // Convert Mat to GpuMat
  img_gray_gpu.upload(img_gray);
  img_cur_gpu.upload(cur_img_);

  cv::cuda::cvtColor(img_cur_gpu, img_gray_gpu, CV_BGR2GRAY);
  cv::cuda::equalizeHist(img_gray_gpu, img_gray_gpu);
	
  cv::cuda::GpuMat objbuf;
	
  // Find faces in image that are greater than min size (10,10) and store in vector<cv::Rect>.
  fc_->detectMultiScale(img_gray_gpu, objbuf);
  fc_->convert(objbuf, faces_);
  std::cout << "Faces detected...: " << faces_.size() << std::endl;
  
  face_detect::Faces faces_msg;
  faces_msg.img.header.stamp = ros::Time::now();

  // Fill out location of detected faces and add to array of faces
  std::for_each(faces_.begin(), faces_.end(), [&](cv::Rect face_src) {
    face_detect::Face face;
    face.x = face_src.x;
    face.y = face_src.y;
    face.h = face_src.height;
    face.w = face_src.width;
    faces_msg.faces.push_back(face);
  });
  
  // Need to manually fill out attributes as cur_img_ was allocated separately
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cur_img_).toImageMsg();

  faces_msg.img = *msg;
  detected_faces_coord_pub_.publish(faces_msg);

  // Set boolean in launch file, if you want to publish image with rectangles
  if (bounding_) {
    drawBoundRect();
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_with_bounding_).toImageMsg();
    detected_faces_pub_.publish(msg);
  }
}


void FaceDetect::drawBoundRect()
{
  img_with_bounding_ = cur_img_.clone();
  // Draws bounding rectangles around each detected face.
  std::for_each(faces_.begin(), faces_.end(), [&](cv::Rect face) {
     cv::rectangle(img_with_bounding_, face, cv::Scalar(0,255,0), 10); 
  });
}
} // face_detect
