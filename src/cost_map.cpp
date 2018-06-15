#include <tf/transform_listener.h>
#include <costmap_2d/costmap_2d_ros.h>
#include "image_transport/image_transport.h"
#include <cv_bridge/cv_bridge.h>
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <ros/ros.h>

int main(int argc, char** argv)
{
	ros::init(argc, argv, "cost_map_node");
	ros::NodeHandle n("~");

	tf::TransformListener tf(ros::Duration(10));
	costmap_2d::Costmap2DROS costmap("my_costmap", tf);
	image_transport::ImageTransport image_transport(n);
	image_transport::Publisher map_image_pub = image_transport.advertise("img", 1);
//	boost::lock_guard<costmap_2d::Costmap2D::mutex_t> lock(*(costmap.getLayeredCostmap()->getCostmap()->getMutex()));
	float width = 0;
	float height = 0;
	auto timerCallback = [&](const ros::TimerEvent& event) {
//		costmap.pause();
//	  boost::lock_guard<costmap_2d::Costmap2D::mutex_t> lock(*costmap.getLayeredCostmap()->getCostmap()->getMutex());
//		boost::lock_guard<costmap_2d::Costmap2D::mutex_t> lock2(*costmap.getCostmap()->getMutex());
		boost::lock_guard<costmap_2d::Costmap2D::mutex_t> lock(*(costmap.getLayeredCostmap()->getCostmap()->getMutex()));

		auto t1 = boost::chrono::high_resolution_clock::now();
		auto raw_costmap = costmap.getLayeredCostmap()->getCostmap();

		cv::Mat cv_img;
		{
//			while(!costmap.getLayeredCostmap()->isCurrent())
//			{
//				ROS_WARN("not current	");
//				boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
//			}
//			while(!costmap.getLayeredCostmap()->isSizeLocked())
//			{
//				ROS_WARN("size not locked	");
//				boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
//			}
			auto char_array = raw_costmap->getCharMap();
			cv_img = cv::Mat(raw_costmap->getSizeInCellsY(), raw_costmap->getSizeInCellsX(), CV_8UC1, char_array);
			cv_img = cv_img.clone();
			}
//		costmap.resume();
		if (width == raw_costmap->getSizeInCellsX() && height == raw_costmap->getSizeInCellsY())
		{
			cv_bridge::CvImage cv_ptr;
			cv_ptr.image = cv_img;
			cv_ptr.encoding = "mono8";
			cv_ptr.header.stamp = ros::Time::now();
			map_image_pub.publish(cv_ptr.toImageMsg());
		}
		else
		{
			ROS_WARN("size changed");
			width = raw_costmap->getSizeInCellsX();
			height = raw_costmap->getSizeInCellsY();
			boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
		}

		auto time = boost::chrono::duration_cast<boost::chrono::milliseconds>(boost::chrono::high_resolution_clock::now()-t1);
//		ROS_INFO("execution time %d",time);

	};
	ros::Timer timer = n.createTimer(ros::Duration(0.01), timerCallback);
	ros::spin();

}