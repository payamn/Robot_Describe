#include <tf/transform_listener.h>
#include <costmap_2d/costmap_2d_ros.h>
#include "image_transport/image_transport.h"
#include <cv_bridge/cv_bridge.h>
#include <boost/chrono.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>
#include <ros/ros.h>
#include <nav_msgs/GetMap.h>
//#include <costmap_2d.h>

bool map_info(
							nav_msgs::GetMap::Request &reg,
							nav_msgs::GetMap::Response &res,
							costmap_2d::Costmap2D *costmap
							)
{
	res.map.info.resolution = costmap->getResolution();
	res.map.info.origin.position.x = costmap->getOriginX();
	res.map.info.origin.position.y= costmap->getOriginY();
	return true;
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "cost_map_node");
	ros::NodeHandle n("~");

	tf::TransformListener tf(ros::Duration(10));
	costmap_2d::Costmap2DROS costmap("my_costmap", tf);
	auto my_cost_map = costmap.getLayeredCostmap()->getCostmap();

	image_transport::ImageTransport image_transport(n);
	image_transport::Publisher map_image_pub = image_transport.advertise("img", 1);
//	boost::lock_guard<costmap_2d::Costmap2D::mutex_t> lock(*(costmap.getLayeredCostmap()->getCostmap()->getMutex()));
	float width = 0;
	float height = 0;
	cv::Mat cv_img_prev(2,2, CV_8UC1, cv::Scalar::all(0));
	auto timerCallback = [&](const ros::TimerEvent& event)
	{
		boost::lock_guard<costmap_2d::Costmap2D::mutex_t> lock(*(costmap.getLayeredCostmap()->getCostmap()->getMutex()));

		auto t1 = boost::chrono::high_resolution_clock::now();
		auto raw_costmap = costmap.getLayeredCostmap()->getCostmap();

		cv::Mat cv_img;

		auto char_array = raw_costmap->getCharMap();
//		ROS_WARN("x: %f, y: %f, res:%f", raw_costmap->getOriginX (), raw_costmap->getOriginY (), raw_costmap->getResolution() );
		cv_img = cv::Mat(raw_costmap->getSizeInCellsY(), raw_costmap->getSizeInCellsX(), CV_8UC1, char_array);
		cv_img = cv_img.clone();
		bool eq = 0;
		if (cv_img.rows == cv_img_prev.rows && cv_img.cols == cv_img_prev.cols)
		{
			cv::Mat diff = cv_img != cv_img_prev;
			eq = cv::countNonZero(diff) == 0;
		}

		//		costmap.resume();
		if (eq)
		{
			// skipp this
		}
		else if (width == raw_costmap->getSizeInCellsX() && height == raw_costmap->getSizeInCellsY())
		{
			cv_bridge::CvImage cv_ptr;
			cv::Mat cv_img_flipped;               // dst must be a different Mat
		//			cv::flip(cv_img, cv_img_flipped, 0);
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
		cv_img_prev = cv_img.clone();

		auto time = boost::chrono::duration_cast<boost::chrono::milliseconds>(boost::chrono::high_resolution_clock::now()-t1);
			ROS_INFO("execution time %d",time);
	};
	ros::Timer timer = n.createTimer(ros::Duration(0.01), timerCallback);

	ros::ServiceServer service = n.advertiseService<nav_msgs::GetMap::Request, nav_msgs::GetMap::Response>("map_info", boost::bind(&map_info, _1, _2, my_cost_map));


	ros::spin();

}