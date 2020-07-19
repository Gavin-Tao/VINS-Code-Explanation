#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

//extern 引入其它文件中定义的全局变量或函数 (不能是static修饰的，因为static修饰的变量或函数的作用域只存在于当前文件)
//这是变量声明，是告诉编译器应该到该文件外部去找这个文件的定义 
extern int ROW; //图像宽度
extern int COL; //图像高度
extern int FOCAL_LENGTH; //焦距
const int NUM_OF_CAM = 1; //相机的个数


extern std::string IMAGE_TOPIC; //图像的ROS TOPIC
extern std::string IMU_TOPIC; //IMU的ROS TOPIC
extern std::string FISHEYE_MASK; //鱼眼相机mask（掩膜）图像的位置
extern std::vector<std::string> CAM_NAMES; //相机参数配置文件名
extern int MAX_CNT; //特征点最大个数
extern int MIN_DIST; //特征点间的最小距离
extern int WINDOW_SIZE; //滑动窗口的尺寸
extern int FREQ; //控制图像光流跟踪的频率
extern double F_THRESHOLD; //RANSAC算法的阈值
extern int SHOW_TRACK; //是否发布追踪点的图像
extern int STEREO_TRACK; //双目追踪则为1
extern int EQUALIZE; //如果光太亮或太暗则为1，进行直方图均衡化
extern int FISHEYE; //如果是鱼眼相机则为1
extern bool PUB_THIS_FRAME; //是否需要发布帧

void readParameters(ros::NodeHandle &n);
