#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;


/**
 * @breif 判断追踪的2D特征点是否在图像边界范围内
**/
bool inBorder(const cv::Point2f &pt);

/**
 * @breif 去除无法追踪的2D特征点
**/
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);

/**
 * @breif 去除无法追踪的2D特征点
**/
void reduceVector(vector<int> &v, vector<uchar> status);

/**
* @class FeatureTracker
* @Description 视觉前端预处理：对每个相机进行角点LK光流跟踪
**/
class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time); //对图像使用光流法进行特征点跟踪

    void setMask(); //对跟踪点进行排序并去除密集点

    void addPoints(); //添将新检测到的特征点n_pts

    bool updateID(unsigned int i); //更新特征点id

    void readIntrinsicParameter(const string &calib_file); //读取相机内参

    void showUndistortion(const string &name); //显示去畸变矫正后的特征点  name为图像帧名称

    void rejectWithF(); //通过F矩阵去除outliers

    void undistortedPoints();

    cv::Mat mask; //图像掩码
    cv::Mat fisheye_mask; //鱼眼相机图像掩码，用来去除边缘噪点
    cv::Mat prev_img, cur_img, forw_img; //prev_img 光流追踪的上一帧图像数据，cur_img 光流追踪的当前帧图像数据，forw_img 光流追踪的后一帧图像数据
    vector<cv::Point2f> n_pts; //每一帧中新提取的特征点
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts; //分别是上一帧2D特征点、当前帧2D特征点、后一帧2D特征点
    vector<cv::Point2f> prev_un_pts, cur_un_pts; //归一化相机坐标系下的2D特征点坐标
    vector<cv::Point2f> pts_velocity; //当前帧相对前一帧特征点沿x,y方向的像素移动速度
    vector<int> ids; //能够被追踪到的特征点的id
    vector<int> track_cnt; //当前帧forw_img中每个特征点被追踪的时间次数
    map<int, cv::Point2f> cur_un_pts_map; 
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera; //相机模型
    double cur_time; //当前帧时间
    double prev_time; //上一帧时间

    static int n_id; //特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
};
