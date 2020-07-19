#include "feature_tracker.h"

//FeatureTracker的static成员变量n_id初始化为0
//n_id 特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
int FeatureTracker::n_id = 0;

/**
 * @breif 判断追踪的2D特征点是否在图像边界范围内，在边界范围内返回true
 * @Description 相当于横竖各画了两条线，是否在中间新形成的四边形内
 * @input 2D特征点
**/
//Point2f 代表float型的2D点
/**语句中最后的i，f等所表达的意思为：
 * b是unsigned character,
 * s是short integer, 
 * i是32-bit integer,
 * f是32-bit floating-point number, 
 * d是64-bit floating-point number.
**/
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1; //边界尺寸
    /**函数cvRound，cvFloor，cvCeil 都是用一种舍入的方法将输入浮点数转换成整数：
    * cvRound()：返回跟参数最接近的整数值，即四舍五入；
    * cvFloor()：返回不大于参数的最大整数值，即向下取整；
    * cvCeil()：返回不小于参数的最小整数值，即向上取整；
    **/
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    //看是否在新四边形内
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE; //COL：图像高度  ROW：图像宽度
}

/**
 * @breif 去除无法追踪的2D特征点
 * @Description 如果是无法追踪的2D特征点，则status[i]为空。
 *              如果可以追踪的话，将第i个特征点赋值给第j个特征点，同时j自加，这样就可剔除无法追踪的特征点。
 *              最后v[j] j从0到最后都是可以追踪的特征点，再resize一下。
**/
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

/**
 * @breif 去除无法追踪的2D特征点
 * @Description 同上
**/
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

//空构造
FeatureTracker::FeatureTracker()
{
}

/**
 * @brief   对跟踪点进行排序并去除密集点
 * @Description 对跟踪到的特征点，按照被追踪到的次数排序并依次选点
 *              使用mask进行类似非极大抑制，半径为30，去掉密集点，使特征点分布均匀            
 * @return      void
*/
void FeatureTracker::setMask()
{
    //如果采用鱼眼相机，则FISHEYE =1
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255)); //x=col=width ,y=row=height
        //在Mat类型变量访问时下标是反着写的，即：按照(y, x)的关系形式访问
        //cv::Scalar(255)表示将单通道图像每个像素值初始化设为255
    

    // prefer to keep features that are tracked for long time
    //构造（track_cnt,forw_pts,ids）序列   当前帧forw_img中每个特征点被追踪的时间次数，2D特征点，id
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        //将三个数据合成一个数据（cnt,pts,id）
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i]))); //将两个数据组合成一个数据，两个数据可以是同一类型或者不同类型。

    //降序 lambda表达式  eg.sort(begin(vec), end(vec), [](const Student& lhs, const Student& rhs) 
    //          {
	//			return lhs.grade < rhs.grade；
    //          }); 
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first; //first代表track_cnt，按照被追踪的次数来降序
         });

    //清空cnt，pts，id并重新存入
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        //当前特征点位置对应的mask值为255，则保留当前特征点，将对应的特征点位置pts，id，被追踪次数cnt分别存入
        if (mask.at<uchar>(it.second.first) == 255) //at 访问对应index中存储的bai数据
        {
            //it的first是track_cnt，it的second是（forw_pts,ids）
            //it.second.first是forw_pts,it.second.second是ids
            forw_pts.push_back(it.second.first); 
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            //画圆  void cv::circle (InputOutputArray img, Point center, int radius, const Scalar &color, int thickness=1, int lineType=LINE_8, int shift=0)
            //thickness 如果是正数，表示组成圆的线条的粗细程度。否则，-1表示圆是否被填充
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1); 
        }
    }
}

//添加新检测到的特征点n_pts
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p); //在forw_pts中加入新检测的特征点
        ids.push_back(-1); //ids中加入-1
        track_cnt.push_back(1); //track_cnt中加入1
    }
}


/**
 * @brief   对图像使用光流法进行特征点跟踪
 * @Description createCLAHE() 对图像进行自适应直方图均衡化
 *              calcOpticalFlowPyrLK() LK金字塔光流法
 *              setMask() 对跟踪点进行排序，设置mask
 *              rejectWithF() 通过基本矩阵剔除outliers
 *              goodFeaturesToTrack() 添加特征点(shi-tomasi角点)，确保每帧都有足够的特征点
 *              addPoints()添加新的追踪点
 *              undistortedPoints() 对角点图像坐标去畸变矫正，并计算每个角点的速度
 * @param[in]   _img 输入图像
 * @param[in]   _cur_time 当前时间（图像时间戳）
 * @return      void
*/
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    //EQUALIZE: 如果光太亮或太暗则为1，进行直方图均衡化
    if (EQUALIZE)
    {
        //createCLAHA(clipLimit=8.0, titleGridSize=(8, 8)) 构建CLAHA对象
            //clipLimit 颜色对比度阈值
            //titleGridSize 进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        //限制对比度的自适应直方图均衡化
        clahe->apply(_img, img); //_img 原图； img 对比度增强后的图
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    //forw_img 光流追踪的后一帧图像数据
    //如果forw_img为空，说明当前是第一次读入图像数据
    if (forw_img.empty())
    {
        //将读入的图像赋给forw_img，同时还赋给prev_img、cur_img
        prev_img = cur_img = forw_img = img;
    }
    //否则，说明之前就已经有图像读入，只需要更新当前帧？forw_img的数据
    else
    {
        forw_img = img;
    }

    //此时forw_pts还保存的是上一帧图像中的特征点，所以把它清除
    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        
        //LK金字塔光流法
        // calcOpticalFlowPyrLK	(	
        //     InputArray 	prevImg, 第一个输入图像或金字塔
        //     InputArray 	nextImg, 第二个输入图像或金字塔
        //     InputArray 	prevPts, 需要找到流的2D点的矢量，点坐标必须是单精度浮点数。
        //     InputOutputArray 	nextPts, 输出二维点的矢量（具有单精度浮点坐标），包含第二图像中输入特征的计算新位置
        //     OutputArray 	status, 输出状态向量（无符号字符）;如果找到相应特征的流，则向量的每个元素设置为1，否则设置为0。
        //     OutputArray 	err, 输出错误的矢量
        //     Size 	winSize = Size(21, 21), 每个金字塔等级的搜索窗口的winSize大小
        //     int 	maxLevel = 3, //基于0的最大金字塔等级数;如果设置为0，则不使用金字塔（单级），如果设置为1，则使用两个级别.
        //                        //此处是使用四个级别
        //                      ）
        //     https://blog.csdn.net/weixin_42905141/article/details/93745116
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            //可以找到相应特征的流（输出状态向量为1）并且后一帧的2D特征点不在边界内（！inBorder为true）
            //将位于图像边界外的点的输出状态向量设置为0
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        
        //根据status，把追踪失败的点去除
        //不仅要从当前帧数据forw_pts中剔除，而且还要从cur_un_pts、prev_pts和cur_pts中剔除
        //prev_pts和cur_pts中的特征点是一一对应的
        //记录特征点id的ids，和记录特征点被跟踪次数的track_cnt也要剔除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    //光流追踪成功,特征点被成功跟踪的次数就加1
    //数值代表被追踪的次数，数值越大，说明被追踪的就越久
    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        //通过F矩阵去除outliers
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        //对跟踪点进行排序并去除密集点
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        
        //计算是否需要提取新的特征点
        //特征点最大个数-当前帧特征点个数  static_cast强制转换数据类型
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
             /** 
             *Description:  在mask中不为0的区域检测新的特征点
             *void cv::goodFeaturesToTrack(    
             *   InputArray  image,              输入图像
             *   OutputArray     corners,        存放检测到的角点的vector
             *   int     maxCorners,             返回的角点的数量的最大值
             *   double  qualityLevel,           角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
             *   double  minDistance,            返回角点之间欧式距离的最小值
             *   InputArray  mask = noArray(),   和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
             *   int     blockSize = 3,          计算协方差矩阵时的窗口大小
             *   bool    useHarrisDetector = false,  指示是否使用Harris角点检测，如不指定则使用shi-tomasi算法
             *   double  k = 0.04                Harris角点检测需要的k值
             *)   
             */
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    //当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    //把当前帧的数据forw_img、forw_pts赋给上一帧cur_img、cur_pts
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints(); //根据不同的相机模型去畸变矫正和转换到归一化坐标系上，计算速度，prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
}


/**
 * @brief   通过F矩阵去除outliers
 * @Description 将图像坐标转换为归一化坐标
 *              cv::findFundamentalMat()计算F矩阵
 *              reduceVector()去除outliers 
 * @return      void
*/
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8) //RANSAC 算法，点数目 >= 8 
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size()); //归一化坐标系下的2D特征点
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            //Vector3d 实际上是Eigen::Matrix<double, 3, 1>,即三维向量
            Eigen::Vector3d tmp_p;
            //liftProjective函数是对鱼眼相机模型的标定及去畸变过程
            //liftProjective（）将图像特征点的坐标映射到空间坐标，里面涉及处理畸变的过程。最后得到两组特征点的位置
            //通过这两组特征点得到基础矩阵，通过基础矩阵剔除一些不好的点
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p); //提升tem_p 3D点到归一化平面
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0; //u=fx*X/Z+cx
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0; //v=fy*Y/Z+cy
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        //  cvFindFundamentalMat(  points1, //特征点1 ：点坐标应该是浮点数(双精度或单精度)
        //                         points2, //特征点2 ：点坐标应该是浮点数(双精度或单精度)
        //                         int    method=CV_FM_RANSAC, //计算基本矩阵的方法
                                                                    //CV_FM_7POINT – 7-点算法，点数目＝ 7
                                                                    //CV_FM_8POINT – 8-点算法，点数目 >= 8
                                                                    //CV_FM_RANSAC – RANSAC 算法，点数目 >= 8
                                                                    //CV_FM_LMEDS - LMedS 算法，点数目 >= 8 
        //                         double param1=1., //点到对极线的最大距离，超过这个值的点将被舍弃，不用于后面的计算
        //                         double param2=0.99, //表示矩阵正确的可信度
        //                         status=NULL); //在计算过程中没有被舍弃的点被置为1；否则置为0
        //调用cv::findFundamentalMat对un_cur_pts和un_forw_pts计算F矩阵
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        //根据status，把追踪失败的点去除
        //不仅要从当前帧数据forw_pts中剔除，而且还要从cur_un_pts、prev_pts和cur_pts中剔除
        //prev_pts和cur_pts中的特征点是一一对应的
        //记录特征点id的ids，和记录特征点被跟踪次数的track_cnt也要剔除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

//更新特征点id
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++; //n_id 特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
        return true;
    }
    else
        return false;
}

//读取相机内参
void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str()); //c_str()：输出文件名
    //比如：
    //string s = "Hello World!";
    //printf("%s", s.c_str());    // 输出 "Hello World!"
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

//显示去畸变矫正后的特征点  name为图像帧名称
void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

/**
 * @breif 对上一帧归一化下特征点及地图进行push_back
 * @Description  上一帧归一化坐标系下特征点及地图清空clear()
 *               liftProjective(2D, 3D)
 *               尾插上一帧归一化坐标系下的特征点（x/z,y/z)
 *               在上一帧地图中插入（ids[i],2D坐标点)、
 *               计算v_x,v_y并进行pts_velocity.push_back
 *               当前帧归一化平面下的特征地图赋值给上一帧归一化平面下的特征地图
*/
void FeatureTracker::undistortedPoints()
{
    //上一帧归一化坐标系下特征点及地图清空clear()
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        ////根据不同的相机模型将二维坐标转换到三维坐标
        m_camera->liftProjective(a, b);
        //再延伸到深度归一化平面上,尾插上一帧归一化坐标系下的特征点（x/z,y/z)
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        //在上一帧地图中插入（ids[i],2D坐标点)
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    //计算特征点速度
    //如果之前归一化特征地图不是空的，执行循环
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear(); //当前帧相对前一帧特征点沿x,y方向的像素移动速度清空
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            //检测到新特征点时的ids[i]=-1
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    //it->second 指cv::Point2f    
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt; //计算x方向的像素移动速度
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt; //计算y方向的像素移动速度
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
