#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator; //Estimator类的对象

std::condition_variable con; //条件变量
double current_time = -1;

//三个buf
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;

//等待IMU刷新时间
int sum_of_wait = 0;

//互斥量，锁
//队列imu_buf、feature_buf、relo_buf是被多线程共享的，因而在回调函数将相应的msg放入buf或进行pop时，需要设置互斥锁m_buf，在操作前lock()，操作后unlock()
std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;

//IMU项[P,Q,B,Ba,Bg,a,g]
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

//始化feature、imu、last_imu_t
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

//从IMU测量值imu_msg和上一个PVQ递推得到当前PVQ
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    //判断是否是第一帧imu_msg
    if (init_imu) //if语句里的return，使程序跳出if所在的函数，返回到母函数中继续执行。
    {
        latest_time = t;
        init_imu = 0;
        return;
    }

    //计算当前imu_msg距离上一个imu_msg的时间间隔
    double dt = t - latest_time;
    latest_time = t;

    //取x,y,z三个方向上的线加速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    //获取x,y,z三个方向上的角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    //tmp_P,tmp_V,tmp_Q更新,这一部分有公式推导 ，利用的是中值积分
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g; //a理想值=R(a测量值-Ba)-g  w理想值=w测量值-BW 此式为k时刻的加速度的模型

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg; //中值法：取k时刻和k+1时刻的角速度再除2
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt); //旋转的更新

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g; //k+1时刻的加速度

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    //迭代赋值
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

//得到窗口最后一个图像帧的IMU项[P、Q、V、ba、bg、a、g]，并对imu_buf中剩余imu_msg进行PVQ递推
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

//对IMU和图像数据进行对齐并组合
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        //IMU或feature是空的直接返回空结果
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        //对齐标准：IMU最后一个数据的时间要大于第一个图像特征数据的时间，否则继续等待，返回空值
        // IMU最新的时间戳 小于 最旧的图像帧，说明IMU数据太少，等待IMU刷新
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        //对齐标准：不满足 最旧的IMU在最旧的相机帧之前，即相机帧太旧了，要丢弃。保证了图像帧之前一定要有IMU
        // 开头，只有图像，后来突然来了IMU，但是第一帧图像太旧了，就丢地第一帧图像。
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;

        //图像数据(img_msg)，对应多组在时间戳内的imu数据,然后塞入measurements
        // 理想：IMU最旧的在图像帧最旧的之前，说明IMU数据足够
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front()); //emplace_back相比push_back能更好地避免内存的拷贝与移动
            imu_buf.pop();
        }

        //这里把下一个imu_msg也放进去了,但没有pop，因此当前图像帧和下一图像帧会共用这个imu_msg
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

//将imu_msg保存到imu_buf，IMU状态递推并发布[PVQ，header] "imu_propagate"
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //判断时间间隔是否为正
    if (imu_msg->header.stamp.toSec() <= last_imu_t) //toSec()单位转换，last_imu_t=0
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec(); //获取最新的imu时间戳

    //将imu_msg保存到imu_buf，记得加锁解锁
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();

    con.notify_one(); //在feature_callback和imu_callback中还设置了条件锁，在完成将msg放入buf的操作后唤醒作用于process线程中的获取观测值数据的函数

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        //构造互斥锁m_state，析构时解锁
        std::lock_guard<std::mutex> lg(m_state); //在imu_callback中还通过lock_guard的方式构造互斥锁m_state，能够自动加锁、解锁
        
        //递推得到IMU的PQV
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";

        //发布最新的由IMU直接递推得到的PQV
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

//feature回调函数，将feature_msg放入feature_buf
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    //判断是否第一帧，光流没有速度
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }

    //feature_msg加入到feature_buf中，记得加锁解锁
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();

    con.notify_one(); //在feature_callback和imu_callback中还设置了条件锁，在完成将msg放入buf的操作后唤醒作用于process线程中的获取观测值数据的函数
}

//收到restart时清空feature_buf和imu_buf，估计器重置，时间重置
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");

        //清空feature和imu的buf，记得对m_uf加锁解锁
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();

        //估计器重置，参数重置，记得对m_eastimator加锁解锁
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();

        //时间重置
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

//relocalization回调函数，将points_msg放入relo_buf
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
//给RVIZ发送里程计信息PQV、关键点三维坐标、相机位姿、点云信息、IMU到相机的外参、重定位位姿等
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        //在提取measurements时互斥锁m_buf会锁住，此时无法接收数据;等待measurements上面两个接收数据完成就会被唤醒
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0; //调用wait函数，先解锁lk，然后判断lambda的返回值
                 });
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements) //遍历获取的Feature和IMU测量值
        {
            //IMU 预积分
            auto img_msg = measurement.second; //对应这段的img data
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first) //某一图像帧下遍历对齐的imu
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;

                //发送IMU数据进行预积分
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time; //两个imu时间间隔
                    ROS_ASSERT(dt >= 0);
                    current_time = t; //更新当前imu时间
                    dx = imu_msg->linear_acceleration.x; //线速度
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x; //角速度
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    //imu预积分
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame 重定位
            //在relo_buf中取出最后一个重定位帧，拿出其中的信息并执行setReloFrame()
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            //取出最后一个重定位帧
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                //重定位
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            //处理图像
            //建立每个特征点的(camera_id,[x,y,z,u,v,vx,vy])s的map，索引为feature_id
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            //处理图像特征
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header); //"odometry"
            pubKeyPoses(estimator, header); //"key_poses"
            pubCameraPose(estimator, header); //"camera_pose"
            pubPointCloud(estimator, header); //"history_cloud"
            pubTF(estimator, header); //"extrinsic"
            pubKeyframe(estimator); //"keyframe_point"、"keyframe_pose"
            if (relo_msg != NULL)
                pubRelocalization(estimator); //"relo_relative_pose"
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        //更新IMU参数
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) //非线性优化
            update(); //更新IMU参数[P,Q,V,ba,bg,a,g]
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    //ROS初始化设置节点vins_estimator，设置句柄n
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    
    //读取参数，设置估计器参数
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    //用于RVIZ显示的Topic
    registerPub(n);

    //订阅IMU、feature、restart、match_points的话题topic,执行各自回调函数
    //每当订阅的节点由数据送过来就会进入到相应的回调函数中
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    //创建VIO主线程
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
