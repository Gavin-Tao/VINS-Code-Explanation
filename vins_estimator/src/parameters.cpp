#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMU_TOPIC;
double ROW, COL;
double TD, TR;

//读取配置文件
template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

//读取配置文件中的参数
void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    //FileStorage::isOpened（） 检查文件是否已经打开
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    //设置参数
    fsSettings["imu_topic"] >> IMU_TOPIC;

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"]; //10
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH; //最小视差=最小视差/焦距=10.0/460.0

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;

    // create folder if not exists如果不存在则创建文件夹
    FileSystemHelper::createDirectoryIfNotExists(OUTPUT_PATH.c_str());

    //ofstream是从内存到硬盘，ifstream是从硬盘到内存
    //参数：关联文件filename，默认模式 ios_base::out
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ACC_N = fsSettings["acc_n"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %f COL: %f ", ROW, COL);

    //IMU和相机的外参Rt:0准确；1不准确；2没有
    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2) //不提供
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";

    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1) //不准确
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0) //准确
            ROS_WARN(" fix extrinsic param ");

        //读取初始R,t并存入
        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);

        //Eigen::quaternion不同于其它matrix,按照类似于向量类型输出需要q.coeffs(),其包括x,y,z,w四个元素
        /**
        //q1(1,2,3,4), q2(3,2,1,4)    
	    cout << q2.coeffs() << endl;
        q2.normalize();
        cout << q2.coeffs() << endl;
        Eigen::Quaternion<double> q3;
        q3 = q1.normalized();
        cout << q1.coeffs() << endl;
        cout << q3.coeffs() << endl;

        ----------------------------------
        3
        2
        1
        4
        0.547723
        0.365148
        0.182574
        0.730297
        1
        2
        3
        4
        0.182574
        0.365148
        0.547723
        0.730297
        **/
        //normalize()对使用它的变量单位化，无返回
        //normalized() 返回使用它变量的单位化后的值，但是使用它的变量无变化

        /**
        Quaterniond q1(1, 2, 3, 4);  // 第一种方式
        Quaterniond q2(Vector4d(1, 2, 3, 4)); // 第二种方式
        **/
        // 在Quaternion内部的保存中，虚部在前，实部在后，如果以第一种方式构造四元数，则实部是1， 虚部是[2, 3, 4]^T；对于第二种方式，则实部是4，虚部是[1, 2, 3]^T。
        // Quaternion from (1, 2, 3, 4) is: 2 3 4 1
        // Quaternion from vector4d(1, 2, 3, 4) is: 1 2 3 4
        //Eigen::Quaterniond（）四元数初始化
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized(); //四元数归一化
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
        ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
        ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
        
    } 

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    //IMU和cam的时间差. unit: s. readed image clock + TD = real image clock (IMU clock)  配置文件中为0.0
    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    //快门参数
    //0：全局快门相机； 1：滚动快门相机
    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
    {
        //单位s  卷帘门每帧的读出时间（来自数据表）。 
        TR = fsSettings["rolling_shutter_tr"];
        ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }
    
    fsSettings.release();
}
