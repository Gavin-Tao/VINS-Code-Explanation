#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

/**
 * @breif 调试信息：开启节点，name是节点名称
 * 读取name配置文件
**/
//ros::NodeHandle 创建时候，如果内部节点没有开始，ros::NodeHandle会开始节点，ros::NodeHandle实例销毁，节点就会关闭。
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

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    //配置文件数据名（string类型）
    config_file = readParam<std::string>(n, "config_file");
    //FileStorage类将各种OpenCV数据结构的数据存储为XML 或 YAML格式。同时，也可以将其他类型的数值数据存储为这两种格式。
    //构造函数  cv::FileStorage(const string& source, int flags， const string& encoding=string());
    //参数：
        //source –存储或读取数据的文件名（字符串），其扩展名(.xml 或 .yml/.yaml)决定文件格式。
        //flags – 操作模式，包括：
            //FileStorage::READ 打开文件进行读操作
            //FileStorage::WRITE 打开文件进行写操作
            //FileStorage::APPEND打开文件进行附加操作
            //FileStorage::MEMORY 从source读数据，或向内部缓存写入数据（由FileStorage::release返回）
        //encoding – 文件编码方式。目前不支持UTF-16 XML 编码，应使用 8-bit 编码。
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    //vins_folder路径
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];
    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file); //相机参数配置文件名

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100; //控制图像光流跟踪的频率

    fsSettings.release(); //由FileStorage::release返回


}
