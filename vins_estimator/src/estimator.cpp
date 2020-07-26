#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

//视觉测量残差的协方差矩阵
void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric); //Matrix3d
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD; //camera数据与IMU数据时间戳的偏移值
}

//清空或初始化滑动窗口中所有的状态量      
void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false, //是不是第一个imu消息
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

//IMU预积分，中值积分得到当前PQV作为优化初值
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    //判断是不是第一个imu消息，如果不是第一个imu消息，则将当前传入的imu消息作为第一个imu消息，将当前传入的线加速度和角速度作为初始的加速度和角速度
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration; //linear_acceleration是刚传入的线加速度
        gyr_0 = angular_velocity; //angular_velocity是刚传入的角速度
    }

    //IMU预积分类对象还没出现，创建一个对象
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]}; //创建对象
    }
    
    //当frame_count==0的时候表示滑动窗口中还没有图像帧数据，不需要进行预积分，只进行线加速度和角速度初始值的更新
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;     

        //采用的是中值积分的传播方式     
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g; //a0=Q(a^-ba)-g 
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j]; //w=0.5(w0+w1)-bg
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1); //中值积分下的加速度a=1/2(a0+a1)
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc; //P=P+v*t+1/2*a*t^2
        Vs[j] += dt * un_acc; //V=V+a*t
    }

    //当frame_count==0的时候表示滑动窗口中还没有图像帧数据，不需要进行预积分，只进行线加速度和角速度初始值的更新
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * @brief   处理图像特征数据
 * @Description addFeatureCheckParallax()添加特征点到feature中，计算点跟踪的次数和视差，判断是否是关键帧               
 *              判断并进行外参标定
 *              进行视觉惯性联合初始化或基于滑动窗口非线性优化的紧耦合VIO
 * @param[in]   image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   header 某帧图像的头信息
 * @return  void
*/
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());

    //通过检测两帧之间的视差决定次新帧是否作为关键帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td)) //添加之前检测到的特征点到feature容器中，计算每一个点跟踪的次数，以及它的视差
        marginalization_flag = MARGIN_OLD; //=0是的话就边缘化最后一帧
    else
        marginalization_flag = MARGIN_SECOND_NEW; //=1 是的话就边缘化第二新的帧

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    //填充imageframe的容器以及更新临时预积分初始值
    ImageFrame imageframe(image, header.stamp.toSec()); //ImageFrame类包括特征点、时间、位姿Rt、预积分对象pre_integration、是否关键帧
    imageframe.pre_integration = tmp_pre_integration; //tmp_pre_integration是之前IMU 预积分计算的
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe)); //map<double, ImageFrame> all_image_frame;

    //更新临时预积分初始值
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    //如果没有外参则标定IMU到相机的外参
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            //得到两帧之间归一化特征点
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;

            //标定从camera到IMU之间的外参数
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1; //完成外参标定
            }
        }
    }

    //初始化
    if (solver_flag == INITIAL)
    {
        //frame_count是滑动窗口中图像帧的数量，一开始初始化为0，滑动窗口总帧数WINDOW_SIZE=10
        //确保有足够的frame参与初始化
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;

            //有外参且当前帧时间戳大于初始化时间戳0.1秒，就进行初始化操作
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
               //视觉惯性联合初始化
               result = initialStructure();

               //更新初始化时间戳
               initial_timestamp = header.stamp.toSec();
            }

            //初始化成功
            if(result)
            {
                //先进行一次滑动窗口非线性优化，得到当前帧与第一帧的位姿
                solver_flag = NON_LINEAR; //初始化更改为非线性
                solveOdometry(); //非线性化求解里程计
                slideWindow(); //滑动窗
                f_manager.removeFailures(); //除feature中估计失败的点（solve_flag == 2）0 haven't solve yet; 1 solve succ; 2 solve fail;
                ROS_INFO("Initialization finish!");

                //初始化窗口中PVQ
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                
            }
            else
                slideWindow(); //初始化失败则直接滑动窗口
        }
        else
            frame_count++; //图像帧数量+1
    }
    else //紧耦合的非线性优化
    {
        TicToc t_solve;
        solveOdometry(); //非线性化求解里程计
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        //故障检测与恢复,一旦检测到故障，系统将切换回初始化阶段
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState(); //清空状态
            setParameter(); //重设参数
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow(); //滑动窗口
        f_manager.removeFailures(); //移除失败的
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]); //关键位姿的位置 Ps

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

//多传感器融合过程中，当单个传感器数据不确定性较高，需要依赖其他传感器降低不确定性。先对纯视觉SFM初始化相机位姿，再和IMU对齐
//主要分为1、纯视觉SFM估计滑动窗内相机位姿和路标点逆深度。
    //（1）选择一个滑动窗，在最后一帧与滑动窗之前帧：跟踪到的点数目大于30个的并且视差超过20的，找到后用5点法本质矩阵初始化恢复出R和t。否则滑动窗内保留最新图像帧，继续等待下一帧
    //（2）随意设置一个尺度因子，三角化这两帧观测到的所有路标点。再用PnP算法估计滑动窗内所有其余帧的位姿。滑动窗内全局BA重投影误差优化所有帧位姿。
    //（3）假设IMU-Camera外参已知，乘上视觉得到的位姿，转换到IMU坐标系下。

//2、视觉惯性联合校准，SFM与IMU积分对齐（松耦合），继而恢复尺度、重力、imu速度、陀螺仪偏置
    //（1）陀螺仪零偏标定
    //（2）IMU速度v、重力g、尺度s初始化
    //（3）重力矢量修正（优化方向）
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    //通过加速度标准差判断IMU是否有充分运动以初始化。
    //求解标准差的过程需要先求解均值，再求每个值和均值的差，最后需要判断加速度标准差大于0.25即可满足imu充分激励，可以初始化。
    {
        map<double, ImageFrame>::iterator frame_it; //迭代器（时间戳、ImageFrame)  ImageFrame(特征点ID、相机ID、(x,y,z,u,v,vx,vy))

        //先求均值 aver_g  其实也就是平均加速度
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt; //预计分中的时间
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt; //a=v/t
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); //总帧数减1，因为all_image_frame第一帧不算

        //再求标准差var,表示线加速度的标准差
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g); 
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1)); //加速度的标准差
        //ROS_WARN("IMU variation %f!", var);
        
        //判断，加速度标准差大于0.25则代表imu充分激励，足够初始化
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1]; //旋转四元数Q
    Vector3d T[frame_count + 1]; //平移矩阵T
    map<int, Vector3d> sfm_tracked_points; //存储SFM重建出特征点的坐标
    vector<SFMFeature> sfm_f; //SFMFeature三角化状态、特征点索引、平面观测、位置坐标、深度

    //将f_manager中的所有feature保存到存有SFMFeature对象的sfm_f中
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1; //第一次观测到特征点的帧数-1
        SFMFeature tmp_feature;
        tmp_feature.state = false; //状态state
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame) //能够观测到某个特征点的所有相关帧
        {
            imu_j++; //特征点ID
            Vector3d pts_j = it_per_frame.point;
            //所有观测到该特征点的图像帧ID和图像坐标
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }

    //对窗口中每个图像帧求解sfm问题
    GlobalSFM sfm;
    //得到所有图像帧相对于参考帧l的姿态四元数Q、平移向量T和特征点坐标sfm_tracked_points
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        //解失败则边缘化最早一帧并滑动窗口
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    //对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计，然后solvePnP进行优化,得到每一帧的姿态
    //由于并不是第一次视觉初始化就能成功，此时图像帧数目有可能会超过滑动窗口的大小
    // 所以再视觉初始化的最后，要求出滑动窗口外的帧的位姿
    // 最后把世界坐标系从帧l的相机坐标系，转到帧l的IMU坐标系
    // 对于非滑动窗口的所有帧，提供一个初始的R,T，然后solve pnp求解pose
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;

        //初始化估计值
        if((frame_it->first) == Headers[i].stamp.toSec()) //all_image_frame与滑动窗口中对应的帧
        {
            frame_it->second.is_key_frame = true; //滑动窗口中所有帧都是关键帧
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose(); //相机旋转矩阵（第i帧相对于第l帧的旋转）*第0帧的imu到camera系的旋转矩阵 
                                                                               //根据各帧相机坐标系的姿态和外参，得到用各帧IMU坐标系的姿态（对应VINS Mono论文(2018年的期刊版论文)中的公式（6））
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }

        //为滑动窗口外的帧提供一个初始位姿
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i]; 

        //德里格斯公式将旋转矩阵转换成旋转向量
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false; //初始化时位于滑动窗口外的帧是非关键帧
        vector<cv::Point3f> pts_3_vector; //用于pnp解算的3D点
        vector<cv::Point2f> pts_2_vector; //用于pnp解算的2D点
        for (auto &id_pts : frame_it->second.points) //对于该帧中的特征点
        {
            int feature_id = id_pts.first; //特征点id
            for (auto &i_p : id_pts.second) //由于可能有多个相机，所以需要遍历。i_p对应着一个相机所拍图像帧的特征点信息
            {
                it = sfm_tracked_points.find(feature_id);

                // 如果it不是尾部迭代器，说明在sfm_tracked_points中找到了相应的3D点
                if(it != sfm_tracked_points.end())
                {
                    //记录该id特征点的3D位置
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);

                    //记录该id的特征点在该帧图像中的2D位置
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     

        //如果匹配到的3D点数量少于6个，则认为初始化失败
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }

        //pnp求解失败
        /** 
         *bool cv::solvePnP(    求解pnp问题
         *   InputArray  objectPoints,   特征点的3D坐标数组
         *   InputArray  imagePoints,    特征点对应的图像坐标
         *   InputArray  cameraMatrix,   相机内参矩阵
         *   InputArray  distCoeffs,     失真系数的输入向量
         *   OutputArray     rvec,       旋转向量
         *   OutputArray     tvec,       平移向量
         *   bool    useExtrinsicGuess = false, 为真则使用提供的初始估计值
         *   int     flags = SOLVEPNP_ITERATIVE 采用LM优化
         *)   
         */
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }

        //pnp求解成功
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp; //tmp_R_pnp 是将各帧变换到第l帧
        cv::cv2eigen(r, tmp_R_pnp);
        //将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
        R_pnp = tmp_R_pnp.transpose(); //R_pnp 是将第l帧变换到各帧
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose(); //RIC[0] 第l帧camera到imu    根据各帧相机坐标系的姿态和外参，得到用各帧IMU坐标系的姿态。
        frame_it->second.T = T_pnp;
    }

    ////camera与IMU对齐
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

/**
 * @brief   视觉惯性联合初始化
 * @Description 陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
 *              更新了Bgs后，IMU测量量需要repropagate  
 *              得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 * @return  bool true：成功
 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale 1.视觉惯性联合初始化，先计算陀螺仪的偏置，再尺度、重力加速度和速度
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x); //优化量x
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state 2.获取滑动窗口内所有图像帧相对于第l帧的位姿信息，并设置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    //3.获取特征点深度  将所有特征点的深度置为-1
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep); //设为-1

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();

    //RIC中存放的是相机到IMU的旋转，在相机-IMU外参标定部分求得
    ric[0] = RIC[0];
    f_manager.setRic(ric);

    //三角化计算地图点的深度，Ps中存放的是各个帧相对于参考帧之间的平移，RIC[0]为相机-IMU之间的旋转
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);

    //4.陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }

    //5.计算各帧相对于b0的位姿信息，前边计算的都是相对于第l帧的位姿
    /**
     * 前面初始化中，计算出来的是相对滑动窗口中第l帧的位姿，在这里转换到第b0帧坐标系下
     * s*p_bk^​b0​​=s*p_bk^​cl​​−s*p_b0^​cl​​=(s*p_ck^​cl​​−R_bk​^cl​​*p_c^b​)−(s*p_c0^​cl​​−R_b0​^cl​​*p_c^b​)  //p_bk^​b0代表bk到b0的平移  R_bk​^cl代表cl到bk的旋转
     * TIC[0]是相机到IMU的平移量
     * Rs是IMU第k帧到滑动窗口中图像第l帧的旋转
     * Ps是滑动窗口中第k帧到第l帧的平移量
     * 注意：如果运行的脚本是配置文件中无外参的脚本，那么这里的TIC都是0
    */
    //将Ps、Vs、depth尺度s缩放
    for (int i = frame_count; i >= 0; i--)
        //Ps转变为第i帧imu坐标系到第0帧imu坐标系的变换
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]); //Rs是IMU第k帧到滑动窗口中图像第l帧的旋转  Ps是滑动窗口中第k帧到第l帧的平移量
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            ////Vs为优化得到的速度
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    //更新每个地图点被观测到的帧数(used_num)和预测的深度(estimated_depth)
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    /**
     * refine之后就获得了C_0坐标系下的重力g^{c_0}，此时通过将g^{c_0}旋转至z轴方向，
     * 这样就可以计算相机系到世界坐标系的旋转矩阵q_{c_0}^w，这里求得的是rot_diff,这样就可以将所有变量调整至世界系中。
    */
    ////通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g; //R0是c0到world系
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    //所有变量从参考坐标系c_0旋转到世界坐标系w
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i]; //Ps[i]是第i帧相对于第l帧的平移，也就是ci相对于c0的平移
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

/**
 * @brief   返回滑动窗口中第一个满足视差的帧，为l帧，以及RT,可以三角化。
 * @Description    判断每帧到窗口最后一帧对应特征点的平均视差是否大于30
                solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
 * @param[out]   relative_R 当前帧到第l帧之间的旋转矩阵R
 * @param[out]   relative_T 当前帧到第l帧之间的平移向量T
 * @param[out]   L 从第一帧开始到滑动窗口中第一个满足视差足够的帧，这里的l帧之后作为参考帧做全局SFM用
 * @return  bool 1:可以进行初始化;0:不满足初始化条件
*/
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    //寻找第i帧到窗口最后一帧的对应特征点
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE); //得到第i帧和最后一帧的特征匹配,得到给定两帧之间的对应特征点3D坐标
        
        //首先corres数目足够
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                //第j个对应点在第i帧和最后一帧的(x,y)
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm(); //norm()范数，即模长
                sum_parallax = sum_parallax + parallax;
            }

            average_parallax = 1.0 * sum_parallax / int(corres.size());

            //判断是否满足初始化条件：视差>30和内点数满足要求
            //同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的Rt
            //找到的第l帧作为参考帧，并通过solveRelativeRT(corres, relative_R, relative_T)计算当前帧和第l帧的相对位姿。
            //这里的relative_R,relative_T是最后一帧到 l 帧坐标系的R,T。是第l帧到最后一帧的逆，见slam十四讲公式3.14
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

////三角化求解所有特征点的深度，并进行非线性优化
void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization();
    }
}

/* 
从Ps、Rs、Vs、Bas、Bgs转化为para_Pose（6维，相机位姿）和para_SpeedBias（9维，相机速度、加速度偏置、角速度偏置）
//vector转换成double数组，因为ceres使用数值数组
//Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
从tic和q转化为para_Ex_Pose （6维，Cam到IMU外参）
从dep到para_Feature（1维，特征点深度）
从td转化为para_Td（1维，标定同步时间）
*/
void Estimator::vector2double()
{
    //遍历滑动窗，IMU的15个自由度的优化变量
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    
    //Cam到IMU的外参，6自由度由7个参数表示
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        //深度
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        //标定同步时间
        para_Td[0][0] = td;
}

////数据转换，vector2double的相反过程
void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}

////系统故障检测 -> Paper VI-G
bool Estimator::failureDetection()
{
    //在最新帧中跟踪的特征数小于某一阈值
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    
    ////偏置或外部参数估计有较大的变化
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */

   //最近两个估计器输出之间的位置或旋转有较大的不连续性
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

/**
 * @brief   基于滑动窗口紧耦合的非线性优化，残差项的构造和求解
 * @Description 添加要优化的变量 (p,v,q,ba,bg) 一共15个自由度，IMU的外参也可以加进来
 *              添加残差，残差项分为4块 先验残差+IMU残差+视觉残差+闭环检测残差
 *              根据倒数第二帧是不是关键帧确定边缘化的结果           
 * @return      void
*/
void Estimator::optimization()
{
    //创建一个ceres Problem实例, loss_function定义为CauchyLoss.
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    //先添加优化参数量, ceres中参数用ParameterBlock来表示,类似于g2o中的vertex, 因为ceres用的是double数组，所以在下面用vector2double做类型装换。
    //这里的参数块有sliding windows中所有帧的para_Pose(7维) 和 para_SpeedBias(9维). 
    /*add vertex of: 1)pose, 2)speed and 3)bias of acc and gyro */
    //添加要优化的变量：相机位姿、速度、加速度偏差、陀螺仪偏差
    //Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization(); //这个参数是告诉求解器这个是个单元四元数
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization); //para_Pose[i]中存放的是滑动窗口中第i帧的位姿，para_Pose[i]的大小为SIZE_POSE(值为7)
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS); //SIZE_SPEEDBIAS值为9
    }
    /*add vertex of: camera extrinsic */
    //添加要优化的变量：相机到IMU的外参
    //ESTIMATE_EXTRINSIC!=0则camera到IMU的外参也添加到估计
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC) //外参如果确定了就不再做优化，如果不确定就需要进一步优化
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }

    //添加要优化的变量：时间偏差（相机和IMU硬件不同步时估计两者的时间偏差）
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    //将要优化的变量转为数组形式
    vector2double();

    //添加marginalization的residual，这个marginalization的结构是始终存在的，随着下面marginazation的结构更新，last_marginalization_parameter_blocks对应的是还在sliding window的变量
    //关于这一部分的理解，请看heyijia大神的BLOG:http://blog.csdn.net/heyijia0327/article/details/53707261,可能是说的最清楚的一个了
    //这里可以这样理解，下面会添加对IMU和视觉的残差，但是，这些对应的变量实际上跟之前被margin掉的变量是有约束的，这里的last_marginalization_parameter_blocks就是保存的这些变量，也就是heyijia博客中对应的Xb变量，
    //last_marginalization_info中对应的是Xb对应的测量Zb，这里用先验来表示这个约束，整个margin部分实际上就是在维护这个结构

    //依次加入margin项,IMU项和视觉feature项. 每一项都是一个factor, 这是ceres的使用方法, 
    //创建一个类继承ceres::CostFunction类, 重写Evaluate()函数定义residual的计算形式. 
    //分别对应marginalization_factor.h, imu_factor.h, projection_factor.h中的MarginalizationInfo, IMUFactor, ProjectionFactor三个类. 
    ////添加边缘化残差   添加边缘化的先验残差信息，第一次进行优化的时候last_marginalization_info还为空值
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    //这里IMU项和camera项之间是有一个系数，这个系数就是他们各自的协方差矩阵：
    //IMU的协方差是预积分的协方差(IMUFactor::Evaluate，中添加IMU协方差，求解jacobian矩阵)，而camera的测量残差则是一个固定的系数（f/1.5）
    //添加IMU残差
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]); //这里面会计算残差以及残差对优化变量雅克比矩阵
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]); //指定了相关的优化变量
    }

    //添加视觉残差
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature) //遍历滑窗内所有的空间点
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        //第一个观测到该特征的帧对应的特征点坐标
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        //遍历能观测到该特征的每个帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            //是否估计时间同步
            if (ESTIMATE_TD)
            {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j); //会计算残差以及残差对优化变量雅克比矩阵
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]); //定了相关的优化变量
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    ////添加闭环检测残差，计算滑动窗口中与每一个闭环关键帧的相对位姿，这个相对位置是为后面的图优化准备
    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;

            //获取观测到该特征的起始帧
            int start = it_per_id.start_frame;
            if(start <= relo_frame_local_index)
            {   
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }     
            }
        }

    }

    //创建优化求解器
    ceres::Solver::Options options;

    //设定优化器的solver_type类型
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;

    //设定优化使用的算法。这里使用置信域类优化算法中的dogleg方法
    options.trust_region_strategy_type = ceres::DOGLEG;

    //设置迭代求解的次数，这个在配置文件中设置
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary; //优化信息
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    //求解完成后，将数组转化为向量
    double2vector();

    TicToc t_whole_marginalization;

    //margin部分，如果倒数第二帧是关键帧：如果次新帧是关键帧，将边缘化最老帧，及其看到的路标点和IMU数据，将其转化为先验： 
    //1.把之前的存的残差部分加进来
    //2.把与当前要margin掉帧所有相关的残差项都加进来，IMU,vision
    //3.preMarginalize-> 调用Evaluate计算所有ResidualBlock的残差，parameter_block_data parameter_block_idx parameter_block_size是marinazation中存参数块的容器(unordered_map),key都是addr,
    //分别对应这些参数的data，在稀疏矩阵A中的index(要被margin掉的参数会被移到前面)，A中的大小
    //4.Marginalize->多线程构造Hx=b的结构，H是边缘化后的结果，First Esitimate Jacobian,在X0处线性化
    //5.margin结束，调整参数块在下一次window中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，这里只是提前占座（知乎上有人问：https://www.zhihu.com/question/63754583/answer/259699612）
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        //将上一次先验残差项传递给marginalization_info
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) //last_marginalization_parameter_blocks是上一轮留下来的残差块
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0]) //需要marg掉的优化变量，也就是滑窗内第一个变量
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        //将第0帧和第1帧间的IMU因子IMUFactor(pre_integrations[1])，添加到marginalization_info中
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]}, //优化变量
                                                                           vector<int>{0, 1}); //这里是0,1的原因是0和1是para_Pose[0], para_SpeedBias[0]是需要marg的变量
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        //将第一次观测为第0帧的所有路标点对应的视觉观测，添加到marginalization_info中
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1; //从特征点的第一个观察帧开始
                if (imu_i != 0) //如果第一个观察帧不是第一帧就不进行考虑，因此后面用来构建marg矩阵的都是和第一帧有共视关系的滑窗帧
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3}); //为0和3的原因是，para_Pose[imu_i]是第一帧的位姿，需要marg掉，而3是para_Feature[feature_index]是和第一帧相关的特征点，需要marg掉
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        //计算每个残差，对应的Jacobian，并将各参数块拷贝到统一的内存（parameter_block_data）中
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        //多线程构造先验项舒尔补AX=b的结构，在X0处线性化计算Jacobian和残差
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        //调整参数块在下一次窗口中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，这里只是提前占座
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++) //从1开始，因为第一帧的状态不要了
        {
            //这一步的操作指的是第i的位置存放的的是i-1的内容，这就意味着窗口向前移动了一格
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1]; //因此para_Pose这些变量都是双指针变量，因此这一步是指针操作
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info; //删除掉上一次的marg相关的内容
        //更新last_marginalization_info
        last_marginalization_info = marginalization_info; //marg相关内容的递归
        last_marginalization_parameter_blocks = parameter_blocks; //优化变量的递归，这里面仅仅是指针

        
    }
    else
    {
        /**
         * 当次新帧不是关键帧时，直接剪切掉次新帧和它的视觉观测边（该帧和路标点之间的关联），而不对次新帧进行marginalize处理
         * 但是要保留次新帧的IMU数据，从而保证IMU预积分的连续性，这样才能积分计算出下一帧的测量值
         * */
        //如果倒数第二帧不是关键帧
        //1.保留该帧的IMU测量，margin该帧的visual
        //2.premargin
        //3.marginalize
        //4.滑动窗口移动（去掉倒数第二个）
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            //保留次新帧的IMU测量，丢弃该帧的视觉测量，将上一次先验残差项传递给marginalization_info
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");

            //premargin 计算每次IMU和视觉观测(cost_function)对应的参数块(parameter_blocks),雅可比矩阵(jacobians),残差值(residuals)
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");

            //marginalize  多线程计算整个先验项的参数块,雅可比矩阵和残差值
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            //调整参数块在下一次窗口中对应的位置（去掉次新帧）
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

/**
 * @brief   实现滑动窗口all_image_frame的函数
 * @Description 如果次新帧是关键帧，则边缘化最老帧，将其看到的特征点和IMU数据转化为先验信息
                如果次新帧不是关键帧，则舍弃视觉测量而保留IMU测量值，从而保证IMU预积分的连贯性
 * @return      void
*/
////实际滑动窗口的地方，如果第二最新帧是关键帧的话，那么这个关键帧就会留在滑动窗口中，时间最长的一帧和其测量值就会被边缘化掉
//如果第二最新帧不是关键帧的话，则把这帧的视觉测量舍弃掉而保留IMU测量值在滑动窗口中这样的策略会保证系统的稀疏性
void Estimator::slideWindow()
{
    TicToc t_margin;
    
    //从滑动窗口中删除掉最旧的帧
    if (marginalization_flag == MARGIN_OLD)
    {
        //获取第一帧对应的时间戳
        double t_0 = Headers[0].stamp.toSec();
        //获得滑动窗口中第一帧图像对应的旋转
        back_R0 = Rs[0];
        //获得滑动窗口中第一帧图像对应的位置
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            //交换滑动窗口中的位姿、速度、陀螺仪偏差、加速度buf、角速度buf等的位置
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }//将所有的值都转移到前一帧的值上面去

            /**
             * 执行完上边的for循环后，滑动窗口中最旧的帧已经被交换到对应状态数组的最后位置了
             * 下面将WINDOW_SIZE-1上的值赋值给WINDOW_SIZE，相当于删掉了最旧的帧的数据
            */
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE]; //删除掉pre_integrations数组中最后一个元素，相当于删除了最旧帧对应的预积分
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                //在all_image_frame中找到第一帧图像对应的数据
                it_0 = all_image_frame.find(t_0);
                //删除第一帧图像对应的预积分
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
 
                //删除all_image_frame当中第一帧之外的其他帧对应的预积分
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                //从all_image_frame这个map中删除掉t_0项元素
                all_image_frame.erase(t_0);

            }
            slideWindowOld();
        }
    }
    else
    {
        //边缘化掉滑动窗口中次新的帧
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                //最新帧的时间、加速度和角速度
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity); //将最新帧的时间间隔、加速度、角度加入次新帧的预积分中

                //将最新帧的时间间隔、加速度和角速度等存入次新帧
                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            //用滑动窗口中最新的帧的状态替换掉次新的帧的状态
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            //删除掉滑动窗口中最后一个预积分
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
//滑动窗口边缘化次新帧时处理特征点被观测的帧号
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
////滑动窗口边缘化最老帧时处理特征点被观测的帧
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        //back_R0、back_P0为窗口中最老帧的位姿
        //Rs、Ps为滑动窗口后第0帧的位姿，即原来的第1帧
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1); ////如果这个特征点的开始帧不是最老帧，则特征点所有帧数减一，如果是，则通过三维变换变换到下一帧上去
    }
    else
        f_manager.removeBack();
}

/**
 * @brief   进行重定位
 * @optional    
 * @param[in]   _frame_stamp    重定位帧时间戳
 * @param[in]   _frame_index    重定位帧索引值
 * @param[in]   _match_points   重定位帧的所有匹配点
 * @param[in]   _relo_t     重定位帧平移向量
 * @param[in]   _relo_r     重定位帧旋转矩阵
 * @return      void
*/
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for(int i = 0; i < WINDOW_SIZE; i++) //在当前窗口内寻找重定位帧，因为此时这个线程里面的当前帧不一定是闭环帧了
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

