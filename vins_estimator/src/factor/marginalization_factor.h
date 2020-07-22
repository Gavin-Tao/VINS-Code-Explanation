#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4; //线程数

//将不同的损失函数_cost_function以及优化变量_parameter_blocks统一起来再一起添加到marginalization_info中
struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function; //代价函数，观测数据与估计值的差
    ceres::LossFunction *loss_function; //核函数，用来减小Outlier的影响
    std::vector<double *> parameter_blocks; //优化变量数据
    std::vector<int> drop_set; //待边缘化的优化变量id

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians; //矩阵定义时可以使用Dynamic来表示矩阵的行列数为未知
    Eigen::VectorXd residuals; //维度未知  //残差 IMU:15X1 视觉2X1

    int localSize(int size)
    {
        return size == 7 ? 6 : size; //如果size=7，则返回6；否则返回size
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors; 
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    // 添加残差块相关信息
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    // 计算每个残差对应的雅克比，并更新 parameter_block_data
    void preMarginalize();
    //多线程构造先验项舒尔补AX=b的结构，计算Jacobian和残差
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors; //所有观测项 类似图优化中的边 
    int m, n; //m为要边缘化的变量个数，n为要保留下来的变量个数
    std::unordered_map<long, int> parameter_block_size; ////<优化变量内存地址,localSize>  localsize：优化变量的长度
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //<待边缘化的优化变量内存地址,在parameter_block_size中的id>
    std::unordered_map<long, double *> parameter_block_data; //<优化变量内存地址,数据> 数据：优化变量对应的double指针

    //进行边缘化之后保留下来的各个优化变量的长度，各个优化变量在id以各个优化变量对应的double指针类型的数据
    std::vector<int> keep_block_size; //global size 
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    //边缘化之后从信息矩阵H恢复出来雅克比矩阵和残差向量
    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
