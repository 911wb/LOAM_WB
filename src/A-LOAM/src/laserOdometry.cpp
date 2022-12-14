// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"

#define DISTORTION 0


int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum = 5;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;
// 创建KD-Tree对象
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame
// 全局变量，被不断更新
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex mBuf;

/*
计算当前位置到起始位置的坐标变换，将当前的点转换到起始点位置，
即将100毫秒内的一帧的点云，统一到一个时间点的坐标系上
P_start = T_curr2start * P_curr
如何获取 T_curr2start:
1.如果有高频里程计，可以方便的获取每个点相对于起始扫描的位姿
2.如果有imu，可以方便的求出每个点对起始点的旋转
3.如果没有其他里程计，可以使用匀速运动模型，使用上一个帧间里程计的结果作为当前两帧之间的
运动，同时假设当前帧也是匀速运动，也可以估计出每个点相对于起始时刻的位姿

使用激光点云的强度构建图像，根据两帧数据图像之间的位移估计地面车辆的线速度和角速度
有了速度之后，再根据两帧之间的时间间隔，乘以速度，就可以得到两帧的位移增量和速度增量
在加上上一时刻的后验位姿就可以得到当前

k-1 到 k 帧 和 k到k+1帧的运动是一至的,用k-1到k帧的位姿变换当做k到k+1帧的位姿变换, 可以求到k到k+1帧的每个点的位姿变换

*/

// undistort lidar point
// 把所有的点补偿到起始时刻，输出的是一个带有强度信息的三维点
void TransformToStart(PointType const *const pi, PointType *const po)
{
    // interpolation ratio
    double s;
    // 由于kitti数据集上的lidar已经做过运动补偿，因此这里就不做具体补偿了DISTORTION = false
    if (DISTORTION)
        // 相对与起始时刻的时间差/100ms = 比例
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;// s = 1s说明全部补偿到点云结束的时刻
    
    // s = 1 ->  T_curr2start =  T_end2start;
    // 做一个匀速模型假设,即上一帧的位姿变换,就是这帧的位姿变换
    // 以此来求出输入点坐标系到起始时刻点坐标系的位姿变换,通过上面求的时间占比s，这里把位姿变换 分解成了 旋转 + 平移 的方式
    // 由于四元数是一个超复数,不是简单的乘法,求它的占比用的 Eigen的slerp函数(球面线性插值)

    // R_curr2start  = R_end2start * 差值比例为s
    // 根据相对时间内插出某点i相对于起始时刻的位姿变换
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    // 平移增量 * s
    Eigen::Vector3d t_point_last = s * t_last_curr;
    // 输入点，即未去畸变的点
    Eigen::Vector3d point(pi->x, pi->y, pi->z);

    // 当前点转到帧起始时刻坐标系下的坐标：P_start = T_curr2start * P_curr
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

// transform all lidar points to the start of the next frame
// 把所有的点补偿到结束时刻
void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    pcl::PointXYZI un_point_tmp;
    // 把所有的点补偿到起始时刻
    TransformToStart(pi, &un_point_tmp);
    // 保存去畸变后的点 P_start
    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    /*
    把所有的点补偿到结束时刻:
    P_end * R_end2start + t_end2start = p_start
    P_end * R_curr2last + t_curr2last = p_start
    */
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;
    // 往后端发送数据的频率
    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    printf("Mapping %d Hz \n", 10 / skipFrameNum);

    // 订阅曲率大的角点，并传入回调函数中处理，消息队列的长度为100
    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);
    // 订阅曲率较大的角点
    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);
    // 订阅平坦的面点
    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);
    // 订阅较平坦的面点
    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);
    // 当前点云
    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);


    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);
    // 发布里程计数据(位姿轨迹)给后端，后端接收
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    // 发布前端里程计的高频低精 位姿轨迹
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    nav_msgs::Path laserPath;

    int frameCount = 0;

    // 100hz
    ros::Rate rate(100);
    
    // 不断循环直到ctrl c
    while (ros::ok())
    {
        /*
        ros::spin() 和 ros::spinOnce() 区别及详解:
            其实看函数名也能理解个差不多，一个是一直调用回调函数；
            ros::spin() 在调用后不会再返回，也就是你的主程序到这儿就不往下执行了。
            另一个是只调用一次回调函数，如果还想再调用，就需要加上循环了，后者在调用后还可以继续执行之后的程序

        Subscriber每接收到一个消息(一帧)，就会触发一次spinOnce，其滴调用回调函数
        如果spinOnce下面的代码没有执行完，就不会触发下一次回调，就不会丢帧

        对于有些传输特别快的消息，尤其需要注意合理控制消息池大小和ros::spinOnce()执行频率; 
        比如消息送达频率为10Hz, ros::spinOnce()的调用回调函数频率为5Hz，那么消息池的大小就一定要大于2，才能保证数据不丢失，无延迟。
        */

        ros::spinOnce();// 如果改为ros::spin则下面的代码就不会被执行

        // 确保订阅的五个消息队列都有，有一个队列为空都不行
        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
        {
            // 取出每一个队列最早的时间戳
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();
            // 判断是否是同一帧，同一帧的时间戳相同
            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync messeage!");
                ROS_BREAK();
            }

            mBuf.lock();
            // 每次进来前，都要记得清除
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            cornerSharpBuf.pop();

            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();

            TicToc t_whole;
            // initializing，跳过第一帧
            if (!systemInited)
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            else
            {
                // 一帧中，曲率大的点的个数
                int cornerPointsSharpNum = cornerPointsSharp->points.size();
                // 一帧中，平坦的点的个数
                int surfPointsFlatNum = surfPointsFlat->points.size();
                // 计算优化的时间
                TicToc t_opt;

                // 点到线以及点到面的非线性优化，迭代2次（选择当前优化位姿的特征点匹配，并优化位姿（4次迭代），然后重新选择特征点匹配并优化）
                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
                {
                    corner_correspondence = 0;
                    plane_correspondence = 0;

                    //ceres::LossFunction *loss_function = NULL;
                    // 定义一下ceres的核函数，使用Huber核函数来减少外点的影响，即去除outliers
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    // 由于旋转不满足一般意义的加法，因此这里使用ceres自带的local param
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                    
                    ceres::Problem::Options problem_options;
                    // 实例化求解最优化问题
                    ceres::Problem problem(problem_options);
                    // 添加待优化变量q，到参数块，维度为4
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    // 添加待优化变量t，到参数块，维度为3
                    problem.AddParameterBlock(para_t, 3);

                    // 去运动畸变后的角点
                    pcl::PointXYZI pointSel;
                    // 找到的点的ID
                    std::vector<int> pointSearchInd;
                    // 当前点到找到点的距离
                    std::vector<float> pointSearchSqDis;
                    // 计算寻找关联点的时间
                    TicToc t_data;
                    // find correspondence for corner features
                    /*
                    基于最近邻原理建立corner特征点（边线点）之间的关联，每一个极大边线点去上一帧的次极大边线
                    点中找匹配；采用边线点匹配方法:假如在第k+1帧中发现了边线点i，通过KD-tree查询在第k帧中的最
                    近邻点j，查询j的附近扫描线上的最近邻点l，j与l相连形成一条直线l-j，让点i与这条直线的距离最短。
                    构建一个非线性优化问题：以点i与直线lj的距离为代价函数，以位姿变换T(四元数表示旋转+位移t)为优化变量。

                    */
                    // 寻找角点约束，遍历所有曲率大的角点 先进行线点的匹配
                    for (int i = 0; i < cornerPointsSharpNum; ++i)
                    {
                        // 运动畸变，输出去畸变后的角点，也就是运动补偿到起始时刻的角点
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
                        /*  
                        brief Search for k-nearest neighbors for the given query point.  //搜素给定点的K近邻。
                            参数1： 给定的查询点。
                            参数2： 要搜索的近邻点的数量
                            参数3： 输出的k个近邻点索引
                            参数4： 输出查询点到邻近点的平方距离
                            返回值：返回找到的近邻点的数量
                        在上一帧所有角点(弱角点)构成的kdtree中寻找距离当前帧最近的一个点   因为前面有初始化的判断 所有 第二帧肯定有上一帧
                        */
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1;
                        // 帧间的距离不能过大，因为是匀速运动模型
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            // 目标点对应的上一帧最近距离点的索引
                            closestPointInd = pointSearchInd[0];
                            // 找到其所在线束id，线束信息藏在intensity的整数部分
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                            // search in the direction of increasing scan line
                            // 寻找第二个角点，在刚刚角点id上下分别继续寻找，目的是寻找最近的角点，由于其按照线束排列，所以就是向上寻找
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                // if in the same scan line, continue
                                // 第二个次近点与第一个最近点不在同一根线束的
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                // 要求找到的线束距离第一个点的线束不能太远，如果当前的点不满足，后面的点更不满足
                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                // 计算当前找到的角点之间的距离，上一帧的点到当前帧的点的距离
                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);
                                
                                // 寻找距离最小的角点及其索引
                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point，第二个最近点
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }

                        // 如果这个点是有效的角点
                        // 如果特征点i的两个最近邻点j和m都有效，构建非线性优化问题
                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                            // 取出当前点和上一帧的两个角点a、b，a、b不能在同一个scan下
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            // 最近的点a
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            // 次近点b
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);

                            double s;
                            if (DISTORTION)
                                // Δt/T
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                            else
                                s = 1.0;

                            // notice: 构建点到线的约束，残差
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            // 添加残差到约束问题，残差项、损失函数、具体待优化变量
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            corner_correspondence++;
                        }
                    }

                    /*
                    notice: 下面采用平面点匹配方法：假如在第k+1帧中发现了平面点i，通过KD-tree查询在第k帧（上一帧）中
                    的最近邻点j，查询j的附近扫描线上的最近邻点l和同一条扫描线的最近邻点m，这三点确定一个平面，
                    让点i与这个平面的距离最短；
                    构建一个非线性优化问题：以点i与平面lmj的距离为代价函数，以位姿变换T(四元数表示旋转+t)为优化变量。

                    找面的第一个点与找线的第一个点一致
                    第二个点与第一个点在一条scan上，第三个点与其他两个点不能在同一条线上
                         .        
                        / \     .
                    -----.---.-----
                    */
                    // find correspondence for plane features
                    for (int i = 0; i < surfPointsFlatNum; ++i)
                    {
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                        // 先寻找上一帧距离这个面点最近的面点
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                        // 距离必须小于给定阈值
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            // 取出找到的上一帧面点的索引
                            closestPointInd = pointSearchInd[0];

                            // get closest point's scan ID
                            // 取出最近的面点在上一帧的第几根scan上面
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                            // search in the direction of increasing scan line
                            // 额外再寻找 两个点，要求一个点和最近点同一个scan，另一个是不同scan，先升序遍历搜索点寻找这些点
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                // if not in nearby scans, end the loop
                                // 不能和当前找到的上一帧面点线束距离太远
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;
                                // 计算和当前帧该点距离
                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or lower scan line
                                // 如果是同一根scan且距离最近
                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    // 第二个点
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                // if in the higher scan line
                                // 如果是其他线束点
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // 第三个点
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            // 同样的方式，按照降序方向寻找这两个点
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                // 如果是同一scan上的点，并且求出的距离比前面的更小
                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // 如果另外找到的两个点是有效点，就取出他们的3d坐标
                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {

                                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                            surfPointsFlat->points[i].y,
                                                            surfPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                                laserCloudSurfLast->points[closestPointInd].y,
                                                                laserCloudSurfLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                                laserCloudSurfLast->points[minPointInd2].y,
                                                                laserCloudSurfLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                                laserCloudSurfLast->points[minPointInd3].y,
                                                                laserCloudSurfLast->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)//去运动畸变，这里没有做，kitii数据已经做了
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;

                                // 构建点到面的约束，构建cere的非线性优化问题。
                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                plane_correspondence++;
                            }
                        }
                    }// 面点约束

                    // 输出寻找关联点消耗的时间
                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    // 如果总的线约束和面约束太少，就打印一下 
                    printf("data association time %f ms \n", t_data.toc());
                    
                    // 如果总的约束小于10，就可能有问题
                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence! *************************************************\n");
                    }

                    // 调用ceres求解器求解 ，设定求解器类型，最大迭代次数，不输出过程信息，优化报告存入summary
                    TicToc t_solver;
                    ceres::Solver::Options options;
                    // QR分解
                    options.linear_solver_type = ceres::DENSE_QR;
                    // 前端需要保证实时，不能迭代太久
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    printf("solver time %f ms \n", t_solver.toc());
                }

                // 经过两次 LM仅位姿优化 消耗的时间
                printf("optimization twice time %f \n", t_opt.toc());
                // 这里的w_curr 实际上是 w_last，即上一帧

                // notice: 更新帧间匹配的结果，得到lidar odom位姿
                
                /*
                           Tw1   T12   T23   T34
                T_w_curr o-----o-----o-----o-----o
                         w     1     2     3     4
                */

                // 以下两个全局变量(t_w_curr、q_w_curr，当前坐标系到世界坐标系的坐标变换)被不断更新
                // t_last_curr、q_last_curr，是被LM优化过的位姿
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;
            }

            TicToc t_pub;//计算发布运行时间
            // 发布lidar里程计结果
            // publish odometry
            // 创建nav_msgs::Odometry消息类型，把信息导入，并发布
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            // 以四元数和平移向量发出去
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            // 发布里程计数据(位姿轨迹)，后端接收
            pubLaserOdometry.publish(laserOdometry);

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            // 位姿容器，即轨迹
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/camera_init";
            // 发布位姿轨迹可视化
            pubLaserPath.publish(laserPath);

            // transform corner features and plane features to the scan end point
            if (0)//去畸变，没有调用
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }

            // 位姿估计完毕之后，当前边线点和平面点就变成了上一帧的边线点和平面点，把索引和数量都转移过去
            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;

            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';
            // kdtree设置当前帧，用来下一帧lidar odom使用，把当前帧点云送到KD树，用来下一帧的匹配
            // 向KDTREE中传入数据，即将点云数据设置成KD-Tree结构
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

            // 控制后端节点的执行频率，降频后给后端发送，只有整除时才发布结果
            // 每隔skipFrameNum帧，往后端发送一帧，如果算力不够skipFrameNum就要增大
            if (frameCount % skipFrameNum == 0)
            {
                frameCount = 0;
                // 发布当前角点
                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/camera";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);
                // 发布当前平面点
                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
                
                // 原封不动的转发当前帧点云，后端优化是低频，高精的，需要更多的点加入，约束越多鲁棒性越好
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }

            printf("publication time %f ms \n", t_pub.toc());
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
            if(t_whole.toc() > 100)//里程计超过100ms则有问题，算力不够??
                ROS_WARN("odometry process over 100ms");

            frameCount++;
        }

        // 每秒执行100次，所以执行一次是0.01s；而如果执行完小于0.01s，就sleep一下，直到0.01s再继续走
        rate.sleep();
    }
    return 0;
}