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
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;

const int systemDelay = 0;
int systemInitCount = 0;
bool systemInited = false;
int N_SCANS = 0;
float cloudCurvature[400000];
int cloudSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;

double MINIMUM_RANGE = 0.1; 

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{

    // 入参和出参的地址一样
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;
 
    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        // 如果 x² + y² +　z² < th ，则跳过
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }

    // 如果有点被去除了
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    // 有效点云个数
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

// 订阅激光数据，放到回调函数中进行处理
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    // 如果系统没有初始化就等待几帧，丢掉前面几帧，等待系统稳定
    if (!systemInited)
    {
        systemInitCount++;
        // kitti数据集比较好，所有设置延迟为0
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    TicToc t_whole;
    TicToc t_prepare;
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    // pcl不能直接处理ros消息格式，将ros格式的点云数据转换为激光数据
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);

    std::vector<int> indices;
    // 取出点云中的nan点，即无返回的点
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    // 取出距离小于阈值的点，不然会发生鬼影，引用传递(专门为传递对象而生)
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

    // 点云个数
    int cloudSize = laserCloudIn.points.size();
    /*
    公式：
    将点云投影到xoy，其原点到投影点的长度为 R*cos(ω)
    X = R * cos(ω) * sin(α)
    Y = R * cos(ω) * cos(α) 
    Z = R * sin(ω)
    X/Y = sin(α)/cos(α) = tan(α)
    α = arctan(X/Y)

             ^ y
             |
    π        |          0
    ---------|---------->  x
   -π        |
    */
    // 计算起始点和结束点的水平角度，由于激光雷达是顺时针旋转，这里取反就是逆时针坐标系
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    // atan2的取值范围：[-π, π]，这里加上2π是为了保证起始和结束相差2π，即一周，从哪里开始就从那里结束
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;

    // start = -179，end = 179 + 360
    if (endOri - startOri > 3 * M_PI)
    {
        // 去除补偿
        endOri -= 2 * M_PI;
    }
    // end = -179 + 360= 181，start = 179
    else if (endOri - startOri < M_PI)
    {
        // 再次补偿2π
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);

    bool halfPassed = false;
    int count = cloudSize;
    // XYZI
    PointType point;
    // 二维容器，大容器中装有N_SCANS个小容器，每个小容器中装有各各自线束的点云
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    
    // 遍历每一个点
    for (int i = 0; i < cloudSize; i++)
    {
        // 当前点坐标
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;

        // 计算每一个点的俯仰角ω， Z = R * sin(ω) --> ω = arcsin(Z/R)
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;
        
        // notice: 以前每个点对应的线束是未知的，需要计算，现在的雷达驱动可以提前计算出来
        // LIO_SAM可以直接从激光的驱动中获取点云对应的线束
        // 16线的激光雷达的俯仰角的范围是[-15°, 15°]
        if (N_SCANS == 16)
        {
            // 每条线束之间的俯仰角间隔Δω = 2°，根据俯仰角找到点云对应的线束ID，+0.5为了四舍五入
            scanID = int((angle + 15) / 2 + 0.5);
            // 如果点云对应的线束不合理就跳过该点云
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }

        //printf("angle %f scanID %d \n", angle, scanID);
        // 计算该点的水平角
        float ori = -atan2(point.y, point.x);

        // 保证当前水平角度在开始和结束区间之内
        // 保证  startori < ori < endori
        if (!halfPassed)
        {
            // 确保 π/2 < ori - startori < 3/2*π 
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            // 这种case不会发生
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }
            // 超过了圆周的一半
            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            // 确保 π*3/2 < ori - endori < π
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }

         /*
         * relTime 是一个0~1之间的小数，代表占用一帧扫描时间的比例，乘以扫描时间得到真实扫描时刻，
         * scanPeriod扫描时间默认为0.1s
         * 角度的计算是为了计算相对起始时刻的时间
         */
        // 计算当前时刻相对于起始时刻的时间，后面要做运动畸变的补偿
        // 百分比
        float relTime = (ori - startOri) / (endOri - startOri);
        // scanPeriod * relTime = 当前点相对于起始点时刻的时间戳Δt
        // 现在雷达的驱动会计算每个点的时间戳，不需要我们额外计算
        // 整数部分是scan线束的索引，小数部分是相对起始时刻的时间
        point.intensity = scanID + scanPeriod * relTime;
        // 根据每条线的idx送入各自数组，表示这一条扫描线上的点，把每个点存放到对应的线束数组中，后面提取特征的时候是在每一根线上提取的
        laserCloudScans[scanID].push_back(point);
    }

    // 有效点云个数
    cloudSize = count;
    printf("points size %d \n", cloudSize);

    /*
    前面处理了雷达点云数据，下面是前端的雷达特征提取，主要提取了线特征和面特征。
    LOAM提出了一种简单而高效的特征点提取方式，根据点云点的曲率来提取特征点。
    即把特别尖锐的边线点与特别平坦的平面点作为特征点。
    曲率是求取做法是同一条扫描线上取目标点左右两侧各5个点，分别与目标点的坐标作差，得到的结
    果就是目标点的曲率。当目标点处在棱或角的位置时，自然与周围点的差值较大，得到的曲率较大，
    这时属于线特征；反之当目标点在平面上时，周围点与目标点的坐标相近，得到的曲率自然较小，这时属于面特征。
    */
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    
    // 所有线束全部集合到一个点云容器中去，使用两个数组标记起始和结束
    for (int i = 0; i < N_SCANS; i++)
    {
        // 这里+5、-6是为了方便曲率计算，不计算最左边的五个和最右边的五个点云的曲率
        // 每条scan的起始标志位
        // 前5个点和后5个点都无法计算曲率，因为他们不满足左右两侧各有5个点
        scanStartInd[i] = laserCloud->size() + 5;
        *laserCloud += laserCloudScans[i];
        // 每条scan的终止标志位
        scanEndInd[i] = laserCloud->size() - 6;
    }

    // 将一帧无序点云转换成有序点云消耗的时间，这里指的是前面处理雷达数据的时间
    printf("prepare time %f \n", t_prepare.toc());
    
    /*
    C = || ∑(X - Xi)||/ ||X||
    */

    // notice: 开始计算曲率，不计算起始和最后的五个点，这里的laserCloud是有序的点云，故可以直接这样计算
    for (int i = 5; i < cloudSize - 5; i++)
    {
        // ∑(X - Xi)
        // X方向上，左边五个点的x坐标 + 右边五个点的x坐标 - 10 * 当前点的x坐标
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
        // 计算当前点的曲率，|| ∑(X - Xi)||
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        // 储存当前计算曲率的点的ID，cloudSortInd[i] = i相当于所有点的初始自然序列，每个点得到它自己的序号(索引)
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
    }


    TicToc t_pts;//计算特征提取的时间

    pcl::PointCloud<PointType> cornerPointsSharp;// 极大边线点
    pcl::PointCloud<PointType> cornerPointsLessSharp;// 次极大边线点
    pcl::PointCloud<PointType> surfPointsFlat;// 极小平面点
    pcl::PointCloud<PointType> surfPointsLessFlat;// 次极小平面
    /*
    曲率计算完成后进行特征分类，提取特征点有几点原则：
    1.为了提高效率，每条扫描线分成6个扇区，在每个扇区内，寻找曲率最大的20个点，作为次极大边线点，其中最大的2个点，同时作为极大边线点；
    2. 寻找曲率最小的4个点，作为极小平面点，剩下未被标记的点，全部作为次极小平面点。
    3. 为了防止特征点过多聚堆，每提取一个特征点（极大/次极大边线点，极小平面点），都要将这个点和它附近的点全都标记为“已选中”，
    在下次提取特征点时，将会跳过这些点。对于次极小平面点，采取降采样的方法避免过多聚堆。
    */
    float t_q_sort = 0;// 用来记录排序花费的总时间
    // 提取每一条线束上的特征
    for (int i = 0; i < N_SCANS; i++)
    {
        // 如果最后一个可算曲率的点与第一个的数量差小于6，说明无法分成6个扇区，跳过
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        // 用来存储不太平整的点
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        // 将每条scan平均分成6等份，为了使特征点均匀分布，将一个scan分成6个扇区
        for (int j = 0; j < 6; j++)
        {
            // 每一个等份的起始标志位
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
            // 每一个等份的终止标志位
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp;//计算排序时间

            // 对每一个等份中的点，根据曲率的大小排序，曲率小的在前，大的在后面
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            // t_q_sort累计每个扇区曲率排序时间总和
            t_q_sort += t_tmp.toc();
            // 选取极大边线点（2个）和次极大边线点（20个）
            int largestPickedNum = 0;

            // 遍历当前等份，因为曲率大的在后面，这里从后往前找
            for (int k = ep; k >= sp; k--)
            {
                // 排序后顺序就乱了，这个时候索引的作用就体现出来了
                // 曲率最大点的ID??
                int ind = cloudSortInd[k]; 

                // 判断当前点是否是有效点，同时对应曲率是否大于阈值
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {

                    largestPickedNum++;
                    // 选取两个曲率大的点
                    if (largestPickedNum <= 2)
                    {        
                        // 给曲率大的点打上标记               
                        cloudLabel[ind] = 2;
                        // 储存曲率大的点
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        // 曲率大的点也放在这里是为了，当前帧大曲率的点和上一帧小曲率的点建立约束
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    // 选取20个曲率稍微大的点
                    else if (largestPickedNum <= 20)
                    {                        
                        // 给曲率稍微大的点打上标签
                        cloudLabel[ind] = 1; 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else// 超过20个就跳过
                    {
                        break;
                    }
                    // 当前点被选取后，Picked被置位1
                    cloudNeighborPicked[ind] = 1;

                    // 为了保证曲率大的特征点不过度集中，将当前点的左右各五个点置位1，避免后续会选择到作为特征点
                    for (int l = 1; l <= 5; l++)
                    {
                        // 一圈是1800个点，1800/360 = 5，每1°有五个点，1/5 = 0.2，每一个点的间隔为0.2°
                        // 计算当前点与相邻点的距离
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        // 如果距离大，说明点云在此处不连续，是特征边缘，就会有新的特征，因此就不置位了
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 下面开始挑选面点，选取极小平面点（4个）
            int smallestPickedNum = 0;
            // 遍历当前等份，曲率是从小往大排序
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];
                // 确保这个点没有被pick且曲率小于阈值
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {
                    // -1认为是平坦的点
                    cloudLabel[ind] = -1;
                    // 表面平坦点容器
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    // 这里不区分平坦和比较平坦，因为剩下的点label默认是0，就是比较平坦
                    // 每等分只挑选四个曲率小的点
                    if (smallestPickedNum >= 4)
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
            // 遍历当前等份
            for (int k = sp; k <= ep; k++)
            {
                // 如果不是角点，那么认为是面点
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }

        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        // 面点多，角点少，所以对面点进行一个体素滤波
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        // 正方体的长宽高为0.2，在给定的正方体中无论有多少个点，只保留具有代表的一个重心
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        // 储存体素滤波后的点云
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());

    // 分别将当前点云、四种特征的点云发布出去可视化
    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    // 发布曲率大的角点
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    // 曲率较大的角点
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);// 平坦的面点

    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);// 比较平坦的面点

    // pub each scan
    if(PUB_EACH_LINE)// false
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    printf("scan registration time %f ms *************\n", t_whole.toc());
    // 如果在回调函数中做一次配准超过0.1秒，就会警告，发生丢帧的情况
    // 10HZ = 处理每帧只给0.1秒
    if(t_whole.toc() > 100)
        // 算力不够，发生丢帧，建图的精度不高
        ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv)
{
    // 节点名称
    ros::init(argc, argv, "scanRegistration");
    // 节点句柄，可以发布接收
    ros::NodeHandle nh;
    // 从配置文件中获取激光雷达参数，默认16
    nh.param<int>("scan_line", N_SCANS, 16);
    // 5米内的点云无效，如果不设置会产生连续不断的鬼影
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

    printf("scan line number %d \n", N_SCANS);
    // A_LOAM仅仅支持这三种类型
    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }
    // 开始订阅rosbag或者驱动发布的激光信息
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
    // 
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();

    return 0;
}
