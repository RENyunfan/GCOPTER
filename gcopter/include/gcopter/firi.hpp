/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

/* This is an old version of FIRI for temporary usage here. */

#ifndef FIRI_HPP
#define FIRI_HPP

#include "lbfgs.hpp"
#include "sdlp.hpp"

#include <Eigen/Eigen>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <vector>
#include "misc/scope_timer.hpp"

namespace firi
{

    inline void chol3d(const Eigen::Matrix3d &A, // 用于表示椭圆的归一化矩阵
                       Eigen::Matrix3d &L)      // 输出是？
    {
        L(0, 0) = sqrt(A(0, 0));
        L(0, 1) = 0.0;
        L(0, 2) = 0.0;
        L(1, 0) = 0.5 * (A(0, 1) + A(1, 0)) / L(0, 0);
        L(1, 1) = sqrt(A(1, 1) - L(1, 0) * L(1, 0));
        L(1, 2) = 0.0;
        L(2, 0) = 0.5 * (A(0, 2) + A(2, 0)) / L(0, 0);
        L(2, 1) = (0.5 * (A(1, 2) + A(2, 1)) - L(2, 0) * L(1, 0)) / L(1, 1);
        L(2, 2) = sqrt(A(2, 2) - L(2, 0) * L(2, 0) - L(2, 1) * L(2, 1));
        return;
    }

    inline bool smoothedL1(const double &mu,
                           const double &x,
                           double &f,
                           double &df)
    {
        if (x < 0.0)
        {
            return false;
        }
        else if (x > mu)
        {
            f = x - 0.5 * mu;
            df = 1.0;
            return true;
        }
        else
        {
            const double xdmu = x / mu;
            const double sqrxdmu = xdmu * xdmu;
            const double mumxd2 = mu - 0.5 * x;
            f = mumxd2 * sqrxdmu * xdmu;
            df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
            return true;
        }
    }

    inline double costMVIE(void *data,
                           const Eigen::VectorXd &x,
                           Eigen::VectorXd &grad)
    {
        const int *pM = (int *)data;
        const double *pSmoothEps = (double *)(pM + 1);
        const double *pPenaltyWt = pSmoothEps + 1;
        const double *pA = pPenaltyWt + 1;

        const int M = *pM;
        const double smoothEps = *pSmoothEps;
        const double penaltyWt = *pPenaltyWt;
        // 先从 pA中取出所有多面体
        Eigen::Map<const Eigen::MatrixX3d> A(pA, M, 3);
        // 取出seed
        Eigen::Map<const Eigen::Vector3d> p(x.data());
        // 取出  L 矩阵中的非零元素 前三个是对角元素
        Eigen::Map<const Eigen::Vector3d> rtd(x.data() + 3);
        // 后三个是左下角元素
        Eigen::Map<const Eigen::Vector3d> cde(x.data() + 6);
        // 取出seed和半径的梯度
        Eigen::Map<Eigen::Vector3d> gdp(grad.data());
        Eigen::Map<Eigen::Vector3d> gdrtd(grad.data() + 3);
        Eigen::Map<Eigen::Vector3d> gdcde(grad.data() + 6);

        double cost = 0;
        gdp.setZero();
        gdrtd.setZero();
        gdcde.setZero();

        Eigen::Matrix3d L;
        L(0, 0) = rtd(0) * rtd(0) + DBL_EPSILON;
        L(0, 1) = 0.0;
        L(0, 2) = 0.0;
        L(1, 0) = cde(0);
        L(1, 1) = rtd(1) * rtd(1) + DBL_EPSILON;
        L(1, 2) = 0.0;
        L(2, 0) = cde(2);
        L(2, 1) = cde(1);
        L(2, 2) = rtd(2) * rtd(2) + DBL_EPSILON;

        const Eigen::MatrixX3d AL = A * L;
        const Eigen::VectorXd normAL = AL.rowwise().norm();
        const Eigen::Matrix3Xd adjNormAL = (AL.array().colwise() / normAL.array()).transpose();
        const Eigen::VectorXd consViola = (normAL + A * p).array() - 1.0;

        double c, dc;
        Eigen::Vector3d vec;
        for (int i = 0; i < M; ++i)
        {
            if (smoothedL1(smoothEps, consViola(i), c, dc))
            {
                cost += c;
                vec = dc * A.row(i).transpose();
                gdp += vec;
                gdrtd += adjNormAL.col(i).cwiseProduct(vec);
                gdcde(0) += adjNormAL(0, i) * vec(1);
                gdcde(1) += adjNormAL(1, i) * vec(2);
                gdcde(2) += adjNormAL(0, i) * vec(2);
            }
        }
        cost *= penaltyWt;
        gdp *= penaltyWt;
        gdrtd *= penaltyWt;
        gdcde *= penaltyWt;

        cost -= log(L(0, 0)) + log(L(1, 1)) + log(L(2, 2));
        gdrtd(0) -= 1.0 / L(0, 0);
        gdrtd(1) -= 1.0 / L(1, 1);
        gdrtd(2) -= 1.0 / L(2, 2);

        gdrtd(0) *= 2.0 * rtd(0);
        gdrtd(1) *= 2.0 * rtd(1);
        gdrtd(2) *= 2.0 * rtd(2);

        return cost;
    }

    // Each row of hPoly is defined by h0, h1, h2, h3 as
    // h0*x + h1*y + h2*z + h3 <= 0
    // R, p, r are ALWAYS taken as the initial guess
    // R is also assumed to be a rotation matrix
    inline bool maxVolInsEllipsoid(const Eigen::MatrixX4d &hPoly,
                                   Eigen::Matrix3d &R,
                                   Eigen::Vector3d &p,
                                   Eigen::Vector3d &r)
    {
        // Find the deepest interior point [ Anylitical center]
        const int M = hPoly.rows();
        Eigen::MatrixX4d Alp(M, 4);
        Eigen::VectorXd blp(M);
        Eigen::Vector4d clp, xlp;
        const Eigen::ArrayXd hNorm = hPoly.leftCols<3>().rowwise().norm();
        Alp.leftCols<3>() = hPoly.leftCols<3>().array().colwise() / hNorm;
        Alp.rightCols<1>().setConstant(1.0);
        blp = -hPoly.rightCols<1>().array() / hNorm;
        clp.setZero();
        clp(3) = -1.0;
        const double maxdepth = -sdlp::linprog<4>(clp, Alp, blp, xlp);
        if (!(maxdepth > 0.0) || std::isinf(maxdepth))
        {
            return false;
        }
        const Eigen::Vector3d interior = xlp.head<3>();

        // Prepare the data for MVIE optimization
        // Maximum Volume Inscribed Ellipsoid
        // TODO 优化变量为 2 + 3 * M 也就是每个平面三个维度 +2+int，看不懂
        uint8_t *optData = new uint8_t[sizeof(int) + (2 + 3 * M) * sizeof(double)];
        int *pM = (int *)optData;
        double *pSmoothEps = (double *)(pM + 1);
        double *pPenaltyWt = pSmoothEps + 1;
        double *pA = pPenaltyWt + 1;

        *pM = M;
        Eigen::Map<Eigen::MatrixX3d> A(pA, M, 3);
        A = Alp.leftCols<3>().array().colwise() /
            (blp - Alp.leftCols<3>() * interior).array();

        // 优化变量包括 B 和 d
        Eigen::VectorXd x(9);
        const Eigen::Matrix3d Q = R * (r.cwiseProduct(r)).asDiagonal() * R.transpose();
        Eigen::Matrix3d L;
        chol3d(Q, L);

        // seed到多边形中心的距离
        x.head<3>() = p - interior;
        // L 矩阵中的非零元素
        x(3) = sqrt(L(0, 0));
        x(4) = sqrt(L(1, 1));
        x(5) = sqrt(L(2, 2));
        x(6) = L(1, 0);
        x(7) = L(2, 1);
        x(8) = L(2, 0);

        double minCost;
        lbfgs::lbfgs_parameter_t paramsMVIE;
        paramsMVIE.mem_size = 18;
        paramsMVIE.g_epsilon = 0.0;
        paramsMVIE.min_step = 1.0e-32;
        paramsMVIE.past = 3;
        paramsMVIE.delta = 1.0e-7;
        *pSmoothEps = 1.0e-2;
        *pPenaltyWt = 1.0e+3;

        int ret = lbfgs::lbfgs_optimize(x,
                                        minCost,
                                        &costMVIE,
                                        nullptr,
                                        nullptr,
                                        optData,
                                        paramsMVIE);

        if (ret < 0)
        {
            printf("FIRI WARNING: %s\n", lbfgs::lbfgs_strerror(ret));
        }

        p = x.head<3>() + interior;
        L(0, 0) = x(3) * x(3);
        L(0, 1) = 0.0;
        L(0, 2) = 0.0;
        L(1, 0) = x(6);
        L(1, 1) = x(4) * x(4);
        L(1, 2) = 0.0;
        L(2, 0) = x(8);
        L(2, 1) = x(7);
        L(2, 2) = x(5) * x(5);
        // 构建 L矩阵之后进行SVD 可以得到 U和S，分别作为旋转矩阵和半径向量。
        Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::FullPivHouseholderQRPreconditioner> svd(L, Eigen::ComputeFullU);
        const Eigen::Matrix3d U = svd.matrixU();
        const Eigen::Vector3d S = svd.singularValues();
        if (U.determinant() < 0.0)
        {
            R.col(0) = U.col(1);
            R.col(1) = U.col(0);
            R.col(2) = U.col(2);
            r(0) = S(1);
            r(1) = S(0);
            r(2) = S(2);
        }
        else
        {
            R = U;
            r = S;
        }

        delete[] optData;

        return ret >= 0;
    }

    inline bool firi(const Eigen::MatrixX4d &bd, // 最外圈的bounding box
                     const Eigen::Matrix3Xd &pc, // 有效点云
                     const Eigen::Vector3d &a,   // 起点
                     const Eigen::Vector3d &b,   // 终点
                     Eigen::MatrixX4d &hPoly,    // 输出多面体
                     const int iterations = 4,   // IRIS迭代次数
                     const double epsilon = 1.0e-6) // 凸优化近似精度
    {
        TimeConsuming t__("FIRI");
        const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);
        const Eigen::Vector4d bh(b(0), b(1), b(2), 1.0);

        // 如果起点和终点不在BD中，直接失败退出
        if ((bd * ah).maxCoeff() > 0.0 ||
            (bd * bh).maxCoeff() > 0.0)
        {
            return false;
        }
        // M个多边形约束 N个障碍物点
        const int M = bd.rows();
        const int N = pc.cols();

        // R初始化为一个3x3的单位的矩阵
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        // p 初始化seed为起点和终点的中点 也就是椭球的球心
        Eigen::Vector3d p = 0.5 * (a + b);
        // TODO  r初始化为一个1向量
        Eigen::Vector3d r = Eigen::Vector3d::Ones();
        // TODO 向前的H？为什么初始化为（M+N） * 4
        Eigen::MatrixX4d forwardH(M + N, 4);
        // TODO 盲猜是超平面的个数
        int nH = 0;

        // 设置最大迭代次数为4次
        for (int loop = 0; loop < iterations; ++loop)
        {
            // 这里Forward和Backward应该是转化椭球坐标系的地方。
            // C^{-1} = forward
            const Eigen::Matrix3d forward = r.cwiseInverse().asDiagonal() * R.transpose();
            // C = R * r
            const Eigen::Matrix3d backward = R * r.asDiagonal();
            // forwardB应该是提供了边界条件转换的矩阵，这就成了球坐标系下的两个范围
            // TODO 为什么要把 bd左边诚意backward呢。 这个 bd * C 是个什么概念
            const Eigen::MatrixX3d forwardB = bd.leftCols<3>() * backward;
            // bd的最后一列 + 前三列 6x3 * 3x1 得到一个6*1的边界距离
            const Eigen::VectorXd forwardD = bd.rightCols<1>() + bd.leftCols<3>() * p;

            // 将障碍物点云全部转化为了球坐标系
            const Eigen::Matrix3Xd forwardPC = forward * (pc.colwise() - p);
            const Eigen::Vector3d fwd_a = forward * (a - p);
            const Eigen::Vector3d fwd_b = forward * (b - p);
            // TODO cwiseQuotient是不是逐元素求除法
            // forwardB的rowwise是六行边界条件
            // forwardD的cwise是
            const Eigen::VectorXd distDs = forwardD.cwiseAbs().cwiseQuotient(forwardB.rowwise().norm());
            // 初始化切平面 应该是Ax + By + Cz +D  <=0
            Eigen::MatrixX4d tangents(N, 4);
            // TODO
            Eigen::VectorXd distRs(N);

            // 遍历所有障碍物点
            for (int i = 0; i < N; i++)
            {
                // 距离在球坐标系中就是到原点的距离
                distRs(i) = forwardPC.col(i).norm();
                // 所有点的切平面 N x 4 中，该点的最后一列就是距离的相反数
                tangents(i, 3) = -distRs(i);
                // 该点的前三位数就是自己的方向
                tangents.block<1, 3>(i, 0) = forwardPC.col(i).transpose() / distRs(i);
                // 如果fwd_a 在当前障碍点的切线上方
                if (tangents.block<1, 3>(i, 0).dot(fwd_a) + tangents(i, 3) > epsilon)
                {
                    // 那么这个切线就应该往外扩展到a之外
                    const Eigen::Vector3d delta = forwardPC.col(i) - fwd_a;
                    tangents.block<1, 3>(i, 0) = fwd_a - (delta.dot(fwd_a) / delta.squaredNorm()) * delta;
                    distRs(i) = tangents.block<1, 3>(i, 0).norm();
                    tangents(i, 3) = -distRs(i);
                    tangents.block<1, 3>(i, 0) /= distRs(i);
                }

                if (tangents.block<1, 3>(i, 0).dot(fwd_b) + tangents(i, 3) > epsilon)
                {
                    const Eigen::Vector3d delta = forwardPC.col(i) - fwd_b;
                    tangents.block<1, 3>(i, 0) = fwd_b - (delta.dot(fwd_b) / delta.squaredNorm()) * delta;
                    distRs(i) = tangents.block<1, 3>(i, 0).norm();
                    tangents(i, 3) = -distRs(i);
                    tangents.block<1, 3>(i, 0) /= distRs(i);
                }

                if (tangents.block<1, 3>(i, 0).dot(fwd_a) + tangents(i, 3) > epsilon)
                {
                    tangents.block<1, 3>(i, 0) = (fwd_a - forwardPC.col(i)).cross(fwd_b - forwardPC.col(i)).normalized();
                    tangents(i, 3) = -tangents.block<1, 3>(i, 0).dot(fwd_a);
                    tangents.row(i) *= tangents(i, 3) > 0.0 ? -1.0 : 1.0;
                }
            }
            // 初始化Flag分别为 M * 1 和 N * 1
            Eigen::Matrix<uint8_t, -1, 1> bdFlags = Eigen::Matrix<uint8_t, -1, 1>::Constant(M, 1);
            Eigen::Matrix<uint8_t, -1, 1> pcFlags = Eigen::Matrix<uint8_t, -1, 1>::Constant(N, 1);

            nH = 0;

            bool completed = false;
            int bdMinId, pcMinId;
            // 所有边界中距离最小的
            double minSqrD = distDs.minCoeff(&bdMinId);
            // 所有障碍物点云中距离最小的
            double minSqrR = distRs.minCoeff(&pcMinId);
            for (int i = 0; !completed && i < (M + N); ++i)
            {
                // 如果边界比障碍物还近
                if (minSqrD < minSqrR)
                {
                    // (M+N) * 4 第一个面就用障碍物点生成
                    forwardH.block<1, 3>(nH, 0) = forwardB.row(bdMinId);
                    // 最后一列是D
                    forwardH(nH, 3) = forwardD(bdMinId);
                    // 然后disable掉这个边界
                    bdFlags(bdMinId) = 0;
                }
                else // 否则说明障碍物近
                {
                    // 直接取出这障碍物点的法
                    forwardH.row(nH) = tangents.row(pcMinId);
                    // 然后disable掉这个点
                    pcFlags(pcMinId) = 0;
                }
                // 假设完成
                completed = true;
                minSqrD = INFINITY;
                for (int j = 0; j < M; ++j)
                {
                    // 访问所有平面边界 重新更新切线
                    if (bdFlags(j))
                    {
                        completed = false;
                        if (minSqrD > distDs(j))
                        {
                            bdMinId = j;
                            minSqrD = distDs(j);
                        }
                    }
                }
                // 访问所有障碍物点 重新更新最近点
                minSqrR = INFINITY;
                for (int j = 0; j < N; ++j)
                {
                    if (pcFlags(j))
                    {
                        if (forwardH.block<1, 3>(nH, 0).dot(forwardPC.col(j)) + forwardH(nH, 3) > -epsilon)
                        {
                            pcFlags(j) = 0;
                        }
                        else
                        {
                            completed = false;
                            if (minSqrR > distRs(j))
                            {
                                pcMinId = j;
                                minSqrR = distRs(j);
                            }
                        }
                    }
                }
                ++nH;
            }
            // 更新多面体的尺寸
            hPoly.resize(nH, 4);
            for (int i = 0; i < nH; ++i)
            {
                // 转回到世界坐标系
                hPoly.block<1, 3>(i, 0) = forwardH.block<1, 3>(i, 0) * forward;
                hPoly(i, 3) = forwardH(i, 3) - hPoly.block<1, 3>(i, 0).dot(p);
            }

            if (loop == iterations - 1)
            {
                break;
            }
            //  牛顿法求解最大椭圆
            maxVolInsEllipsoid(hPoly, R, p, r);
        }

        return true;
    }

}

#endif
