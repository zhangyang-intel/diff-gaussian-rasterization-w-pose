/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
// 输入：当前线程处理高斯的ID，球谐函数的阶数，球谐系数的数量，所有高斯的3D位置均值，相机位置，所有3D高斯的球谐系数，被剔除的3D高斯(RGB颜色为负值)，所有3D高斯颜色的梯度，所有3D高斯的位置均值梯度，所有3D高斯球谐系数的梯度，相机位姿梯度
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs,  float *dL_dtau)
{
	// Compute intermediate values, as it is done during forward
	// 读取当前3D高斯的位置均值，计算相机位置到当前3D高斯位置均值的单位方向向量dir，即看向当前3D高斯的视线方向
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);
	// 读取当前3D高斯的球谐系数
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	// 读取当前3D高斯的RGB颜色梯度dL/dC，如果当前3D高斯的颜色被剔除，则颜色梯度为0
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);	// 当前3D高斯颜色的梯度对视线方向x分量的梯度 dC / dcir.x
	glm::vec3 dRGBdy(0, 0, 0);	// 当前3D高斯颜色的梯度对视线方向y分量的梯度 dC / dcir.y
	glm::vec3 dRGBdz(0, 0, 0);	// 当前3D高斯颜色的梯度对视线方向z分量的梯度 dC / dcir.z
	float x = dir.x;			// 视线方向向量的x分量
	float y = dir.y;			// 视线方向向量的y分量
	float z = dir.z;			// 视线方向向量的z分量

	// Target location for this Gaussian to write SH gradients to
	// 读取当前3D高斯球谐系数的梯度 dL/dSH
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	// 更新当前3D高斯球谐系数的梯度：dL / dSH = dL / dC * dC / dSH (dsh0～dsh15)
	float dRGBdsh0 = SH_C0;  			// dC / dsh0
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	// 高阶球谐系数依次计算每个系数的梯度
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;	// dC / dsh1
		float dRGBdsh2 = SH_C1 * z;		// dC / dsh2
		float dRGBdsh3 = -SH_C1 * x;	// dC / dsh3
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	// 计算了看向当前3D高斯的视线方向的梯度：dL / ddir = dL / dC * dC / ddir
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	// 根据视线方向的单位向量dir和其梯度dL / ddir，计算当前3D高斯位置均值的梯度：dL / dmean = dL / ddir * ddir / dmean
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	// 更新当前3D高斯位置均值的梯度：dL / dmean3D
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);

	// 更新相机位姿的梯度：dL / dtau
	dL_dtau[6 * idx + 0] += -dL_dmean.x;
	dL_dtau[6 * idx + 1] += -dL_dmean.y;
	dL_dtau[6 * idx + 2] += -dL_dmean.z;

}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,				// 高斯的数量
	const float3* means,							// 所有3D高斯位置均值
	const int* radii,								// 所有2D高斯的半径
	const float* cov3Ds,							// 所有3D高斯的协方差
	const float h_x, float h_y,						// 相机x和y方向的焦距
	const float tan_fovx, float tan_fovy,			// 相机的视场角
	const float* view_matrix,						// 视图矩阵T_CW
	const float* dL_dconics,						// 所有2D高斯协方差梯度
	float3* dL_dmeans,								// 所有3D高斯位置均值梯度
	float* dL_dcov,									// 所有3D高斯协方差梯度
	float *dL_dtau)									// 相机位姿的梯度
{
	// 当前线程的索引，每个线程处理一个高斯
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	// 读取当前线程的高斯的3D协方差
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	// 读取当前高斯的3D位置均值和2D协方差梯度
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	// 当前3D高斯位置均值从世界系转换到相机系
	float3 t = transformPoint4x3(mean, view_matrix);
	
	// 限制当前3D高斯在相机视锥体范围内
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	// 如果超出视锥体范围，则在乘法过程中忽略当前高斯的梯度
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	// 计算当前高斯2D协方差 cov2D = J * W * cov3D * W^T * J^T 
	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	// 只储存当前高斯2D协方差的左上角3个元素
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;			// 当前高斯2D协方差矩阵的行列式
	float dL_da = 0, dL_db = 0, dL_dc = 0;	// 当前高斯2D协方差的梯度
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);	// 当前高斯2D协方差矩阵行列式平方的倒数

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		// 当前高斯2D协方差的梯度：dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		// 当前高斯3D协方差的梯度：dL / Vrk = dL / cov2D * cov2D / Vrk
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	// 计算当前2D高斯协方差中间矩阵T上半部分的梯度：dL / dT = dL / cov2D * cov2D / dT
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	// 计算当前3D高斯协方差的Jacobian矩阵J上非零元素的梯度：dL / dJ = dL / dT * dT / dJ
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	// 计算当前3D高斯在相机系下深度的倒数、倒数的平方和倒数的立方
	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	// 计算当前3D高斯在相机系下位置均值t的梯度：dL / dmeanC = dL / dJ * dJ / dmeanC
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// 重点！求解相机位置姿态的梯度
	SE3 T_CW(view_matrix);							// 相机视图矩阵T_CW
	mat33 R = T_CW.R().data();						// 相机的姿态
	mat33 RT = R.transpose();						// 相机的姿态的转置
	float3 t_ = T_CW.t();							// 相机的位置

	mat33 dpC_drho = mat33::identity();				// 3D高斯在相机系下的位置均值对相机位置的梯度：dmeanC / dRho
	mat33 dpC_dtheta = -mat33::skew_symmetric(t);	// 3D高斯在相机系下的位置均值对相机姿态的梯度：dmeanC / dTheta
	float dL_dt[6];									// 相机位置和姿态的梯度：dL / dtau
	for (int i = 0; i < 3; i++) {
		float3 c_rho = dpC_drho.cols[i];			// dmeanC/dRho的第 i 列
		float3 c_theta = dpC_dtheta.cols[i];		// dmeanC/dTheta的第 i 列
		// 更新相机位置和姿态的梯度：dL / dtau = dL / dmeanC * dmeanC / dtau
		dL_dt[i] = dL_dtx * c_rho.x + dL_dty * c_rho.y + dL_dtz * c_rho.z;				// 相机位置的梯度
		dL_dt[i + 3] = dL_dtx * c_theta.x + dL_dty * c_theta.y + dL_dtz * c_theta.z;	// 相机姿态的梯度
	}
	// 每一个高斯累积更新相机位置和姿态的梯度dL_dtau
	for (int i = 0; i < 6; i++) {
		dL_dtau[6 * idx + i] += dL_dt[i];
	}

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	// 计算3D高斯在世界坐标系下位置均值的梯度
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	// 更新当前3D高斯在世界坐标系下位置均值的梯度：dL / dmean3D
	dL_dmeans[idx] = dL_dmean;

	// 更新相机位姿T_CW的旋转分量W的梯度：dL / dW = dL / dJ * dJ / dW
	float dL_dW00 = J[0][0] * dL_dT00;
	float dL_dW01 = J[0][0] * dL_dT01;
	float dL_dW02 = J[0][0] * dL_dT02;
	float dL_dW10 = J[1][1] * dL_dT10;
	float dL_dW11 = J[1][1] * dL_dT11;
	float dL_dW12 = J[1][1] * dL_dT12;
	float dL_dW20 = J[0][2] * dL_dT00 + J[1][2] * dL_dT10;
	float dL_dW21 = J[0][2] * dL_dT01 + J[1][2] * dL_dT11;
	float dL_dW22 = J[0][2] * dL_dT02 + J[1][2] * dL_dT12;

	// 相机姿态的旋转矩阵3列
	float3 c1 = R.cols[0];
	float3 c2 = R.cols[1];
	float3 c3 = R.cols[2];

	// 存储相机相机位姿T_CW的旋转分量W的梯度 dL / dW
	float dL_dW_data[9];
	dL_dW_data[0] = dL_dW00;
	dL_dW_data[3] = dL_dW01;
	dL_dW_data[6] = dL_dW02;
	dL_dW_data[1] = dL_dW10;
	dL_dW_data[4] = dL_dW11;
	dL_dW_data[7] = dL_dW12;
	dL_dW_data[2] = dL_dW20;
	dL_dW_data[5] = dL_dW21;
	dL_dW_data[8] = dL_dW22;

	// dL / dW的3列
	mat33 dL_dW(dL_dW_data);
	float3 dL_dWc1 = dL_dW.cols[0];
	float3 dL_dWc2 = dL_dW.cols[1];
	float3 dL_dWc3 = dL_dW.cols[2];

	// 由相机旋转矩阵3列元素构成的对角矩阵
	mat33 n_W1_x = -mat33::skew_symmetric(c1);	// 由c1构成的对角矩阵
	mat33 n_W2_x = -mat33::skew_symmetric(c2);	// 由c2构成的对角矩阵
	mat33 n_W3_x = -mat33::skew_symmetric(c3);	// 由c3构成的对角矩阵

	// 更新相机位姿的梯度：dL / dtheta = dL / dW * dW / dtheta
	float3 dL_dtheta = {};
	dL_dtheta.x = dot(dL_dWc1, n_W1_x.cols[0]) + dot(dL_dWc2, n_W2_x.cols[0]) +
				dot(dL_dWc3, n_W3_x.cols[0]);
	dL_dtheta.y = dot(dL_dWc1, n_W1_x.cols[1]) + dot(dL_dWc2, n_W2_x.cols[1]) +
				dot(dL_dWc3, n_W3_x.cols[1]);
	dL_dtheta.z = dot(dL_dWc1, n_W1_x.cols[2]) + dot(dL_dWc2, n_W2_x.cols[2]) +
				dot(dL_dWc3, n_W3_x.cols[2]);

	dL_dtau[6 * idx + 3] += dL_dtheta.x;
	dL_dtau[6 * idx + 4] += dL_dtheta.y;
	dL_dtau[6 * idx + 5] += dL_dtheta.z;


}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	// 读取当前3D高斯的旋转
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;
	// 四元数 -> 旋转矩阵
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);
	// 读取当前3D高斯的缩放
	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;	// 当前3D高斯修正后的缩放
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;
	// 读取当前3D高斯的协方差梯度
	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);	// 当前3D高斯协方差第一列
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);	// 当前3D高斯协方差第二列和第三列

	// Convert per-element covariance loss gradients to matrix form
	// 计算当前3D高斯的：dL / dSigma = dL / dCov3D
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	// 计算当前3D高斯M的梯度：dL / dM = dL / dSigma * dSigma / dM
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);			// 当前3D高斯旋转矩阵的转置
	glm::mat3 dL_dMt = glm::transpose(dL_dM);	// 当前3D高斯dL / dM的转置

	// Gradients of loss w.r.t. scale
	// 更新和储存当前3D高斯缩放的梯度：dL / dscale = dL / dM * dM / dscale
	glm::vec3* dL_dscale = dL_dscales + idx;	// 读取当前3D高斯的缩放梯度
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	// 更新当前3D高斯M的梯度
	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	// 更新当前3D高斯旋转四元数的梯度：dL / dq = dL / dM * dM / dq
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	// 存储当前3D高斯旋转四元数的梯度：dL / drot
	float4* dL_drot = (float4*)(dL_drots + idx);	// 读取当前3D高斯的旋转梯度
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,								// 3D高斯的数量，球谐函数的阶数，球谐系数的数量
	const float3* means,								// 所有3D高斯位置均值
	const int* radii,									// 所有2D高斯的半径
	const float* shs,									// 所有3D高斯的球谐系数
	const bool* clamped,								// 被剔除的3D高斯(RGB颜色为负值)
	const glm::vec3* scales,							// 所有3D高斯的缩放
	const glm::vec4* rotations,							// 所有3D高斯的旋转
	const float scale_modifier,							// 缩放的修正系数
	const float *viewmatrix,							// 相机视图矩阵T_CW
	const float* proj,									// 经过视图矩阵和投影矩阵相乘后得到的最终投影变换矩阵T_IW
	const float *proj_raw,								// 原始的投影矩阵T_IW
	const glm::vec3* campos,							// 相机位置
	const float3* dL_dmean2D,							// 2D高斯位置均值的梯度
	glm::vec3* dL_dmeans,								// 3D高斯位置均值的梯度
	float* dL_dcolor,									// 3D高斯颜色的梯度
	float *dL_ddepth,									// 3D高斯在相机系下深度的梯度
	float* dL_dcov3D,									// 3D高斯协方差的梯度
	float* dL_dsh,										// 3D高斯球谐系数的梯度
	glm::vec3* dL_dscale,								// 3D高斯缩放的梯度
	glm::vec4* dL_drot,									// 3D高斯旋转的梯度
	float *dL_dtau)										// 相机位姿的梯度
{
	// 获取当前线程的全局索引，每个线程负责一个3D高斯
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// 读取当前3D高斯的位置均值
	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	// 计算当前3D高斯位置均值在图像系下的坐标 (x,y,z) -> (x,y,z,w)
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);	// 1/w

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;	// 当前3D高斯位置均值的梯度
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	// 根据当前2D高斯位置均值的梯度计算其3D位置均值的梯度
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	// 更新当前3D高斯位置均值的梯度
	dL_dmeans[idx] += dL_dmean;

	// 重点！！！求解相机位姿的梯度
	// 计算当前3D高斯位置均值在图像系下的三个分量的导数alpha、beta和gamma
	float alpha = 1.0f * m_w;
	float beta = -m_hom.x * m_w * m_w;
	float gamma = -m_hom.y * m_w * m_w;

	// 原始的投影矩阵T_IW
	float a = proj_raw[0];
	float b = proj_raw[5];
	float c = proj_raw[10];
	float d = proj_raw[14];
	float e = proj_raw[11];

	SE3 T_CW(viewmatrix);					// 相机视图矩阵T_CW
	mat33 R = T_CW.R().data();				// 相机的姿态
	mat33 RT = R.transpose();				// 相机的姿态的转置
	float3 t = T_CW.t();					// 相机的位移
	float3 p_C = T_CW * m;					// 当前3D高斯在相机系下的位置均值
	mat33 dp_C_d_rho = mat33::identity();	// 当前3D高斯在相机系下的位置均值对相机位置的梯度：dmeanC / dRho
	mat33 dp_C_d_theta = -mat33::skew_symmetric(p_C);	// 当前3D高斯在相机系下的位置均值对相机姿态的梯度：dmeanC / dTheta

	// 计算当前3D高斯在图像系下的位置均值对其在相机系下的梯度：dmean2D / dmeanC
	float3 d_proj_dp_C1 = make_float3(alpha * a, 0.f, beta * e);
	float3 d_proj_dp_C2 = make_float3(0.f, alpha * b, gamma * e);

	// 计算当前3D高斯在图像系下的位置均值对相机位移的梯度：dmean2D / dRho = dmean2D / dmeanC * dmeanC / dRho
	float3 d_proj_dp_C1_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C1; // x.T A = A.T x
	float3 d_proj_dp_C2_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C2;
	// 计算当前3D高斯在图像系下的位置均值对相机姿态的梯度：dmean2D / dTheta = dmean2D / dmeanC * dmeanC / dTheta
	float3 d_proj_dp_C1_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C1;
	float3 d_proj_dp_C2_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C2;

	// 储存当前3D高斯在图像系下的位置均值对相机位姿的梯度：dmean2D / dtau
	float2 dmean2D_dtau[6];
	dmean2D_dtau[0].x = d_proj_dp_C1_d_rho.x;
	dmean2D_dtau[1].x = d_proj_dp_C1_d_rho.y;
	dmean2D_dtau[2].x = d_proj_dp_C1_d_rho.z;
	dmean2D_dtau[3].x = d_proj_dp_C1_d_theta.x;
	dmean2D_dtau[4].x = d_proj_dp_C1_d_theta.y;
	dmean2D_dtau[5].x = d_proj_dp_C1_d_theta.z;

	dmean2D_dtau[0].y = d_proj_dp_C2_d_rho.x;
	dmean2D_dtau[1].y = d_proj_dp_C2_d_rho.y;
	dmean2D_dtau[2].y = d_proj_dp_C2_d_rho.z;
	dmean2D_dtau[3].y = d_proj_dp_C2_d_theta.x;
	dmean2D_dtau[4].y = d_proj_dp_C2_d_theta.y;
	dmean2D_dtau[5].y = d_proj_dp_C2_d_theta.z;

	// 计算相机位姿的梯度：dL / dtau = dL / dmean2D * dmean2D / dtau
	float dL_dt[6];
	for (int i = 0; i < 6; i++) {
		dL_dt[i] = dL_dmean2D[idx].x * dmean2D_dtau[i].x + dL_dmean2D[idx].y * dmean2D_dtau[i].y;
	}
	// 更新相机位姿的梯度：dL / dtau
	for (int i = 0; i < 6; i++) {
		dL_dtau[6 * idx + i] += dL_dt[i];
	}

	// Compute gradient update due to computing depths
	// p_orig = m
	// p_view = transformPoint4x3(p_orig, viewmatrix);
	// depth = p_view.z;
	// 当前3D高斯在相机系下的深度的梯度dL / ddepth
	float dL_dpCz = dL_ddepth[idx];
	// 更新当前3D高斯位置均值的梯度：dL / dmean3D = dL / ddepth * ddepth / dmean3D
	dL_dmeans[idx].x += dL_dpCz * viewmatrix[2];
	dL_dmeans[idx].y += dL_dpCz * viewmatrix[6];
	dL_dmeans[idx].z += dL_dpCz * viewmatrix[10];

	for (int i = 0; i < 3; i++) {
		float3 c_rho = dp_C_d_rho.cols[i];			// dmeanC / dRho的第 i 列
		float3 c_theta = dp_C_d_theta.cols[i];		// dmeanC / dTheta的第 i 列
		// 更新相机位姿的梯度：dL / dtau = dL / ddepth * dmeanC.z（ddepth） / dtau
		dL_dtau[6 * idx + i] += dL_dpCz * c_rho.z;
		dL_dtau[6 * idx + i + 3] += dL_dpCz * c_theta.z;
	}



	// Compute gradient updates due to computing colors from SHs
	// 根据当前3D高斯RGB颜色的梯度dL_dcolor计算球谐系数的梯度dL_dsh和相机位姿的梯度dL_dtau
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh, dL_dtau);

	// Compute gradient updates due to computing covariance from scale/rotation
	// 根据当前3D高斯协方差的梯度计算当前3D高斯的缩放和旋转的梯度
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// 辅助函数：将每个线程的数据与相邻线程的数据相加
template <typename T>
__device__ void inline reduce_helper(int lane, int i, T *data) {
  if (lane < i) {
    data[lane] += data[lane + i];
  }
}

// CUDA kernel 函数：对多个数组进行归约求和操作
template <typename group_t, typename... Lists>
__device__ void render_cuda_reduce_sum(group_t g, Lists... lists) {
  int lane = g.thread_rank();  // 获取当前线程在线程组中的索引
  g.sync();  // 同步线程组内的所有线程

  // 循环执行归约操作，每次循环将线程组的大小减半
  for (int i = g.size() / 2; i > 0; i /= 2) {
    (...,
     reduce_helper(
         lane, i, lists)); // Fold expression: 对每个数组调用 reduce_helper 函数
    g.sync();  // 同步线程组内的所有线程，确保每次循环后所有线程都完成上一次迭代的工作
  }
}



// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,						// 每个线程块处理的像素范围
	const uint32_t* __restrict__ point_list,				// 每个像素对应的高斯点索引
	int W, int H,											// 图像宽高
	const float* __restrict__ bg_color,						// 背景颜色
	const float2* __restrict__ points_xy_image,				// 所有2D高斯的位置均值
	const float4* __restrict__ conic_opacity,				// 所有2D高斯的协方差矩阵和不透明度
	const float* __restrict__ colors,						// 所有3D高斯的颜色
	const float* __restrict__ depths,						// 所有3D高斯在相机系下的深度
	const float* __restrict__ final_Ts,						// 存储每个像素最终透光率的数组
	const uint32_t* __restrict__ n_contrib,					// 存储对每个像素颜色有贡献的高斯计数
	const float* __restrict__ dL_dpixels,					// 存储每个像素的RGB颜色梯度
	const float* __restrict__ dL_dpixels_depth,				// 存储每个像素的深度梯度
	float3* __restrict__ dL_dmean2D,						// 存储每个2D高斯均值的梯度
	float4* __restrict__ dL_dconic2D,						// 存储每个2D高斯协方差的梯度
	float* __restrict__ dL_dopacity,						// 存储每个3D高斯不透明度的梯度
	float* __restrict__ dL_dcolors,							// 存储每个3D高斯颜色的梯度
	float* __restrict__ dL_ddepths)							// 存储每个3D高斯在相机系下深度的梯度
{
	// We rasterize again. Compute necessary block info.
	// 再次栅格化，计算必要的块信息
	// 读取当前栅格block和相关的最小/最大像素范围pix_min～pix_max
	auto block = cg::this_thread_block();
	auto tid = block.thread_rank();			// 当前线程在线程块中的索引，表示像素点的索引
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };			
	
	// 当前线程像素点的像素坐标pix和像素ID
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	
	// 判断当前像素点是否在图像范围内
	const bool inside = pix.x < W&& pix.y < H;
	bool done = !inside;
	
	// 读取当前栅格中的高斯的起始和结束位置索引range，并计算需要处理的高斯数量toDo，对其进行分批次处理rounds
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// 为当前栅格中的每个线程分配共享存储空间
	__shared__ int collected_id[BLOCK_SIZE];					// 当前栅格中所有高斯的ID
	__shared__ float2 collected_xy[BLOCK_SIZE];					// 当前栅格中所有2D高斯的位置均值
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];		// 当前栅格中所有2D高斯的协方差和不透明度
	__shared__ float collected_colors[C * BLOCK_SIZE];			// 当前栅格中所有3D高斯的颜色
	__shared__ float collected_depths[BLOCK_SIZE];				// 当前栅格中所有3D高斯的深度

	__shared__ float2 dL_dmean2D_shared[BLOCK_SIZE];			// 当前栅格中所有2D高斯均值的梯度
	__shared__ float3 dL_dcolors_shared[BLOCK_SIZE];			// 当前栅格中所有像素颜色的梯度
	__shared__ float dL_ddepths_shared[BLOCK_SIZE];				// 当前栅格中所有像素深度的梯度
	__shared__ float dL_dopacity_shared[BLOCK_SIZE];			// 当前栅格中所有3D高斯不透明度的梯度
	__shared__ float4 dL_dconic2D_shared[BLOCK_SIZE];			// 当前栅格中所有2D高斯协方差的梯度

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	// 读取当前像素透光率的最终值
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing 
	// Gaussian is known from each pixel from the forward.
	// 读取对当前像素最后一个有贡献的高斯的ID
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };		// 当前像素累积的颜色（当前及其后面的高斯渲染出来的颜色）
	float dL_dpixel[C] = { 0 };		// 当前像素RGB颜色的梯度
	float accum_rec_depth = 0;		// 当前像素累积的深度（当前及其后面的高斯渲染出来的深度）
	float dL_dpixel_depth = 0;		// 当前像素的深度梯度
	if (inside) {
		#pragma unroll				// 编译器循环展开，加速
		// 读取当前像素RGB颜色和深度的梯度
		for (int i = 0; i < C; i++) {
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		}
		dL_dpixel_depth = dL_dpixels_depth[pix_id];
	}

	float last_alpha = 0.f;			// 下一个高斯的不透明度
	float last_color[C] = { 0.f };	// 下一个高斯的RGB颜色
	float last_depth = 0.f;			// 下一个高斯在相机系的深度

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	// 当前像素点的像素坐标对于相机系坐标的梯度（ndc2Pix函数）
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;
	__shared__ int skip_counter;	// 当前像素跳过的高斯数量

	// Traverse all Gaussians
	// 逐批次处理高斯，每批次处理256个高斯
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		// block.sync();
		const int progress = i * BLOCK_SIZE + tid;	// 当前线程处理的高斯索引
		// 读取当前栅格中的高斯的数据到共享内存中
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[tid] = coll_id;								// 当前高斯的ID
			collected_xy[tid] = points_xy_image[coll_id];				// 当前2D高斯的位置均值
			collected_conic_opacity[tid] = conic_opacity[coll_id];		// 当前2D高斯的协方差和不透明度
			#pragma unroll
			for (int i = 0; i < C; i++) {
				collected_colors[i * BLOCK_SIZE + tid] = colors[coll_id * C + i];	// 当前3D高斯的颜色
				
			}
			collected_depths[tid] = depths[coll_id];								// 当前3D高斯在相机系下的深度
		}
		// 遍历当前批次的高斯
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
			// 等待所有线程加载完毕
			block.sync();
			// 在每个栅格的第一个线程清空skip_counter
			if (tid == 0) {
				skip_counter = 0;
			}
			block.sync();

			// Keep track of current Gaussian ID. Skip, if this one 
			// is behind the last contributor for this pixel.
			// 跟踪当前高斯的ID，并检查是否需要跳过当前高斯
			bool skip = done;		// 当前高斯处理完毕则跳过当前高斯
			contributor = done ? contributor : contributor - 1;	// 更新对当前像素有贡献的高斯的ID
			skip |= contributor >= last_contributor;	// 如果当前高斯的ID大于最后一个有贡献的高斯的ID，则跳过当前高斯

			// Compute blending values, as before.
			// 计算当前2D高斯的位置均值和当前像素位置之间的距离power
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			skip |= power > 0.0f;	// 如果power大于0，即当前的2D高斯对于当前像素点不可见，则跳过当前高斯

			const float G = exp(power);
			// 计算当前2D高斯对于当前像素点的不透明度alpha
			const float alpha = min(0.99f, con_o.w * G);
			skip |= alpha < 1.0f / 255.0f;	// 如果不透明度小于1/255，则跳过当前高斯

			// 更新当前像素跳过的高斯数量
			if (skip) {
				atomicAdd(&skip_counter, 1);
			}
			block.sync();
			// 如果当前像素跳过的高斯数量等于当前批次的高斯数量，则跳过当前像素
			if (skip_counter == BLOCK_SIZE) {
				continue;
			}

			// 反向更新当前像素的透光率
			T = skip ? T : T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;	// 当前像素颜色通道相对于颜色参数的梯度

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;						// 当前高斯的不透明度梯度
			const int global_id = collected_id[j];		// 当前高斯的全局ID
			float local_dL_dcolors[3];					// 当前像素的颜色梯度
			#pragma unroll
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];	// 下一个高斯的颜色
				// Update last color (to be used in the next iteration)
				// 更新当前像素的颜色
				accum_rec[ch] = skip ? accum_rec[ch] : last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = skip ? last_color[ch] : c;	// 更新下一个高斯的RGB颜色

				const float dL_dchannel = dL_dpixel[ch];			// 当前像素在ch通道上的RGB颜色梯度
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;		// 更新当前高斯的不透明度梯度
				local_dL_dcolors[ch] = skip ? 0.0f : dchannel_dcolor * dL_dchannel;  // 更新当前像素ch通道的颜色梯度
			}
			// 将当前像素的颜色梯度存储到共享内存中
			dL_dcolors_shared[tid].x = local_dL_dcolors[0];
			dL_dcolors_shared[tid].y = local_dL_dcolors[1];
			dL_dcolors_shared[tid].z = local_dL_dcolors[2];

			const float depth = collected_depths[j];	// 下一个高斯的深度
			// 更新当前像素的深度
			accum_rec_depth = skip ? accum_rec_depth : last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
			last_depth = skip ? last_depth : depth;		// 更新下一个高斯的深度
			dL_dalpha += (depth - accum_rec_depth) * dL_dpixel_depth;	// 更新当前高斯的不透明度梯度
			// 将当前像素的深度梯度存储到共享内存中
			dL_ddepths_shared[tid] = skip ? 0.f : dchannel_dcolor * dL_dpixel_depth;

			// 更新当前高斯的不透明度梯度
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			// 更新下一个高斯的不透明度
			last_alpha = skip ? last_alpha : alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			// 考虑北京颜色的影响，再次更新当前高斯的不透明度梯度
			float bg_dot_dpixel = 0.f;
			#pragma unroll
			for (int i = 0; i < C; i++) {
				bg_dot_dpixel +=  bg_color[i] * dL_dpixel[i];
			}
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Helpful reusable temporary variables
			// 临时变量
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// 更新当前像素的当前高斯2D均值梯度，并存储到共享内存中
			dL_dmean2D_shared[tid].x = skip ? 0.f : dL_dG * dG_ddelx * ddelx_dx;
			dL_dmean2D_shared[tid].y = skip ? 0.f : dL_dG * dG_ddely * ddely_dy;
			// 更新当前像素的当前2D高斯协方差梯度，并存储到共享内存中
			dL_dconic2D_shared[tid].x = skip ? 0.f : -0.5f * gdx * d.x * dL_dG;
			dL_dconic2D_shared[tid].y = skip ? 0.f : -0.5f * gdx * d.y * dL_dG;
			dL_dconic2D_shared[tid].w = skip ? 0.f : -0.5f * gdy * d.y * dL_dG;
			// 更新当前像素的当前3D高斯不透明度梯度，并存储到共享内存中
			dL_dopacity_shared[tid] = skip ? 0.f : G * dL_dalpha;
			// 将共享内存中的数据相加，得到当前像素和当前高斯的最终梯度
			render_cuda_reduce_sum(block, 
				dL_dmean2D_shared,
				dL_dconic2D_shared,
				dL_dopacity_shared,
				dL_dcolors_shared, 
				dL_ddepths_shared
			);	
			// 第一个线程将当前像素和当前高斯的梯度更新到全局内存中
			if (tid == 0) {
				float2 dL_dmean2D_acc = dL_dmean2D_shared[0];
				float4 dL_dconic2D_acc = dL_dconic2D_shared[0];
				float dL_dopacity_acc = dL_dopacity_shared[0];
				float3 dL_dcolors_acc = dL_dcolors_shared[0];
				float dL_ddepths_acc = dL_ddepths_shared[0];

				atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2D_acc.x);
				atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2D_acc.y);
				atomicAdd(&dL_dconic2D[global_id].x, dL_dconic2D_acc.x);
				atomicAdd(&dL_dconic2D[global_id].y, dL_dconic2D_acc.y);
				atomicAdd(&dL_dconic2D[global_id].w, dL_dconic2D_acc.w);
				atomicAdd(&dL_dopacity[global_id], dL_dopacity_acc);
				atomicAdd(&dL_dcolors[global_id * C + 0], dL_dcolors_acc.x);
				atomicAdd(&dL_dcolors[global_id * C + 1], dL_dcolors_acc.y);
				atomicAdd(&dL_dcolors[global_id * C + 2], dL_dcolors_acc.z);
				atomicAdd(&dL_ddepths[global_id], dL_ddepths_acc);
			}
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float* projmatrix_raw,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dtau)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	// 1. 根据2D高斯协方差的梯度计算其3D高斯协方差梯度以及相机位姿的梯度
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		dL_dtau);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		projmatrix_raw,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dtau);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dpixels_depth,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dpixels_depth,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths
	);
}
